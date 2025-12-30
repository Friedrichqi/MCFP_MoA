from __future__ import annotations

import asyncio, json, os, subprocess, time, uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
from pydantic import BaseModel, Field

from MoA_Scheduler import GPUInfo, ModelInfo, Request, moa_scheduler, SimVLLMController

def discover_gpus(alpha: float) -> List[GPUInfo]:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,memory.total", "--format=csv,noheader,nounits"],
            text=True,
        ).strip()
        gpus: List[GPUInfo] = []
        for line in out.splitlines():
            idx_s, mem_mib_s = [x.strip() for x in line.split(",")]
            gpus.append(GPUInfo(int(idx_s), int(mem_mib_s) * 1024 * 1024, alpha))
        if not gpus:
            raise RuntimeError("no GPUs from nvidia-smi")
        return sorted(gpus, key=lambda g: g.gpu_id)
    except Exception:
        # dev fallback (CPU machine)
        fake_gb = float(os.getenv("FAKE_GPU_VRAM_GB", "24"))
        return [GPUInfo(0, int(fake_gb * (1024**3)), alpha)]


def build_hierarchical_sets(gpu_ids: List[int]) -> List[Tuple[int, ...]]:
    """
    Hierarchical sets: TP 1/2/4/8/...
    """
    sets: Set[Tuple[int, ...]] = set((gid,) for gid in gpu_ids)
    n, size = len(gpu_ids), 2
    while size <= n:
        for start in range(0, n, size):
            chunk = gpu_ids[start : start + size]
            if len(chunk) == size:
                sets.add(tuple(chunk))
        size *= 2
    return sorted(sets, key=lambda s: (len(s), s))


def load_model_card(path: str) -> Dict[str, ModelInfo]:
    """
    model_card.json format:
    {
      "models": {
        "M1": {"size_gb": 14, "tp_min": 1, "t_wake_s": 0.7, "t_load_s": 3.5, "avg_service_s": 0.25},
        ...
      }
    }
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    models: Dict[str, ModelInfo] = {}
    for name, cfg in data.items():
        size_gb = float(cfg["size_gb"])
        models[name] = ModelInfo(
            name=name,
            size_bytes=int(size_gb * (1024**3)),
            tp_min=int(cfg["tp_min"]),
            t_wake_s=float(cfg.get("t_wake_s", 1.0)),
            t_load_s=float(cfg.get("t_load_s", 5.0)),
            avg_service_s=float(cfg.get("avg_service_s", 0.2)),
        )
    if not models:
        raise ValueError("model_card.json missing or empty 'models'")
    return models


# -----------------------
# Job graph
# -----------------------

@dataclass
class JobGraph:
    job_id: str
    graph: Dict[str, List[str]]
    inputs: Dict[str, Any] = field(default_factory=dict)

    nodes: Set[str] = field(default_factory=set)
    indegree: Dict[str, int] = field(default_factory=dict)
    parents: Dict[str, List[str]] = field(default_factory=dict)

    discovered: Set[str] = field(default_factory=set)
    completed: Set[str] = field(default_factory=set)
    outputs: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        all_nodes: Set[str] = set(self.graph.keys())
        for u, vs in self.graph.items():
            for v in vs:
                all_nodes.add(v)
        for n in list(all_nodes):
            self.graph.setdefault(n, [])
        self.nodes = all_nodes

        self.indegree = {n: 0 for n in self.nodes}
        self.parents = {n: [] for n in self.nodes}
        for u, vs in self.graph.items():
            for v in vs:
                self.indegree[v] += 1
                self.parents[v].append(u)

        self._validate_dag()

    def _validate_dag(self) -> None:
        indeg = dict(self.indegree)
        q = [n for n, d in indeg.items() if d == 0]
        seen = 0
        while q:
            n = q.pop()
            seen += 1
            for v in self.graph.get(n, []):
                indeg[v] -= 1
                if indeg[v] == 0:
                    q.append(v)
        if seen != len(self.nodes):
            raise ValueError("graph contains a cycle (not a DAG)")

    def done(self) -> bool:
        return len(self.completed) == len(self.nodes)

    def payload(self, node: str) -> Any:
        base = self.inputs.get(node, {})
        parent_out = {p: self.outputs[p] for p in self.parents[node]}
        return {"node": node, "base": base, "parents": parent_out}

    def make_req(self, node: str) -> Request:
        return Request(
            job_id=self.job_id,
            node_id=node,
            model=node,  # vertex == model name
            t_arr=time.monotonic(),
            indegree=self.indegree[node],
            succ=list(self.graph.get(node, [])),
        )


# -----------------------
# FastAPI
# -----------------------

class SubmitBody(BaseModel):
    graph: Dict[str, List[str]] = Field(...)
    inputs: Optional[Dict[str, Any]] = Field(default=None)
    job_id: Optional[str] = Field(default=None)


class SubmitResp(BaseModel):
    job_id: str
    sources: List[str]
    num_nodes: int
    gpu_sets: List[List[int]]


class StatusResp(BaseModel):
    job_id: str
    done: bool
    completed: List[str]
    pending: List[str]
    outputs: Dict[str, Any]


ALPHA = float(os.getenv("ALPHA", "0.4"))
BETA = float(os.getenv("BETA", "1.0"))
INTERVAL_S = float(os.getenv("SCHED_INTERVAL_S", "1.0"))
MODEL_CARD_PATH = os.getenv("MODEL_CARD_PATH", "model_card.json")

scheduler: Optional[moa_scheduler] = None
jobs: Dict[str, JobGraph] = {}


@asynccontextmanager
async def lifespan() -> None:
    global scheduler
    gpus = discover_gpus(ALPHA)
    sets = build_hierarchical_sets([g.gpu_id for g in gpus])
    models = load_model_card(MODEL_CARD_PATH)

    scheduler = moa_scheduler(
        gpus=gpus,
        gpu_sets=sets,
        models=models,
        alpha=ALPHA,
        beta=BETA,
        controller=SimVLLMController(models),  # replace with real vLLM controller adapter
    )

    asyncio.create_task(_sched_loop())

app = FastAPI(title="MoA Proxy Server", version="0.1", lifespan=lifespan)

async def _sched_loop() -> None:
    assert scheduler is not None
    while True:
        # move indegree==0 from POTENTIAL -> WAITING
        await scheduler.move_ready_potential_to_waiting()
        # choose policies & dispatch/switch
        await scheduler.step()
        await asyncio.sleep(INTERVAL_S)


@app.post("/submit_graph", response_model=SubmitResp)
async def submit_graph(body: SubmitBody) -> SubmitResp:
    if scheduler is None:
        raise HTTPException(status_code=503, detail="scheduler not ready")

    job_id = body.job_id or str(uuid.uuid4())
    if job_id in jobs:
        raise HTTPException(status_code=409, detail="job_id already exists")

    try:
        job = JobGraph(job_id=job_id, graph=body.graph, inputs=body.inputs or {})
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    unknown = [n for n in job.nodes if n not in scheduler.models]
    if unknown:
        raise HTTPException(status_code=400, detail=f"unknown model(s): {unknown}")

    jobs[job_id] = job

    sources = [n for n, d in job.indegree.items() if d == 0]
    for n in sources:
        req = job.make_req(n)
        req.on_dispatched = lambda r, job=job: _on_dispatched(job, r)
        req.on_completed = lambda r, out, job=job: _on_completed(job, r, out)
        job.discovered.add(n)
        await scheduler.add_to_potential(req)

    return SubmitResp(
        job_id=job_id,
        sources=sources,
        num_nodes=len(job.nodes),
        gpu_sets=[list(s) for s in scheduler.gpu_sets],
    )


@app.get("/status/{job_id}", response_model=StatusResp)
async def status(job_id: str) -> StatusResp:
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="job not found")
    return StatusResp(
        job_id=job.job_id,
        done=job.done(),
        completed=sorted(job.completed),
        pending=sorted(list(job.nodes - job.completed)),
        outputs=job.outputs if job.done() else {},
    )


# -----------------------
# Callbacks
# -----------------------

def _on_dispatched(job: JobGraph, req: Request) -> List[Request]:
    """
    Called synchronously by moa_scheduler while it holds its lock.

    Returns:
      - newly discovered successor Requests to insert into POTENTIAL immediately.
    """
    # Build payload for this node (parents already done when indegree==0)
    req.payload = job.payload(req.node_id)

    newly: List[Request] = []
    now = time.monotonic()
    for succ in job.graph.get(req.node_id, []):
        if succ in job.discovered:
            continue
        job.discovered.add(succ)
        sreq = job.make_req(succ)
        sreq.t_arr = now
        sreq.on_dispatched = lambda r, job=job: _on_dispatched(job, r)
        sreq.on_completed = lambda r, out, job=job: _on_completed(job, r, out)
        newly.append(sreq)

    return newly


def _on_completed(job: JobGraph, req: Request, output: Any) -> None:
    """
    Called when a node finishes:
      - record output
      - decrement indegree of successors
      - update scheduler's indegree view (so they can move POTENTIAL->WAITING)
    """
    assert scheduler is not None
    node = req.node_id
    job.completed.add(node)
    job.outputs[node] = output

    for succ in job.graph.get(node, []):
        job.indegree[succ] = max(0, job.indegree[succ] - 1)
        asyncio.create_task(scheduler.update_indegree(job.job_id, succ, job.indegree[succ]))
