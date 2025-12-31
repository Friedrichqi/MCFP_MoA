from __future__ import annotations

import asyncio, json, os, subprocess, time, uuid, math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
from pydantic import BaseModel, Field
from huggingface_hub import HfApi, hf_hub_download

import sys, pathlib as _pathlib
sys.path.append(str(_pathlib.Path(__file__).resolve().parent))

from MoA_Scheduler import GPUInfo, ModelInfo, Request, moa_scheduler, SimVLLMController, RealVLLMController


def discover_gpus(alpha: float) -> List[GPUInfo]:
    out = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,memory.total", "--format=csv,noheader,nounits"],
        text=True,
    ).strip()
    gpus: List[GPUInfo] = []
    for line in out.splitlines():
        idx_s, mem_mib_s = [x.strip() for x in line.split(",")]
        gpus.append(GPUInfo(int(idx_s), int(mem_mib_s), alpha))
    if not gpus:
        raise RuntimeError("no GPUs from nvidia-smi")
    return sorted(gpus, key=lambda g: g.gpu_id)


def build_hierarchical_sets(gpu_ids: List[int]) -> List[Tuple[int, ...]]:
    """Hierarchical sets: TP 1/2/4/8/... aligned by index."""
    sets: Set[Tuple[int, ...]] = set((gid,) for gid in gpu_ids)
    n, size = len(gpu_ids), 2
    while size <= n:
        for start in range(0, n, size):
            chunk = gpu_ids[start : start + size]
            if len(chunk) == size:
                sets.add(tuple(chunk))
        size *= 2
    return sorted(sets, key=lambda s: (len(s), s))


def read_model_card(path: str) -> Dict[str, Dict[str, Any]]:
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f) or {}


def write_model_card(path: str, data: Dict[str, Dict[str, Any]]) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, path)


def load_model_card(path: str) -> Dict[str, ModelInfo]:
    """model_card.json format:
    {
      "repo_or_name": {
        "tp_min": 1,
        "t_wake_s": 2,
        "t_load_s": 90,
        "t_offload_s": 3,
        "slept_mem_mib_tp1": 2048,
        "slept_mem_mib_tpgt1": 4096,
        "avg_service_s": 0.2,
        "size_gb": 14            # optional (only used to guess tp_min when missing)
      },
      ...
    }
    """
    data = read_model_card(path)
    models: Dict[str, ModelInfo] = {}
    for name, cfg in data.items():
        tp_min = int(cfg.get("tp_min", 1))
        models[name] = ModelInfo(
            name=name,
            tp_min=tp_min,
            t_wake_s=float(cfg.get("t_wake_s", 2.0)),
            t_load_s=float(cfg.get("t_load_s", 90.0)),
            t_offload_s=float(cfg.get("t_offload_s", 3.0)),
            slept_mem_mib_tp1=float(cfg.get("slept_mem_mib_tp1", 2048.0)),
            slept_mem_mib_tpgt1=float(cfg.get("slept_mem_mib_tpgt1", 4096.0)),
            avg_service_s=float(cfg.get("avg_service_s", 0.2)),
        )
    return models


def hf_guess_model_size_gb(repo_id: str) -> float:
    '''
    Best-effort model size estimate from HF repo metadata.
    Prefers safetensors/bin weights sizes; falls back to index.json metadata.
    '''
    api = HfApi()

    # 1) sum sibling sizes
    try:
        info = api.repo_info(repo_id=repo_id, repo_type="model")
        total = 0
        for s in info.siblings:
            if s.size is None:
                continue
            fn = s.rfilename.lower()
            if fn.endswith(".safetensors") or fn.endswith(".bin"):
                total += int(s.size)
        if total > 0:
            return float(total) / float(1024**3)
    except Exception:
        pass

    # 2) index file metadata
    for idx_name in ("model.safetensors.index.json", "pytorch_model.bin.index.json"):
        try:
            p = hf_hub_download(repo_id=repo_id, filename=idx_name)
            with open(p, "r", encoding="utf-8") as f:
                j = json.load(f)
            total_size = j.get("metadata", {}).get("total_size", 0)
            if total_size:
                return float(total_size) / float(1024**3)
        except Exception:
            continue

    raise RuntimeError(f"Couldn't determine model size for {repo_id} from Hugging Face")


def choose_tp_min_homogeneous(size_gb: float, gpus: List[GPUInfo]) -> int:
    if not gpus:
        return 1
    min_vram_mib = min(g.vram_total_mib for g in gpus)
    if min_vram_mib <= 0:
        return 1
    size_mib = float(size_gb) * 1024.0
    raw = int(math.ceil(size_mib / min_vram_mib))
    raw = max(1, raw)
    tp = 1
    while tp < raw:
        tp <<= 1
    return min(tp, len(gpus))


async def ensure_model(scheduler: moa_scheduler, model_id: str, gpus: List[GPUInfo], model_card_path: str) -> None:
    '''
    If model isn't in scheduler.models, pull size_gb from HuggingFace, choose tp_min, set defaults:
      t_wake_s=2, t_load_s=90, t_offload_s=3.
    Also persist to model_card.json for next runs.
    '''
    if model_id in scheduler.models:
        return

    size_gb = hf_guess_model_size_gb(model_id)
    tp_min = choose_tp_min_homogeneous(size_gb, gpus)

    scheduler.models[model_id] = ModelInfo(
        name=model_id,
        tp_min=tp_min,
        t_wake_s=2.0,
        t_load_s=90.0,
        t_offload_s=3.0,
        slept_mem_mib_tp1=2048.0,
        slept_mem_mib_tpgt1=4096.0,
        avg_service_s=0.2,
    )

    data = read_model_card(model_card_path)
    data[model_id] = {
        "tp_min": tp_min,
        "t_wake_s": 2.0,
        "t_load_s": 90.0,
        "t_offload_s": 3.0,
        "slept_mem_mib_tp1": 2048.0,
        "slept_mem_mib_tpgt1": 4096.0,
        "avg_service_s": 0.2,
    }
    if size_gb is not None:
        data[model_id]["size_gb"] = size_gb
    write_model_card(model_card_path, data)


# -----------------------
# Job graph (supports repeated models via node_models mapping)
# -----------------------

@dataclass
class JobGraph:
    job_id: str
    graph: Dict[str, List[str]] = field(default_factory=dict)           # node_id -> [node_id]
    node_models: Dict[str, str] = field(default_factory=dict)           # node_id -> model_id
    inputs: Dict[str, Any] = field(default_factory=dict)                # node_id -> base input

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

        if not self.node_models:
            self.node_models = {n: n for n in self.nodes}
        else:
            for n in self.nodes:
                if n not in self.node_models:
                    raise ValueError(f"node_models missing mapping for node '{n}'")

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

    def model_of(self, node: str) -> str:
        return self.node_models[node]

    def payload(self, node: str) -> Any:
        base = self.inputs.get(node, {})
        parent_out = {p: self.outputs[p] for p in self.parents[node]}
        return {"node_id": node, "model": self.model_of(node), "base": base, "parents": parent_out}

    def make_req(self, node: str) -> Request:
        return Request(
            job_id=self.job_id,
            node_id=node,
            model=self.model_of(node),
            t_arr=time.monotonic(),
            indegree=self.indegree[node],
            succ=list(self.graph.get(node, [])),
        )


# -----------------------
# FastAPI
# -----------------------

class SubmitBody(BaseModel):
    graph: Dict[str, List[str]] = Field(..., description="Adjacency list over node_ids (stages).")
    node_models: Optional[Dict[str, str]] = Field(
        default=None,
        description="Optional map node_id -> model_id. If omitted, node_id is treated as model_id."
    )
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


class RegisterInstanceBody(BaseModel):
    gpu_set: List[int]
    model: str
    pid_by_gpu: Dict[str, int] = Field(..., description="Map gpu_id (string) -> pid")
    metrics_url: Optional[str] = Field(default=None)


ALPHA = float(os.getenv("ALPHA", "0.4"))
BETA = float(os.getenv("BETA", "1.0"))
INTERVAL_S = float(os.getenv("SCHED_INTERVAL_S", "1.0"))
MODEL_CARD_PATH = os.getenv("MODEL_CARD_PATH", "model_card.json")
DEFAULT_VLLM_METRICS_URL = os.getenv("VLLM_METRICS_URL", "http://127.0.0.1:8001/metrics")
SIMULATE = os.getenv("SIMULATE", "1") == "1"

scheduler: Optional[moa_scheduler] = None
jobs: Dict[str, JobGraph] = {}
gpu_snapshot: List[GPUInfo] = []


@asynccontextmanager
async def lifespan(app: FastAPI):
    global scheduler, gpu_snapshot

    gpu_snapshot = discover_gpus(ALPHA)
    sets = build_hierarchical_sets([g.gpu_id for g in gpu_snapshot])

    models = load_model_card(MODEL_CARD_PATH)

    controller = SimVLLMController(models) if SIMULATE else RealVLLMController(default_metrics_url=DEFAULT_VLLM_METRICS_URL)

    scheduler = moa_scheduler(
        gpus=gpu_snapshot,
        gpu_sets=sets,
        models=models,
        beta=BETA,
        controller=controller,
    )

    asyncio.create_task(_sched_loop())
    yield


app = FastAPI(title="MoA Proxy Server", version="0.3", lifespan=lifespan)


async def _sched_loop() -> None:
    assert scheduler is not None
    while True:
        await scheduler.move_ready_potential_to_waiting()
        await scheduler.step()
        await asyncio.sleep(INTERVAL_S)


@app.post("/register_instance")
async def register_instance(body: RegisterInstanceBody) -> Dict[str, Any]:
    if scheduler is None:
        raise HTTPException(status_code=503, detail="scheduler not ready")

    gpu_set = tuple(int(x) for x in body.gpu_set)
    pid_by_gpu = {int(k): int(v) for k, v in body.pid_by_gpu.items()}

    if body.model not in scheduler.models:
        await ensure_model(scheduler, body.model, gpu_snapshot, MODEL_CARD_PATH, alpha=ALPHA)

    try:
        await scheduler.register_active_instance(gpu_set, body.model, pid_by_gpu, metrics_url=body.metrics_url or "")
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    return {"ok": True, "gpu_set": list(gpu_set), "model": body.model}


@app.post("/submit_graph", response_model=SubmitResp)
async def submit_graph(body: SubmitBody) -> SubmitResp:
    if scheduler is None:
        raise HTTPException(status_code=503, detail="scheduler not ready")

    job_id = body.job_id or str(uuid.uuid4())
    if job_id in jobs:
        raise HTTPException(status_code=409, detail="job_id already exists")

    try:
        job = JobGraph(job_id=job_id, graph=body.graph, node_models=body.node_models or {}, inputs=body.inputs or {})
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    unique_models = sorted({job.model_of(n) for n in job.nodes})
    for mid in unique_models:
        if mid not in scheduler.models:
            await ensure_model(scheduler, mid, gpu_snapshot, MODEL_CARD_PATH, alpha=ALPHA)

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
