"""
online_proxy_server.py - FastAPI server for the MCFP MoA scheduling system.

Coordinates both schedulers:
- Request Scheduler (runs every δt1 = 1s)
- GPU Scheduler (runs every δt2 = 5s)

Provides REST API for:
- Job submission (DAG graphs)
- Job status queries
- Instance management
- System monitoring
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import subprocess
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# Local imports
from entities import (
    GPUInfo, GPUState, ModelCard, Instance, InstState,
    Request, ReqState, DrainLatency, Residence
)
from config import SchedulerConfig, read_model_cards, write_model_cards, ensure_model_card
from vllm_controller import ManagedVLLMController
from request_scheduler import RequestScheduler, JobGraph
from gpu_scheduler import GPUScheduler


# -----------------------------
# Logging with Blue Color
# -----------------------------

class BlueFormatter(logging.Formatter):
    """Custom formatter that outputs log messages in blue."""
    BLUE = "\033[34m"
    RESET = "\033[0m"
    
    def format(self, record):
        message = super().format(record)
        return f"{self.BLUE}{message}{self.RESET}"


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

# Apply blue formatter to root handler
for handler in logging.root.handlers:
    handler.setFormatter(BlueFormatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))

logger = logging.getLogger(__name__)


# -----------------------------
# GPU Discovery
# -----------------------------

def discover_gpus(alpha: float) -> List[GPUInfo]:
    """Discover GPUs via nvidia-smi."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,memory.total", "--format=csv,noheader,nounits"],
            text=True,
        ).strip()
    except Exception as e:
        logger.error(f"Failed to discover GPUs: {e}")
        raise RuntimeError("No GPUs available (nvidia-smi failed)")
    
    gpus: List[GPUInfo] = []
    for line in out.splitlines():
        parts = [x.strip() for x in line.split(",")]
        if len(parts) == 2:
            gpu_id = int(parts[0])
            vram_MB = int(float(parts[1]))
            gpus.append(GPUInfo(
                gpu_id=gpu_id,
                vram_total_MB=vram_MB,
                alpha=alpha,
                state=GPUState.STABLE,
            ))
    
    if not gpus:
        raise RuntimeError("No GPUs discovered")
    
    logger.info(f"Discovered {len(gpus)} GPUs: {[g.gpu_id for g in gpus]}")
    return sorted(gpus, key=lambda g: g.gpu_id)


def build_gpu_sets(gpu_ids: List[int]) -> List[Tuple[int, ...]]:
    """Build hierarchical GPU sets for tensor parallelism (1, 2, 4, 8, ...)."""
    sets: Set[Tuple[int, ...]] = set()
    
    # Single GPU sets
    for gid in gpu_ids:
        sets.add((gid,))
    
    # Multi-GPU sets (aligned)
    n = len(gpu_ids)
    size = 2
    while size <= n:
        for start in range(0, n, size):
            chunk = gpu_ids[start:start + size]
            if len(chunk) == size:
                sets.add(tuple(chunk))
        size *= 2
    
    return sorted(sets, key=lambda s: (len(s), s))


# -----------------------------
# Model Size Estimation
# -----------------------------

def hf_guess_model_size_MB(repo_id: str) -> float:
    """Estimate model size from HuggingFace."""
    try:
        from huggingface_hub import HfApi, hf_hub_download
        
        api = HfApi()
        info = api.repo_info(repo_id=repo_id, repo_type="model")
        
        total = 0
        for s in info.siblings:
            if s.size is None:
                continue
            fn = s.rfilename.lower()
            if fn.endswith(".safetensors") or fn.endswith(".bin"):
                total += int(s.size)
        
        if total > 0:
            return float(total) / (1024 ** 2)
        
        # Try index files
        for idx_name in ("model.safetensors.index.json", "pytorch_model.bin.index.json"):
            try:
                p = hf_hub_download(repo_id=repo_id, filename=idx_name)
                with open(p, "r") as f:
                    j = json.load(f)
                total_size = j.get("metadata", {}).get("total_size", 0)
                if total_size:
                    return float(total_size) / (1024 ** 2)
            except Exception:
                continue
        
        raise RuntimeError(f"Couldn't determine size for {repo_id}")
        
    except Exception as e:
        logger.warning(f"Failed to get model size for {repo_id}: {e}")
        return 4096.0  # Default fallback


def choose_tp_min(size_MB: float, gpus: List[GPUInfo]) -> int:
    """Choose minimum tensor-parallel degree based on model size and GPU memory."""
    if not gpus:
        return 1
    
    min_vram = min(g.vram_total_MB for g in gpus)
    if min_vram <= 0:
        return 1
    
    raw = int(math.ceil(size_MB / min_vram))
    raw = max(1, raw)
    
    # Round up to power of 2
    tp = 1
    while tp < raw:
        tp *= 2
    
    return min(tp, len(gpus))


# -----------------------------
# Pydantic Models
# -----------------------------

class SubmitBody(BaseModel):
    """Request body for submitting a DAG job."""
    graph: Dict[str, List[str]] = Field(..., description="Adjacency list: node_id -> [successor_ids]")
    node_models: Optional[Dict[str, str]] = Field(default=None, description="node_id -> model_id mapping")
    inputs: Optional[Dict[str, Any]] = Field(default=None, description="node_id -> base input")
    job_id: Optional[str] = Field(default=None, description="Optional custom job ID")


class SubmitResponse(BaseModel):
    """Response for job submission."""
    job_id: str
    sources: List[str]
    num_nodes: int
    gpu_sets: List[List[int]]


class StatusResponse(BaseModel):
    """Response for job status query."""
    job_id: str
    done: bool
    completed: List[str]
    pending: List[str]
    outputs: Dict[str, Any]


class RegisterInstanceBody(BaseModel):
    """Request body for registering an external vLLM instance."""
    gpu_set: List[int]
    model: str
    port: int
    pid_by_gpu: Dict[str, int] = Field(..., description="gpu_id -> pid mapping")


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    uptime_s: float
    gpus: int
    active_instances: int
    pending_requests: int


# -----------------------------
# Global State
# -----------------------------

config: Optional[SchedulerConfig] = None
controller: Optional[ManagedVLLMController] = None
request_scheduler: Optional[RequestScheduler] = None
gpu_scheduler: Optional[GPUScheduler] = None

gpus: Dict[int, GPUInfo] = {}
gpu_sets: List[Tuple[int, ...]] = []
model_cards: Dict[str, ModelCard] = {}
instances: Dict[str, Instance] = {}
active_ids: Set[str] = set()
slept_ids: Set[str] = set()
jobs: Dict[str, JobGraph] = {}

start_time: float = 0.0


# -----------------------------
# Lifecycle
# -----------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management."""
    global config, controller, request_scheduler, gpu_scheduler
    global gpus, gpu_sets, model_cards, instances, active_ids, slept_ids
    global start_time
    
    start_time = time.monotonic()
    
    # Load configuration
    config = SchedulerConfig.from_env()
    logger.info(f"Configuration loaded: δt1={config.delta_t1}s, δt2={config.delta_t2}s, alpha={config.alpha}")
    
    # Discover GPUs
    gpu_list = discover_gpus(config.alpha)
    gpus = {g.gpu_id: g for g in gpu_list}
    gpu_sets = build_gpu_sets([g.gpu_id for g in gpu_list])
    logger.info(f"GPU sets: {gpu_sets}")
    
    # Load model cards
    model_cards.update(read_model_cards(config.model_card_path))
    logger.info(f"Loaded {len(model_cards)} model cards from {config.model_card_path}")
    
    # Initialize controller
    controller = ManagedVLLMController(
        host=config.vllm_host,
        port_base=config.vllm_port_base,
        request_timeout_s=config.vllm_request_timeout_s,
        startup_timeout_s=config.vllm_startup_timeout_s,
        vllm_extra_args=config.vllm_extra_args,
    )
    
    # Initialize schedulers
    request_scheduler = RequestScheduler(
        controller=controller,
        instances=instances,
        active_ids=active_ids,
        model_cards=model_cards,
        beta=config.beta,
        max_decisions_per_step=config.max_decisions_per_step,
    )
    
    gpu_scheduler = GPUScheduler(
        gpus=gpus,
        gpu_sets=gpu_sets,
        model_cards=model_cards,
        instances=instances,
        active_ids=active_ids,
        slept_ids=slept_ids,
        controller=controller,
        timing_ema_alpha=config.timing_ema_alpha,
    )
    
    # Start scheduling loops
    request_task = asyncio.create_task(_request_loop())
    gpu_task = asyncio.create_task(_gpu_loop())
    
    logger.info("MCFP MoA Proxy Server started")
    
    yield
    
    # Cleanup
    request_task.cancel()
    gpu_task.cancel()
    
    try:
        await request_task
    except asyncio.CancelledError:
        pass
    
    try:
        await gpu_task
    except asyncio.CancelledError:
        pass
    
    # Persist model cards
    write_model_cards(config.model_card_path, model_cards)
    logger.info("MCFP MoA Proxy Server stopped")


async def _request_loop():
    """Request scheduling loop (runs every δt1)."""
    assert config is not None
    assert request_scheduler is not None
    
    while True:
        try:
            unmet = await request_scheduler.step()
            if unmet:
                logger.debug(f"Unmet demand for models: {unmet}")
        except Exception as e:
            logger.error(f"Request scheduler error: {e}")
        
        await asyncio.sleep(config.delta_t1)


async def _gpu_loop():
    """GPU reconfiguration loop (runs every δt2)."""
    assert config is not None
    assert request_scheduler is not None
    assert gpu_scheduler is not None
    
    while True:
        try:
            potential = await request_scheduler.get_potential_requests()
            waiting = await request_scheduler.get_waiting_requests()
            
            plan = await gpu_scheduler.reconfigure(potential, waiting)
            
            if not plan.is_empty:
                logger.info(
                    f"Reconfiguration: loads={len(plan.to_load)}, "
                    f"wakes={len(plan.to_wake)}, sleeps={len(plan.to_sleep)}, "
                    f"offloads={len(plan.to_offload)}"
                )
        except Exception as e:
            logger.error(f"GPU scheduler error: {e}")
        
        await asyncio.sleep(config.delta_t2)


# -----------------------------
# FastAPI Application
# -----------------------------

app = FastAPI(
    title="MCFP MoA Proxy Server",
    version="1.0.0",
    description="Multi-agent DAG job scheduler with min-cost flow GPU optimization",
    lifespan=lifespan,
)


# -----------------------------
# Endpoints
# -----------------------------

@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        uptime_s=time.monotonic() - start_time,
        gpus=len(gpus),
        active_instances=len(active_ids),
        pending_requests=(
            request_scheduler.get_stats()["waiting_queue_size"]
            if request_scheduler else 0
        ),
    )


@app.post("/submit_graph", response_model=SubmitResponse)
async def submit_graph(body: SubmitBody):
    """Submit a DAG job for execution."""
    if request_scheduler is None:
        raise HTTPException(status_code=503, detail="Scheduler not ready")
    
    job_id = body.job_id or str(uuid.uuid4())
    if job_id in jobs:
        raise HTTPException(status_code=409, detail="Job ID already exists")
    
    # Create job graph
    try:
        job = JobGraph(
            job_id=job_id,
            graph=body.graph,
            node_models=body.node_models or {},
            inputs=body.inputs or {},
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    
    # Ensure all models have cards
    unique_models = set(job.model_of(n) for n in job.nodes)
    for model_id in unique_models:
        if model_id not in model_cards:
            await _ensure_model(model_id)
    
    # Register job
    jobs[job_id] = job
    
    # Submit source nodes
    sources = job.get_sources()
    for node in sources:
        req = job.make_request(node)
        req.payload = job.build_payload(node)
        req.on_dispatched = lambda r, j=job: _on_dispatched(j, r)
        req.on_completed = lambda r, out, j=job: _on_completed(j, r, out)
        job.discovered.add(node)
        await request_scheduler.add_request(req)
    
    logger.info(f"Job {job_id} submitted with {len(job.nodes)} nodes")
    
    return SubmitResponse(
        job_id=job_id,
        sources=sources,
        num_nodes=len(job.nodes),
        gpu_sets=[list(s) for s in gpu_sets],
    )


@app.get("/status/{job_id}", response_model=StatusResponse)
async def status(job_id: str):
    """Get job status."""
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    return StatusResponse(
        job_id=job.job_id,
        done=job.done,
        completed=sorted(job.completed),
        pending=sorted(job.nodes - job.completed),
        outputs=job.outputs if job.done else {},
    )


@app.post("/register_instance")
async def register_instance(body: RegisterInstanceBody) -> Dict[str, Any]:
    """Register an external vLLM instance."""
    if request_scheduler is None:
        raise HTTPException(status_code=503, detail="Scheduler not ready")
    
    gpu_set = tuple(body.gpu_set)
    pid_by_gpu = {int(k): int(v) for k, v in body.pid_by_gpu.items()}
    
    # Ensure model card exists
    if body.model not in model_cards:
        await _ensure_model(body.model)
    
    # Create instance
    base_url = f"http://{config.vllm_host}:{body.port}"
    inst = Instance(
        instance_id=f"{','.join(map(str, gpu_set))}|{body.model}|{body.port}",
        model_id=body.model,
        gpus=gpu_set,
        base_url=base_url,
        metrics_url=f"{base_url}/metrics",
        state=InstState.ACTIVE,
        pid_by_gpu=pid_by_gpu,
    )
    
    # Register
    instances[inst.instance_id] = inst
    active_ids.add(inst.instance_id)
    
    # Update GPU state
    now = time.monotonic()
    for g in gpu_set:
        if g in gpus:
            gpus[g].set_resident(
                body.model, Residence.ACTIVE, now,
                pid=pid_by_gpu.get(g),
                tp=len(gpu_set),
                url=base_url
            )
    
    logger.info(f"Registered external instance {inst.instance_id}")
    
    return {"ok": True, "instance_id": inst.instance_id, "base_url": base_url}


@app.get("/instances")
async def list_instances() -> List[Dict[str, Any]]:
    """List all known vLLM instances, sorted by GPUs."""
    if controller is None:
        raise HTTPException(status_code=503, detail="Controller not ready")
    
    # Sort instances by GPU IDs
    inst_list = sorted(instances.values(), key=lambda i: i.gpus)
    
    # Fetch metrics
    metrics = await asyncio.gather(
        *[controller.metrics(i) for i in inst_list],
        return_exceptions=True
    )
    
    result = []
    for inst, m in zip(inst_list, metrics):
        entry = {
            "instance_id": inst.instance_id,
            "model": inst.model_id,
            "gpus": list(inst.gpus),
            "base_url": inst.base_url,
            "state": inst.state.value,
            "accept_new": inst.accept_new,
            "pid_by_gpu": {str(k): v for k, v in inst.pid_by_gpu.items()},
        }
        
        if not isinstance(m, Exception):
            entry["num_requests"] = m.num_requests
            entry["avg_latency"] = m.avg_latency
            entry["drain_latency"] = m.drain_latency
        
        result.append(entry)
    
    return result


@app.get("/model_stats")
async def model_stats() -> Dict[str, Any]:
    """Get EMA timing/memory stats per model."""
    return {
        "models": {
            mid: card.to_dict()
            for mid, card in model_cards.items()
        }
    }


@app.get("/scheduler_stats")
async def scheduler_stats() -> Dict[str, Any]:
    """Get scheduler statistics."""
    result = {}
    
    if request_scheduler:
        result["request_scheduler"] = request_scheduler.get_stats()
    
    if gpu_scheduler:
        result["gpu_scheduler"] = gpu_scheduler.get_stats()
    
    return result


@app.get("/gpus")
async def list_gpus() -> Dict[str, Any]:
    """List GPU status."""
    return {
        "gpus": [
            {
                "gpu_id": g.gpu_id,
                "vram_total_MB": g.vram_total_MB,
                "alpha": g.alpha,
                "weight_cap_MB": g.weight_cap_MB,
                "state": g.state.value,
                "resident": {m: r.value for m, r in g.resident.items()},
                "active_models": g.get_active_models(),
                "slept_models": g.get_slept_models(),
            }
            for g in gpus.values()
        ],
        "gpu_sets": [list(s) for s in gpu_sets],
    }


# -----------------------------
# Job Callbacks
# -----------------------------

def _on_dispatched(job: JobGraph, req: Request) -> List[Request]:
    """
    Called when a request is dispatched.
    
    Returns newly discovered successor requests.
    """
    # Build payload if not already set
    if req.payload is None:
        req.payload = job.build_payload(req.node_id)
    
    # Discover successors
    newly: List[Request] = []
    for succ in job.graph.get(req.node_id, []):
        if succ in job.discovered:
            continue
        
        job.discovered.add(succ)
        sreq = job.make_request(succ)
        sreq.on_dispatched = lambda r, j=job: _on_dispatched(j, r)
        sreq.on_completed = lambda r, out, j=job: _on_completed(j, r, out)
        newly.append(sreq)
    
    return newly


def _on_completed(job: JobGraph, req: Request, output: Any) -> None:
    """
    Called when a request completes.
    
    Updates job state and decrements successor indegrees.
    """
    node = req.node_id
    job.completed.add(node)
    job.outputs[node] = output
    
    # Decrement successor indegrees
    for succ in job.graph.get(node, []):
        job.indegree[succ] = max(0, job.indegree[succ] - 1)
        
        # Update scheduler's view
        if request_scheduler:
            asyncio.create_task(
                request_scheduler.update_indegree(job.job_id, succ, job.indegree[succ])
            )
    
    if job.done:
        logger.info(f"Job {job.job_id} completed")


# -----------------------------
# Helpers
# -----------------------------

async def _ensure_model(model_id: str) -> None:
    """Ensure model card exists, creating if needed."""
    if model_id in model_cards:
        return
    
    # Try to estimate size
    try:
        size_MB = hf_guess_model_size_MB(model_id)
        tp_min = choose_tp_min(size_MB, list(gpus.values()))
    except Exception:
        size_MB = 4096.0
        tp_min = 1
    
    card = ensure_model_card(
        model_cards, model_id, tp_min,
        persist_path=config.model_card_path if config else None
    )
    
    logger.info(f"Created model card for {model_id}: tp_min={tp_min}, size_MB={size_MB:.0f}")


# -----------------------------
# Main
# -----------------------------

if __name__ == "__main__":
    import uvicorn
    
    host = os.getenv("SERVER_HOST", "0.0.0.0")
    port = int(os.getenv("SERVER_PORT", "8000"))
    
    uvicorn.run(app, host=host, port=port)
