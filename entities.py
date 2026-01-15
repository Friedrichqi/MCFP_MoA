"""
entities.py - Core data models for the MCFP MoA scheduling system.

Defines all entities as specified in logic_devemopment_revised.md:
- GPU state and properties
- Model cards with timing/memory profiles
- Instance lifecycle states
- Request states and DAG handling
- Metrics snapshots for drain latency
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple


# -----------------------------
# Enums
# -----------------------------

class GPUState(str, Enum):
    """GPU operational state - only STABLE GPUs participate in scheduling."""
    STABLE = "stable"
    UNSTABLE = "unstable"


class Residence(str, Enum):
    """Model residence state on a GPU."""
    ACTIVE = "active"
    SLEPT = "slept"


class InstState(str, Enum):
    """Instance lifecycle states as per the document."""
    LOADING = "loading"
    ACTIVE = "active"
    SLEPT = "slept"
    VANISHING = "vanishing"


class ReqState(str, Enum):
    """Request states in the scheduling pipeline."""
    POTENTIAL = "potential"  # Awaiting prerequisites (DAG dependencies)
    WAITING = "waiting"      # Ready, waiting for GPU assignment
    RUNNING = "running"      # Currently executing on an instance
    DONE = "done"            # Completed


# -----------------------------
# Model Card
# -----------------------------

@dataclass
class ModelCard:
    """
    Model card with timing and memory profiles.
    
    EMAs are updated online as operations complete:
    - t_wake_s: wake a slept instance
    - t_sleep_s: sleep an instance
    - t_load_s: start vLLM from scratch until ready
    - t_offload_s: kill instance (evict from GPU)
    - avg_latency_s: average e2e request latency
    
    Memory footprints:
    - slept_mem_tp1_MB: memory per GPU when tp=1
    - slept_mem_tpgt1_MB: memory per GPU when tp>1
    """
    
    model_id: str
    tp_min: int = 1
    
    # Timing estimates (EMA-updated)
    t_wake_s: float = 2.0
    t_sleep_s: float = 2.0
    t_load_s: float = 90.0
    t_offload_s: float = 3.0
    avg_latency_s: float = 60.0  # Default 60s as initial estimate
    
    # Memory footprints
    slept_mem_tp1_MB: float = 2048.0
    slept_mem_tpgt1_MB: float = 4096.0
    
    def slept_mem_MB(self, tp: int) -> float:
        """Get slept memory footprint based on tensor-parallel degree."""
        return self.slept_mem_tp1_MB if tp <= 1 else self.slept_mem_tpgt1_MB
    
    @staticmethod
    def _ema(old: float, new: float, alpha: float = 0.2) -> float:
        """Exponential moving average update."""
        return max(1e-3, (1.0 - alpha) * old + alpha * new)
    
    def update_wake(self, observed_s: float, alpha: float = 0.2) -> None:
        self.t_wake_s = self._ema(self.t_wake_s, observed_s, alpha)
    
    def update_sleep(self, observed_s: float, alpha: float = 0.2) -> None:
        self.t_sleep_s = self._ema(self.t_sleep_s, observed_s, alpha)
    
    def update_load(self, observed_s: float, alpha: float = 0.2) -> None:
        self.t_load_s = self._ema(self.t_load_s, observed_s, alpha)
    
    def update_offload(self, observed_s: float, alpha: float = 0.2) -> None:
        self.t_offload_s = self._ema(self.t_offload_s, observed_s, alpha)
    
    def update_avg_latency(self, observed_s: float, alpha: float = 0.2) -> None:
        """Update average latency EMA with observed value."""
        if observed_s > 0:
            self.avg_latency_s = self._ema(self.avg_latency_s, observed_s, alpha)
    
    def update_slept_mem(self, tp: int, observed_MB: float, alpha: float = 0.2) -> None:
        observed_MB = max(1.0, observed_MB)
        if tp <= 1:
            self.slept_mem_tp1_MB = self._ema(self.slept_mem_tp1_MB, observed_MB, alpha)
        else:
            self.slept_mem_tpgt1_MB = self._ema(self.slept_mem_tpgt1_MB, observed_MB, alpha)
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for JSON persistence."""
        return {
            "tp_min": self.tp_min,
            "t_wake_s": self.t_wake_s,
            "t_sleep_s": self.t_sleep_s,
            "t_load_s": self.t_load_s,
            "t_offload_s": self.t_offload_s,
            "avg_latency_s": self.avg_latency_s,
            "slept_mem_tp1_MB": self.slept_mem_tp1_MB,
            "slept_mem_tpgt1_MB": self.slept_mem_tpgt1_MB,
        }
    
    @classmethod
    def from_dict(cls, model_id: str, data: Dict[str, Any]) -> "ModelCard":
        """Deserialize from dictionary."""
        return cls(
            model_id=model_id,
            tp_min=int(data.get("tp_min", 1)),
            t_wake_s=float(data.get("t_wake_s", 2.0)),
            t_sleep_s=float(data.get("t_sleep_s", 2.0)),
            t_load_s=float(data.get("t_load_s", 90.0)),
            t_offload_s=float(data.get("t_offload_s", 3.0)),
            avg_latency_s=float(data.get("avg_latency_s", 60.0)),
            slept_mem_tp1_MB=float(data.get("slept_mem_tp1_MB", data.get("slept_mem_MB_tp1", 2048.0))),
            slept_mem_tpgt1_MB=float(data.get("slept_mem_tpgt1_MB", data.get("slept_mem_MB_tpgt1", 4096.0))),
        )


# -----------------------------
# GPU Info
# -----------------------------

@dataclass
class GPUInfo:
    """
    Per-GPU state and properties.
    
    Tracks:
    - Hardware properties (VRAM, alpha limit)
    - Operational state (STABLE/UNSTABLE)
    - Resident models and their states
    - PIDs and URLs for each model's instance
    """
    
    gpu_id: int
    vram_total_MB: int
    alpha: float  # max fraction of VRAM for slept weights
    
    state: GPUState = GPUState.STABLE
    resident: Dict[str, Residence] = field(default_factory=dict)  # model_id -> state
    pid_by_model: Dict[str, int] = field(default_factory=dict)
    url_by_model: Dict[str, str] = field(default_factory=dict)
    tp_by_model: Dict[str, int] = field(default_factory=dict)
    last_used: Dict[str, float] = field(default_factory=dict)
    
    @property
    def weight_cap_MB(self) -> float:
        """Maximum VRAM allowed for slept weights."""
        return self.alpha * self.vram_total_MB
    
    @property
    def is_stable(self) -> bool:
        """Check if GPU participates in scheduling."""
        return self.state == GPUState.STABLE
    
    def set_resident(
        self,
        model_id: str,
        residence: Residence,
        now: float,
        *,
        pid: Optional[int] = None,
        tp: Optional[int] = None,
        url: Optional[str] = None,
    ) -> None:
        """Update model residence on this GPU."""
        self.resident[model_id] = residence
        self.last_used[model_id] = now
        if pid is not None:
            self.pid_by_model[model_id] = pid
        if tp is not None:
            self.tp_by_model[model_id] = tp
        if url is not None:
            self.url_by_model[model_id] = url
    
    def evict_model(self, model_id: str) -> None:
        """Remove model from GPU residence."""
        self.resident.pop(model_id, None)
        self.pid_by_model.pop(model_id, None)
        self.url_by_model.pop(model_id, None)
        self.tp_by_model.pop(model_id, None)
        self.last_used.pop(model_id, None)
    
    def is_resident(self, model_id: str) -> bool:
        """Check if model is resident on this GPU."""
        return model_id in self.resident
    
    def get_active_models(self) -> List[str]:
        """Get list of active (not slept) models on this GPU."""
        return [m for m, r in self.resident.items() if r == Residence.ACTIVE]
    
    def get_slept_models(self) -> List[str]:
        """Get list of slept models on this GPU."""
        return [m for m, r in self.resident.items() if r == Residence.SLEPT]


# -----------------------------
# Instance
# -----------------------------

@dataclass
class Instance:
    """
    A vLLM server instance serving a specific model.
    
    Each instance:
    - Is bound to a GPU set (for tensor parallelism)
    - Serves a single model
    - Has lifecycle state (LOADING, ACTIVE, SLEPT, VANISHING)
    """
    
    instance_id: str
    model_id: str
    gpus: Tuple[int, ...]
    base_url: str
    metrics_url: str
    
    state: InstState = InstState.ACTIVE
    accept_new: bool = True
    pid_by_gpu: Dict[int, int] = field(default_factory=dict)
    vram_by_gpu: Dict[int, int] = field(default_factory=dict)
    created_time: float = field(default_factory=time.monotonic)
    last_used: float = field(default_factory=time.monotonic)
    
    @property
    def tp(self) -> int:
        """Tensor-parallel degree (number of GPUs)."""
        return len(self.gpus)
    
    @property
    def is_active(self) -> bool:
        return self.state == InstState.ACTIVE
    
    @property
    def is_slept(self) -> bool:
        return self.state == InstState.SLEPT
    
    @property
    def is_loading(self) -> bool:
        return self.state == InstState.LOADING


# -----------------------------
# Request
# -----------------------------

@dataclass
class Request:
    """
    One schedulable unit = one DAG node (stage).
    
    Multiple nodes in a job DAG may share the same model
    but are treated as separate requests.
    
    Edge u -> v means outputs of u concatenate into inputs of v.
    """
    
    job_id: str
    node_id: str
    model: str
    t_arr: float  # Arrival time (monotonic)
    indegree: int  # Number of unmet dependencies
    
    succ: List[str] = field(default_factory=list)  # Successor node_ids
    payload: Any = None
    state: ReqState = ReqState.POTENTIAL
    
    # Callbacks
    on_dispatched: Optional[Callable[["Request"], List["Request"]]] = None
    on_completed: Optional[Callable[["Request", Any], None]] = None
    
    @property
    def key(self) -> str:
        """Unique key for this request."""
        return f"{self.job_id}:{self.node_id}"
    
    @property
    def is_ready(self) -> bool:
        """Check if ready to execute (all dependencies met)."""
        return self.indegree == 0
    
    @property
    def wait_time(self) -> float:
        """Time spent waiting (since arrival)."""
        return time.monotonic() - self.t_arr


# -----------------------------
# Drain Latency (Metrics Snapshot)
# -----------------------------

@dataclass
class DrainLatency:
    """
    Metrics snapshot from vLLM for computing drain latency.
    
    Uses p95 latency as drain estimate since requests are processed
    concurrently in batches, not sequentially.
    
    When no completed requests exist (latency_count=0), falls back to
    fallback_latency which should be set from ModelCard.avg_latency_s.
    
    This is used both for:
    - Immediate routing decisions
    - Edge cost in GPU selection (min-cost flow)
    """
    
    num_requests: float = 0.0
    latency_sum: float = 0.0
    latency_count: float = 0.0
    # Histogram buckets: list of (le_bound, cumulative_count)
    # Sorted by le_bound ascending, with +inf as the last entry
    latency_buckets: List[Tuple[float, float]] = field(default_factory=list)
    # Fallback latency from ModelCard when no data available
    fallback_latency: float = 60.0
    
    @property
    def avg_latency(self) -> float:
        """Average end-to-end latency, with fallback if no data."""
        if self.latency_count > 0:
            return self.latency_sum / self.latency_count
        return self.fallback_latency
    
    def percentile_latency(self, p: float = 0.95) -> float:
        """
        Estimate percentile latency from histogram buckets.
        
        Uses linear interpolation within the bucket containing the percentile.
        Falls back to avg_latency if buckets are not available.
        
        Args:
            p: Percentile as fraction (0.95 = 95th percentile)
        """
        if not self.latency_buckets or self.latency_count <= 0:
            return self.avg_latency
        
        target_count = p * self.latency_count
        
        # Find the bucket containing the target percentile
        prev_bound = 0.0
        prev_count = 0.0
        
        for le_bound, cum_count in self.latency_buckets:
            if cum_count >= target_count:
                # Linear interpolation within this bucket
                if cum_count == prev_count:
                    # No samples in this bucket, use upper bound
                    return le_bound if le_bound != float('inf') else prev_bound
                
                # Fraction within this bucket
                frac = (target_count - prev_count) / (cum_count - prev_count)
                
                if le_bound == float('inf'):
                    # Last bucket is +inf, use previous bound + some margin
                    return prev_bound * 1.5 if prev_bound > 0 else self.avg_latency
                
                return prev_bound + frac * (le_bound - prev_bound)
            
            prev_bound = le_bound
            prev_count = cum_count
        
        # All samples are beyond the last finite bucket
        return self.avg_latency
    
    @property
    def p95_latency(self) -> float:
        """95th percentile latency."""
        return self.percentile_latency(0.95)
    
    @property
    def drain_latency(self) -> float:
        """
        Estimated time to drain current queue.
        
        Uses p95 latency since requests are processed concurrently.
        This represents how long until the slowest request finishes.
        """
        if self.num_requests <= 0:
            return 0.0
        return self.p95_latency
    
    @property
    def backlog_cost(self) -> float:
        """Alias for drain_latency (used in cost calculations)."""
        return self.drain_latency


# -----------------------------
# Reconfiguration Plan
# -----------------------------

@dataclass(frozen=True)
class ReconfigAction:
    """A single GPU reconfiguration action."""
    action: str  # "load", "wake", "sleep", "offload"
    instance_id: Optional[str] = None
    model_id: Optional[str] = None
    gpu_set: Optional[Tuple[int, ...]] = None


@dataclass
class ReconfigPlan:
    """
    Output of GPU scheduler: what to load/wake/sleep/offload.
    
    Generated by solving the min-cost flow problem.
    """
    
    to_load: List[Tuple[Tuple[int, ...], str]] = field(default_factory=list)   # (gpu_set, model_id)
    to_wake: List[str] = field(default_factory=list)                            # instance_ids
    to_sleep: List[str] = field(default_factory=list)                           # instance_ids  
    to_offload: List[str] = field(default_factory=list)                         # instance_ids
    
    estimated_cost_s: float = 0.0
    
    @property
    def is_empty(self) -> bool:
        return not (self.to_load or self.to_wake or self.to_sleep or self.to_offload)
    
    def actions(self) -> List[ReconfigAction]:
        """Get ordered list of actions to execute."""
        actions = []
        for iid in self.to_sleep:
            actions.append(ReconfigAction("sleep", instance_id=iid))
        for iid in self.to_offload:
            actions.append(ReconfigAction("offload", instance_id=iid))
        for iid in self.to_wake:
            actions.append(ReconfigAction("wake", instance_id=iid))
        for gpu_set, model_id in self.to_load:
            actions.append(ReconfigAction("load", model_id=model_id, gpu_set=gpu_set))
        return actions
