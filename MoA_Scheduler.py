from __future__ import annotations

import asyncio, heapq, math, time, subprocess, re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple


# -----------------------------
# Basic enums / data structures
# -----------------------------

class Residence(str, Enum):
    ACTIVE = "active"
    SLEPT = "slept"


class InstState(str, Enum):
    IDLE = "idle"
    ACTIVE = "active"
    DRAINING = "draining"
    SWITCHING = "switching"


class ReqState(str, Enum):
    POTENTIAL = "potential"
    WAITING = "waiting"
    RUNNING = "running"
    DONE = "done"


@dataclass
class ModelInfo:
    """Mutable model-card entry + online stats.

    Memory accounting:
      - We do NOT keep bf16 weight size / shards.
      - Instead we maintain an estimated per-GPU "slept weight" memory footprint,
        which is updated when we actually sleep a model (via nvidia-smi pid used_memory).

    Defaults requested:
      - slept_mem_MB_tp1 = 2048
      - slept_mem_MB_tpgt1 = 4096
      - t_wake_s = 2, t_load_s = 90, t_offload_s = 3
    """
    name: str
    tp_min: int

    # Online timings (EMA)
    t_wake_s: float = 2.0
    t_load_s: float = 90.0
    t_offload_s: float = 3.0

    # Online "slept weights" footprint (MB per GPU)
    slept_mem_MB_tp1: float = 2048.0
    slept_mem_MB_tpgt1: float = 4096.0

    # Simulator-only (optional)
    avg_service_s: float = 0.2

    def slept_mem_MB(self, tp: int) -> float:
        return float(self.slept_mem_MB_tp1 if int(tp) <= 1 else self.slept_mem_MB_tpgt1)

    def update_wake(self, observed_s: float, ema: float = 0.2) -> None:
        self.t_wake_s = max(1e-3, (1.0 - ema) * float(self.t_wake_s) + ema * float(observed_s))

    def update_load(self, observed_s: float, ema: float = 0.2) -> None:
        self.t_load_s = max(1e-3, (1.0 - ema) * float(self.t_load_s) + ema * float(observed_s))

    def update_offload(self, observed_s: float, ema: float = 0.2) -> None:
        self.t_offload_s = max(1e-3, (1.0 - ema) * float(self.t_offload_s) + ema * float(observed_s))

    def update_slept_mem(self, tp: int, observed_MB_per_gpu: float, ema: float = 0.2) -> None:
        v = max(1.0, float(observed_MB_per_gpu))
        if int(tp) <= 1:
            self.slept_mem_MB_tp1 = (1.0 - ema) * float(self.slept_mem_MB_tp1) + ema * v
        else:
            self.slept_mem_MB_tpgt1 = (1.0 - ema) * float(self.slept_mem_MB_tpgt1) + ema * v


@dataclass
class GPUInfo:
    gpu_id: int
    vram_total_MB: int
    alpha: float

    # model -> ACTIVE/SLEPT, absent means evicted
    resident: Dict[str, Residence] = field(default_factory=dict)

    # model -> pid on this GPU (vLLM can be multi-process; we track per-GPU PID)
    pid_by_model: Dict[str, int] = field(default_factory=dict)

    # model -> tp used when this model was loaded on this GPU (usually the instance TP)
    tp_by_model: Dict[str, int] = field(default_factory=dict)

    last_used: Dict[str, float] = field(default_factory=dict)

    @property
    def weight_cap_MB(self) -> float:
        return self.alpha * float(self.vram_total_MB)

    def set_resident(self, model: str, st: Residence, now: float, *, pid: Optional[int] = None, tp: Optional[int] = None) -> None:
        self.resident[model] = st
        self.last_used[model] = now
        if pid is not None:
            self.pid_by_model[model] = int(pid)
        if tp is not None:
            self.tp_by_model[model] = int(tp)

    def evict(self, model: str) -> None:
        self.resident.pop(model, None)
        self.pid_by_model.pop(model, None)
        self.tp_by_model.pop(model, None)
        self.last_used.pop(model, None)

    def is_resident(self, model: str) -> bool:
        return model in self.resident


@dataclass
class Instance:
    """
    One logical vLLM instance bound to a GPU set.
    state can be IDLE/ACTIVE/DRAINING/SWITCHING.
    """
    gpus: Tuple[int, ...]
    state: InstState = InstState.IDLE
    model: Optional[str] = None
    accept_new: bool = True

    # per-GPU PID mapping (required for memory accounting)
    pid_by_gpu: Dict[int, int] = field(default_factory=dict)

    # where to fetch metrics; if empty, controller may use a default/global URL
    metrics_url: str = ""


@dataclass
class Request:
    '''
    One schedulable unit = one DAG node (stage). Multiple nodes may share the same model.

    - node_id is the stage identifier (unique within job).
    - model is the underlying model name (may repeat across stages).
    '''
    job_id: str
    node_id: str
    model: str
    t_arr: float
    indegree: int
    succ: List[str] = field(default_factory=list)
    payload: Any = None
    state: ReqState = ReqState.POTENTIAL

    # on_dispatched returns "newly discovered successors" (Requests) to insert into POTENTIAL immediately.
    on_dispatched = None
    on_completed = None

    def key(self) -> str:
        return f"{self.job_id}:{self.node_id}"


@dataclass(frozen=True)
class SwitchPlan:
    target_set: Tuple[int, ...]
    target_model: str
    overlapped_sets: Tuple[Tuple[int, ...], ...]
    displaced_models: Tuple[str, ...]
    evict_ops: Tuple[Tuple[int, str], ...]      # (gpu_id, model)
    evicted_models: Tuple[str, ...]
    activate_kind: str                          # "wake" or "load"
    est_drain_s: float
    est_activate_s: float
    est_evict_s: float
    total_cost_s: float


@dataclass
class MetricsSnapshot:
    num_requests: float = 0.0
    e2e_sum: float = 0.0
    e2e_count: float = 0.0

    @property
    def avg_latency(self) -> float:
        return float(self.e2e_sum) / float(self.e2e_count) if self.e2e_count > 0 else 0.0

    @property
    def backlog_cost(self) -> float:
        return float(self.num_requests) * float(self.avg_latency)


# -----------------------------
# Controller interface
# -----------------------------

class VLLMController:
    """Adapter for vLLM + nvidia-smi + metrics."""

    async def infer(self, model: str, payload: Any, gpu_set: Tuple[int, ...]) -> Any:
        raise NotImplementedError

    async def stop_accepting_new(self, gpu_set: Tuple[int, ...]) -> None:
        return None

    async def metrics(self, instance: Instance) -> MetricsSnapshot:
        return MetricsSnapshot()

    async def pid_used_MB(self) -> Dict[Tuple[int, int], int]:
        return {}

    async def sleep_observe(self, instance: Instance) -> Optional[float]:
        return None

    async def offload_observe(self, gpu_id: int, pid: int) -> Optional[float]:
        return None


class RealVLLMController(VLLMController):
    """Real integrations via subprocess (curl + nvidia-smi)."""

    def __init__(self, default_metrics_url: str = "") -> None:
        self.default_metrics_url = default_metrics_url

    async def metrics(self, instance: Instance) -> MetricsSnapshot:
        url = instance.metrics_url or self.default_metrics_url
        if not url:
            return MetricsSnapshot()
        return await asyncio.to_thread(self._metrics_sync, url, instance.model, instance.pid_by_gpu)

    @staticmethod
    def _parse_prom_line(line: str) -> Optional[Tuple[str, Dict[str, str], float]]:
        line = line.strip()
        if not line or line.startswith("#"):
            return None
        if "{" in line and "}" in line:
            m = re.match(r"^([^\{\s]+)\{([^\}]*)\}\s+([-+eE0-9\.]+)\s*$", line)
            if not m:
                return None
            name = m.group(1)
            labels_raw = m.group(2).strip()
            value = float(m.group(3))
            labels: Dict[str, str] = {}
            if labels_raw:
                for part in labels_raw.split(","):
                    if "=" in part:
                        k, v = part.split("=", 1)
                        labels[k.strip()] = v.strip().strip('"')
            return name, labels, value
        m = re.match(r"^([^\s]+)\s+([-+eE0-9\.]+)\s*$", line)
        if not m:
            return None
        return m.group(1), {}, float(m.group(2))

    @classmethod
    def _metrics_sync(cls, url: str, model: Optional[str], pid_by_gpu: Dict[int, int]) -> MetricsSnapshot:
        try:
            raw = subprocess.check_output(["curl", "-s", url], text=True)
        except Exception:
            return MetricsSnapshot()

        targets = {
            "vllm:num_requests",
            "vllm:e2e_request_latency_seconds_sum",
            "vllm:e2e_request_latency_seconds_count",
        }

        samples: Dict[str, List[Tuple[Dict[str, str], float]]] = {k: [] for k in targets}
        for ln in raw.splitlines():
            parsed = cls._parse_prom_line(ln)
            if not parsed:
                continue
            name, labels, val = parsed
            if name in samples:
                samples[name].append((labels, val))

        def pick(name: str) -> float:
            arr = samples.get(name, [])
            if not arr:
                return 0.0
            if model:
                for labels, v in arr:
                    if labels.get("model_name") == model:
                        return float(v)
            pids = {str(pid) for pid in pid_by_gpu.values() if pid}
            for labels, v in arr:
                if labels.get("pid") in pids:
                    return float(v)
            return float(arr[0][1])

        num = pick("vllm:num_requests")
        s = pick("vllm:e2e_request_latency_seconds_sum")
        c = pick("vllm:e2e_request_latency_seconds_count")
        return MetricsSnapshot(num_requests=num, e2e_sum=s, e2e_count=c)

    async def pid_used_MB(self) -> Dict[Tuple[int, int], int]:
        return await asyncio.to_thread(self._pid_used_MB_sync)

    @staticmethod
    def _pid_used_MB_sync() -> Dict[Tuple[int, int], int]:
        uuid_to_idx: Dict[str, int] = {}
        try:
            out = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"],
                text=True,
            ).strip()
            for ln in out.splitlines():
                idx, uuid = [x.strip() for x in ln.split(",")]
                uuid_to_idx[uuid] = int(idx)
        except Exception:
            uuid_to_idx = {}

        try:
            out = subprocess.check_output(
                ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_memory", "--format=csv,noheader,nounits"],
                text=True,
            ).strip()
            res: Dict[Tuple[int, int], int] = {}
            for ln in out.splitlines():
                parts = [x.strip() for x in ln.split(",")]
                if len(parts) != 3:
                    continue
                gpu_uuid, pid_s, used_s = parts
                if gpu_uuid not in uuid_to_idx:
                    continue
                try:
                    pid = int(pid_s)
                    used = int(float(used_s))
                except Exception:
                    continue
                res[(uuid_to_idx[gpu_uuid], pid)] = used
            return res
        except Exception:
            return {}


class SimVLLMController(VLLMController):
    """Runnable simulator. It doesn't call curl/nvidia-smi."""
    def __init__(self, models: Dict[str, ModelInfo]) -> None:
        self.models = models
        self._inflight: Dict[Tuple[int, ...], int] = {}

    async def infer(self, model: str, payload: Any, gpu_set: Tuple[int, ...]) -> Any:
        self._inflight[gpu_set] = self._inflight.get(gpu_set, 0) + 1
        await asyncio.sleep(self.models[model].avg_service_s)
        self._inflight[gpu_set] = max(0, self._inflight.get(gpu_set, 1) - 1)
        return {"model": model, "gpu_set": list(gpu_set), "ok": True, "payload": payload}

    async def metrics(self, instance: Instance) -> MetricsSnapshot:
        inflight = float(self._inflight.get(instance.gpus, 0))
        mi = self.models.get(instance.model or "")
        if not mi:
            return MetricsSnapshot(num_requests=inflight, e2e_sum=0.0, e2e_count=0.0)
        return MetricsSnapshot(num_requests=inflight, e2e_sum=float(mi.avg_service_s), e2e_count=1.0)


# -----------------------------
# Scheduler core
# -----------------------------

class moa_scheduler:
    '''
    Scheduler with POTENTIAL/WAITING priority queues (by t_arr).

    - Does NOT maintain backlog_q / avg_time_s; those come from vLLM /metrics snapshots.
    - Does NOT maintain model size/shards; C1 uses per-GPU slept-weight footprint estimated from PID used_memory.

    Decisions per waiting request:
      - RUN: dispatch to active instance
      - WAIT: keep waiting
      - SWITCH: drain cost + activate cost + eviction penalty

    Latest costs:

      C_wait(r) = beta * (now - t_arr)

      C_switch(s,j) = C_drain(s) + C_activate(s,j) + C_evict(s,j)

        C_drain(s) = max_{overlap inst} (Q_inst * avg_time_inst)

        C_activate(s,j) = Twake(j) if j resident on all GPUs in s else Tload(j)

        C_evict(s,j) has two scenarios:
          (1) If C1 met after placing target shard(s):
              C_evict = sum_{u displaced} Need(u)*Twake(u)
          (2) If C1 not met:
              Evict models with smallest Need(.)
              C_evict = base_sleep + offload cost + sum_{e evicted} Need(e)*(Tload(e)-Twake(e))

    Need(x) counts models over POTENTIAL ∪ WAITING.
    '''

    def __init__(
        self,
        *,
        gpus: List[GPUInfo],
        gpu_sets: Sequence[Tuple[int, ...]],
        models: Dict[str, ModelInfo],
        beta: float = 1.0,
        controller: Optional[VLLMController] = None,
        max_decisions_per_step: int = 256,
        timing_ema: float = 0.2,
    ) -> None:
        self.beta = float(beta)
        self.models = models
        self.timing_ema = float(timing_ema)

        self.gpus: Dict[int, GPUInfo] = {g.gpu_id: g for g in gpus}
        self.gpu_sets: List[Tuple[int, ...]] = [tuple(s) for s in gpu_sets]
        self.instances: Dict[Tuple[int, ...], Instance] = {s: Instance(s) for s in self.gpu_sets}

        self._reqs: Dict[str, Request] = {}
        self._potential: List[Tuple[float, int, str]] = []
        self._waiting: List[Tuple[float, int, str]] = []
        self._seq = 0

        self.controller = controller or SimVLLMController(models)
        self._lock = asyncio.Lock()
        self._bg: Set[asyncio.Task] = set()
        self.max_decisions_per_step = int(max_decisions_per_step)

        self._pid_mem: Dict[Tuple[int, int], int] = {}

    # -----------------
    # Instance registry
    # -----------------

    async def register_active_instance(
        self,
        gpu_set: Tuple[int, ...],
        model: str,
        pid_by_gpu: Dict[int, int],
        metrics_url: str = "",
    ) -> None:
        now = time.monotonic()
        async with self._lock:
            if gpu_set not in self.instances:
                raise ValueError(f"unknown gpu_set {gpu_set}")
            inst = self.instances[gpu_set]
            inst.state = InstState.ACTIVE
            inst.model = model
            inst.accept_new = True
            inst.pid_by_gpu = dict(pid_by_gpu)
            inst.metrics_url = metrics_url

            tp = len(gpu_set)
            for g in gpu_set:
                pid = pid_by_gpu.get(g)
                self.gpus[g].set_resident(model, Residence.ACTIVE, now, pid=pid, tp=tp)

    async def unregister_instance(self, gpu_set: Tuple[int, ...]) -> None:
        async with self._lock:
            inst = self.instances.get(gpu_set)
            if not inst:
                return
            inst.state = InstState.IDLE
            inst.model = None
            inst.accept_new = True
            inst.pid_by_gpu = {}
            inst.metrics_url = ""

    # -----------------
    # Queue operations
    # -----------------

    async def add_to_potential(self, req: Request) -> bool:
        """Returns True if inserted new, False if merged."""
        async with self._lock:
            return self._add_to_potential_locked(req)

    def _add_to_potential_locked(self, req: Request) -> bool:
        k = req.key()
        if k in self._reqs:
            # merge idempotently
            ex = self._reqs[k]
            ex.indegree = req.indegree
            ex.t_arr = min(ex.t_arr, req.t_arr)
            ex.succ = list({*ex.succ, *req.succ})
            return False
        req.state = ReqState.POTENTIAL
        self._reqs[k] = req
        self._push(self._potential, req.t_arr, k)
        return True

    async def update_indegree(self, job_id: str, node_id: str, new_indegree: int) -> None:
        k = f"{job_id}:{node_id}"
        async with self._lock:
            if k in self._reqs:
                self._reqs[k].indegree = int(new_indegree)

    async def move_ready_potential_to_waiting(self) -> int:
        """
        Every interval seconds:
          move all POTENTIAL requests with indegree==0 into WAITING.
        """
        async with self._lock:
            moved = 0
            new_p: List[Tuple[float, int, str]] = []
            while self._potential:
                t, seq, k = heapq.heappop(self._potential)
                r = self._reqs.get(k)
                if not r or r.state != ReqState.POTENTIAL:
                    continue
                if r.indegree == 0:
                    r.state = ReqState.WAITING
                    r.t_arr = time.monotonic()
                    self._push(self._waiting, r.t_arr, k)
                    moved += 1
                else:
                    new_p.append((t, seq, k))
            self._potential = new_p
            heapq.heapify(self._potential)
            return moved

    # -----------------
    # Main scheduling
    # -----------------

    async def step(self) -> None:
        now = time.monotonic()

        async with self._lock:
            # drain a batch of waiting reqs in arrival order
            batch: List[str] = []
            while self._waiting and len(batch) < self.max_decisions_per_step:
                _, _, k = heapq.heappop(self._waiting)
                r = self._reqs.get(k)
                if r and r.state == ReqState.WAITING:
                    batch.append(k)

            # Need counts over W ∪ P at the start of this step
            need = self._need_counts_locked()
            inst_list = [self.instances[s] for s in self.instances if self.instances[s].state in (InstState.ACTIVE, InstState.DRAINING, InstState.SWITCHING)]

        # snapshot metrics + pid memory outside lock
        metrics_map: Dict[Tuple[int, ...], MetricsSnapshot] = {}
        if inst_list:
            snaps = await asyncio.gather(*[self.controller.metrics(inst) for inst in inst_list], return_exceptions=True)
            for inst, snap in zip(inst_list, snaps):
                metrics_map[inst.gpus] = snap if not isinstance(snap, Exception) else MetricsSnapshot()

        pid_mem = await self.controller.pid_used_MB()

        async with self._lock:
            if pid_mem:
                self._pid_mem = pid_mem
            reserved: Set[int] = set()

        for k in batch:
            async with self._lock:
                r = self._reqs.get(k)
                if not r or r.state != ReqState.WAITING:
                    continue
                if r.model not in self.models:
                    self._requeue_locked(r)
                    continue

                run_cost, run_set = self._cost_run_locked(r, reserved, metrics_map)
                wait_cost = self.beta * max(0.0, now - r.t_arr)
                sw_cost, plan = self._best_switch_locked(r, need, reserved, metrics_map)

                action = min([("RUN", run_cost), ("WAIT", wait_cost), ("SWITCH", sw_cost)], key=lambda x: x[1])[0]

                if action == "RUN" and run_set and math.isfinite(run_cost):
                    # r leaves W -> update Need
                    need[r.model] = max(0, need.get(r.model, 0) - 1)
                    self._dispatch_locked(r, run_set, now)

                    # Discover successors immediately and update Need immediately.
                    if r.on_dispatched:
                        self._discover_successors_locked(r, need)
                elif action == "SWITCH" and plan and math.isfinite(sw_cost):
                    reserved |= set(plan.target_set)
                    need[r.model] = max(0, need.get(r.model, 0) - 1)
                    self._schedule_switch_locked(r, plan, now)
                    if r.on_dispatched:
                        self._discover_successors_locked(r, need)
                else:
                    self._requeue_locked(r)

    def _discover_successors_locked(self, r: Request, need: Dict[str, int]) -> None:
        try:
            succ_reqs = r.on_dispatched(r) or []
        except Exception:
            succ_reqs = []
        for sr in succ_reqs:
            if self._add_to_potential_locked(sr):
                need[sr.model] = need.get(sr.model, 0) + 1

    # -----------------
    # Cost helpers
    # -----------------

    def _need_counts_locked(self) -> Dict[str, int]:
        c: Dict[str, int] = {}
        for r in self._reqs.values():
            if r.state in (ReqState.POTENTIAL, ReqState.WAITING):
                c[r.model] = c.get(r.model, 0) + 1
        return c

    def _cost_run_locked(
        self,
        r: Request,
        reserved: Set[int],
        metrics_map: Dict[Tuple[int, ...], MetricsSnapshot],
    ) -> Tuple[float, Optional[Tuple[int, ...]]]:
        best, best_s = math.inf, None
        for s, inst in self.instances.items():
            if inst.state != InstState.ACTIVE or not inst.accept_new or inst.model != r.model:
                continue
            if any(g in reserved for g in s):
                continue
            c = metrics_map.get(s, MetricsSnapshot()).backlog_cost
            if c < best:
                best, best_s = c, s
        return best, best_s

    def _best_switch_locked(
        self,
        r: Request,
        need: Dict[str, int],
        reserved: Set[int],
        metrics_map: Dict[Tuple[int, ...], MetricsSnapshot],
    ) -> Tuple[float, Optional[SwitchPlan]]:
        info = self.models[r.model]
        best, best_plan = math.inf, None
        for s in self.gpu_sets:
            if len(s) < info.tp_min or any(g in reserved for g in s):
                continue
            c, p = self._simulate_switch_locked(s, r.model, need, metrics_map)
            if c < best:
                best, best_plan = c, p
        return best, best_plan

    def _simulate_switch_locked(
        self,
        target_set: Tuple[int, ...],
        target_model: str,
        need: Dict[str, int],
        metrics_map: Dict[Tuple[int, ...], MetricsSnapshot],
    ) -> Tuple[float, Optional[SwitchPlan]]:
        overlapped: List[Tuple[int, ...]] = []
        displaced: Set[str] = set()

        for s2, inst2 in self.instances.items():
            if inst2.state in (InstState.ACTIVE, InstState.DRAINING, InstState.SWITCHING) and not self._disjoint(s2, target_set):
                overlapped.append(s2)
                if inst2.model:
                    displaced.add(inst2.model)

        # C_drain = max(Q*tau)
        drain_cost = 0.0
        for s2 in overlapped:
            drain_cost = max(drain_cost, metrics_map.get(s2, MetricsSnapshot()).backlog_cost)

        # C_activate (wake vs load)
        is_resident = all(self.gpus[g].is_resident(target_model) for g in target_set)
        info_t = self.models[target_model]
        activate_kind = "wake" if is_resident else "load"
        activate_cost = info_t.t_wake_s if is_resident else info_t.t_load_s

        # C_evict
        evict_cost, evict_ops, evicted = self._evict_cost_locked(target_set, target_model, displaced, need)
        if not math.isfinite(evict_cost):
            return math.inf, None

        total = drain_cost + activate_cost + evict_cost
        return total, SwitchPlan(
            target_set=target_set,
            target_model=target_model,
            overlapped_sets=tuple(overlapped),
            displaced_models=tuple(sorted(displaced)),
            evict_ops=tuple(evict_ops),
            evicted_models=tuple(sorted(evicted)),
            activate_kind=activate_kind,
            est_drain_s=drain_cost,
            est_activate_s=float(activate_cost),
            est_evict_s=evict_cost,
            total_cost_s=total,
        )

    def _evict_cost_locked(
        self,
        target_set: Tuple[int, ...],
        target_model: str,
        displaced: Set[str],
        need: Dict[str, int],
    ) -> Tuple[float, List[Tuple[int, str]], Set[str]]:
        # Base cost: sleep displaced active models
        base_cost = 0.0
        for d in displaced:
            mi = self.models.get(d)
            if mi:
                base_cost += float(need.get(d, 0)) * float(mi.t_wake_s)

        if self._c1_satisfied_locked(target_set, target_model):
            return base_cost, [], set()

        # Scenario 2: must evict models with smallest Need(.), tie by LRU (oldest first).
        ops: List[Tuple[int, str]] = []
        evicted: Set[str] = set()

        tp_target = len(target_set)

        for g in target_set:
            cap = self.gpus[g].weight_cap_MB
            used = self._estimate_gpu_weight_used_MB_locked(g)

            if not self.gpus[g].is_resident(target_model):
                used += self._estimate_model_weight_MB_locked(target_model, tp_target, g)

            excess = used - cap
            if excess <= 1e-6:
                continue

            candidates = [m for m in self.gpus[g].resident.keys() if m != target_model]
            if not candidates:
                return math.inf, [], set()

            def key(m: str) -> Tuple[int, float]:
                n = int(need.get(m, 0))
                last = float(self.gpus[g].last_used.get(m, 0.0))  # older = smaller
                return (n, last)

            candidates.sort(key=key)

            for m in candidates:
                if excess <= 1e-6:
                    break
                tp_m = self.gpus[g].tp_by_model.get(m, self.models.get(m).tp_min if self.models.get(m) else 1)
                freed = self._estimate_model_weight_MB_locked(m, tp_m, g)
                ops.append((g, m))
                evicted.add(m)
                excess -= freed

            if excess > 1e-6:
                return math.inf, [], set()

        # Extra penalty for eviction vs sleep
        delta_cost = 0.0
        for e in evicted:
            mi = self.models.get(e)
            if mi:
                delta_cost += float(mi.t_offload_s) + float(need.get(e, 0)) * float(mi.t_load_s - mi.t_wake_s)

        return base_cost + delta_cost, ops, evicted

    def _c1_satisfied_locked(self, target_set: Tuple[int, ...], target_model: str) -> bool:
        tp_target = len(target_set)
        for g in target_set:
            used = self._estimate_gpu_weight_used_MB_locked(g)
            if not self.gpus[g].is_resident(target_model):
                used += self._estimate_model_weight_MB_locked(target_model, tp_target, g)
            if used > self.gpus[g].weight_cap_MB + 1e-6:
                return False
        return True

    def _estimate_gpu_weight_used_MB_locked(self, gpu_id: int) -> float:
        used = 0.0
        gi = self.gpus[gpu_id]
        for m in gi.resident.keys():
            tp = gi.tp_by_model.get(m, self.models.get(m).tp_min if self.models.get(m) else 1)
            used += self._estimate_model_weight_MB_locked(m, tp, gpu_id)
        return used

    def _estimate_model_weight_MB_locked(self, model: str, tp: int, gpu_id: int) -> float:
        pid = self.gpus[gpu_id].pid_by_model.get(model)
        if pid is not None:
            v = self._pid_mem.get((gpu_id, int(pid)))
            if v is not None:
                return float(v)
        mi = self.models.get(model)
        if not mi:
            return 4096.0 if int(tp) > 1 else 2048.0
        return float(mi.slept_mem_MB(int(tp)))

    # -----------------
    # Apply actions
    # -----------------

    def _dispatch_locked(self, r: Request, s: Tuple[int, ...], now: float) -> None:
        inst = self.instances[s]
        if inst.state != InstState.ACTIVE or inst.model != r.model or not inst.accept_new:
            self._requeue_locked(r)
            return
        r.state = ReqState.RUNNING
        for g in s:
            self.gpus[g].set_resident(r.model, Residence.ACTIVE, now, tp=len(s))
        self._spawn(self._run_infer(r, s))

    def _requeue_locked(self, r: Request) -> None:
        r.state = ReqState.WAITING
        self._push(self._waiting, r.t_arr, r.key())

    def _schedule_switch_locked(self, r: Request, plan: SwitchPlan, now: float) -> None:
        # Mark overlapped draining (no new requests)
        for s2 in plan.overlapped_sets:
            inst2 = self.instances[s2]
            inst2.accept_new = False
            if inst2.state == InstState.ACTIVE:
                inst2.state = InstState.DRAINING

        # Reserve target set
        inst_t = self.instances[plan.target_set]
        inst_t.state = InstState.SWITCHING
        inst_t.accept_new = False

        r.state = ReqState.RUNNING
        self._spawn(self._switch_then_run(r, plan))

    # -----------------
    # Background tasks
    # -----------------

    async def _run_infer(self, r: Request, s: Tuple[int, ...]) -> None:
        try:
            out = await self.controller.infer(r.model, r.payload, s)
        except Exception as e:
            out = {"error": str(e), "model": r.model, "gpu_set": list(s)}

        async with self._lock:
            r.state = ReqState.DONE

        if r.on_completed:
            try:
                r.on_completed(r, out)
            except Exception:
                pass

    async def _switch_then_run(self, r: Request, plan: SwitchPlan) -> None:
        # best-effort stop accepting new
        for s2 in plan.overlapped_sets:
            try:
                await self.controller.stop_accepting_new(s2)
            except Exception:
                pass

        # Drain (estimated)
        await asyncio.sleep(max(0.0, plan.est_drain_s))

        # Deactivate overlapped; apply evictions; stage target weights
        async with self._lock:
            now = time.monotonic()

            # sleep overlapped instances
            for s2 in plan.overlapped_sets:
                inst2 = self.instances[s2]
                if inst2.model:
                    u = inst2.model
                    self._spawn(self._observe_sleep(inst2))
                    for g in s2:
                        pid = inst2.pid_by_gpu.get(g)
                        self.gpus[g].set_resident(u, Residence.SLEPT, now, pid=pid, tp=len(s2))
                inst2.model = None
                inst2.state, inst2.accept_new = InstState.IDLE, True

            # evictions
            for g, m in plan.evict_ops:
                pid = self.gpus[g].pid_by_model.get(m)
                if pid is not None:
                    self._spawn(self._observe_offload(g, m, int(pid)))
                self.gpus[g].evict(m)

            # stage target
            tp_t = len(plan.target_set)
            for g in plan.target_set:
                self.gpus[g].set_resident(plan.target_model, Residence.SLEPT, now, tp=tp_t)

            # keep target instance in SWITCHING until activation completes
            inst_t = self.instances[plan.target_set]
            inst_t.model = plan.target_model
            inst_t.state, inst_t.accept_new = InstState.SWITCHING, False

        # Activation timing (EMA update)
        async with self._lock:
            is_resident_now = all(self.gpus[g].is_resident(plan.target_model) for g in plan.target_set)
        kind = "wake" if is_resident_now else "load"

        t0 = time.monotonic()
        mi = self.models[plan.target_model]
        await asyncio.sleep(max(0.0, mi.t_wake_s if kind == "wake" else mi.t_load_s))
        obs = time.monotonic() - t0

        async with self._lock:
            mi = self.models.get(plan.target_model)
            if mi:
                if kind == "wake":
                    mi.update_wake(obs, ema=self.timing_ema)
                else:
                    mi.update_load(obs, ema=self.timing_ema)

            now = time.monotonic()
            for g in plan.target_set:
                self.gpus[g].set_resident(plan.target_model, Residence.ACTIVE, now, tp=len(plan.target_set))

            inst_t = self.instances[plan.target_set]
            inst_t.state, inst_t.accept_new = InstState.ACTIVE, True

        await self._run_infer(r, plan.target_set)

    async def _observe_sleep(self, inst: Instance) -> None:
        if not inst.model:
            return
        model = inst.model
        tp = len(inst.gpus)

        t_obs = await self.controller.sleep_observe(inst)
        if t_obs is not None:
            mi = self.models.get(model)
            if mi:
                mi.update_wake(float(t_obs), ema=self.timing_ema)

        pid_mem = await self.controller.pid_used_MB()
        if not pid_mem:
            return
        vals: List[int] = []
        for g in inst.gpus:
            pid = inst.pid_by_gpu.get(g)
            if pid is None:
                continue
            v = pid_mem.get((g, int(pid)))
            if v is not None:
                vals.append(int(v))
        if not vals:
            return
        mi = self.models.get(model)
        if mi:
            mi.update_slept_mem(tp, float(sum(vals)) / float(len(vals)), ema=self.timing_ema)

    async def _observe_offload(self, gpu_id: int, model: str, pid: int) -> None:
        t_obs = await self.controller.offload_observe(gpu_id, pid)
        if t_obs is None:
            return
        mi = self.models.get(model)
        if mi:
            mi.update_offload(float(t_obs), ema=self.timing_ema)

    # -----------------
    # Utilities
    # -----------------

    def _push(self, heap: List[Tuple[float, int, str]], t: float, k: str) -> None:
        self._seq += 1
        heapq.heappush(heap, (t, self._seq, k))

    @staticmethod
    def _disjoint(a: Tuple[int, ...], b: Tuple[int, ...]) -> bool:
        sa = set(a)
        return all(x not in sa for x in b)

    def _spawn(self, coro: Any) -> None:
        t = asyncio.create_task(coro)
        self._bg.add(t)
        t.add_done_callback(lambda tt: self._bg.discard(tt))
