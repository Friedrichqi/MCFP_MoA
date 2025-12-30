from __future__ import annotations

import asyncio, heapq, math, time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple


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


@dataclass(frozen=True)
class ModelInfo:
    """
    Model card fields.
    - size_bytes: bf16 weights size
    - tp_min: minimum tensor parallel degree required
    - t_wake_s: slept->active
    - t_load_s: evicted->active
    - avg_service_s: mean per-request service time (from vLLM stats or a model card default)
    """
    name: str
    size_bytes: int
    tp_min: int
    t_wake_s: float
    t_load_s: float
    avg_service_s: float = 0.2

    @property
    def shard_bytes(self) -> float:
        # Uses the user's constraint formula: Size/TP_min
        return float(self.size_bytes) / float(self.tp_min)


@dataclass
class GPUInfo:
    gpu_id: int
    vram_total_bytes: int
    alpha: float

    # model -> ACTIVE/SLEPT, absent means evicted
    resident: Dict[str, Residence] = field(default_factory=dict)

    # model -> last used timestamp
    last_used: Dict[str, float] = field(default_factory=dict)

    @property
    def weight_cap(self) -> float:
        return self.alpha * float(self.vram_total_bytes)

    def is_resident(self, m: str) -> bool:
        return m in self.resident

    def set_state(self, m: str, st: Residence, now: float) -> None:
        self.resident[m] = st
        self.last_used[m] = now

    def evict(self, m: str) -> None:
        self.resident.pop(m, None)
        self.last_used.pop(m, None)


@dataclass
class Instance:
    """
    One "slot" per GPU set, can be IDLE/ACTIVE/DRAINING/SWITCHING.
    backlog_q and avg_time_s are used for C_run and C_drain.
    """
    gpus: Tuple[int, ...]
    state: InstState = InstState.IDLE
    model: Optional[str] = None
    backlog_q: int = 0
    avg_time_s: float = 0.2
    accept_new: bool = True


@dataclass
class Request:
    """
    One schedulable unit = one DAG vertex (vertex == model name).
    indegree drives readiness (0 => ready).
    """
    job_id: str
    node_id: str
    model: str
    t_arr: float
    indegree: int
    succ: List[str] = field(default_factory=list)
    payload: Any = None
    state: ReqState = ReqState.POTENTIAL

    # on_dispatched returns "newly discovered successors" (Requests) to insert into POTENTIAL immediately.
    on_dispatched: Optional[Callable[[Request], List["Request"]]] = None
    on_completed: Optional[Callable[[Request, Any], None]] = None

    def key(self) -> str:
        return f"{self.job_id}:{self.node_id}"


@dataclass(frozen=True)
class SwitchPlan:
    target_set: Tuple[int, ...]
    target_model: str
    overlapped_sets: Tuple[Tuple[int, ...], ...]
    displaced_models: Tuple[str, ...]
    evict_ops: Tuple[Tuple[int, str], ...]     # (gpu, model) evictions to apply
    evicted_models: Tuple[str, ...]            # unique models evicted
    est_drain_s: float
    est_activate_s: float
    est_evict_s: float
    total_cost_s: float


class VLLMController:
    """
    Extension point: replace SimVLLMController with a real adapter that talks to vLLM instances.
    """
    async def infer(self, model: str, payload: Any, gpu_set: Tuple[int, ...]) -> Any:
        raise NotImplementedError

    async def stop_accepting_new(self, gpu_set: Tuple[int, ...]) -> None:
        return None


class SimVLLMController(VLLMController):
    """
    End-to-end runnable simulator (no vLLM needed).
    """
    def __init__(self, models: Dict[str, ModelInfo]):
        self.models = models

    async def infer(self, model: str, payload: Any, gpu_set: Tuple[int, ...]) -> Any:
        await asyncio.sleep(self.models[model].avg_service_s)
        return {"model": model, "gpu_set": list(gpu_set), "ok": True, "payload": payload}


class moa_scheduler:
    """
    POTENTIAL/WAITING PQs and cost-based choice among:
      - RUN: dispatch to an active instance serving req.model
      - WAIT: keep waiting for this step
      - SWITCH: drain overlapped sets, activate req.model on some GPU set

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
              C_evict = base_sleep + sum_{e evicted} Need(e)*(Tload(e)-Twake(e))

    Need(x) counts models over W ∪ P at decision time.
    """

    def __init__(
        self,
        *,
        gpus: List[GPUInfo],
        gpu_sets: Sequence[Tuple[int, ...]],
        models: Dict[str, ModelInfo],
        alpha: float = 0.8,
        beta: float = 1.0,
        controller: Optional[VLLMController] = None,
        max_decisions_per_step: int = 128,
    ) -> None:
        self.alpha, self.beta = alpha, beta
        self.models = models

        self.gpus: Dict[int, GPUInfo] = {g.gpu_id: g for g in gpus}
        self.gpu_sets: List[Tuple[int, ...]] = [tuple(s) for s in gpu_sets]
        self.instances: Dict[Tuple[int, ...], Instance] = {tuple(s): Instance(tuple(s)) for s in self.gpu_sets}

        self._reqs: Dict[str, Request] = {}
        self._potential: List[Tuple[float, int, str]] = []
        self._waiting: List[Tuple[float, int, str]] = []
        self._seq = 0

        self.controller = controller or SimVLLMController(models)
        self._lock = asyncio.Lock()
        self._bg: Set[asyncio.Task] = set()
        self.max_decisions_per_step = max_decisions_per_step

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
                self._reqs[k].indegree = new_indegree

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
            reserved: Set[int] = set()

        for k in batch:
            async with self._lock:
                r = self._reqs.get(k)
                if not r or r.state != ReqState.WAITING:
                    continue
                if r.model not in self.models:
                    self._requeue_locked(r)
                    continue

                run_cost, run_set = self._cost_run_locked(r, reserved)
                wait_cost = self.beta * max(0.0, now - r.t_arr)
                sw_cost, plan = self._best_switch_locked(r, need, reserved)

                action = min([("RUN", run_cost), ("WAIT", wait_cost), ("SWITCH", sw_cost)], key=lambda x: x[1])[0]

                if action == "RUN" and run_set and math.isfinite(run_cost):
                    # r leaves W -> update Need
                    need[r.model] = max(0, need.get(r.model, 0) - 1)

                    self._dispatch_locked(r, run_set, now)

                    # Discover successors immediately and update Need immediately.
                    if r.on_dispatched:
                        try:
                            succ_reqs = r.on_dispatched(r) or []
                        except Exception:
                            succ_reqs = []
                        for sr in succ_reqs:
                            if self._add_to_potential_locked(sr):
                                need[sr.model] = need.get(sr.model, 0) + 1

                elif action == "SWITCH" and plan and math.isfinite(sw_cost):
                    reserved |= set(plan.target_set)
                    need[r.model] = max(0, need.get(r.model, 0) - 1)

                    self._schedule_switch_locked(r, plan, now)

                    if r.on_dispatched:
                        try:
                            succ_reqs = r.on_dispatched(r) or []
                        except Exception:
                            succ_reqs = []
                        for sr in succ_reqs:
                            if self._add_to_potential_locked(sr):
                                need[sr.model] = need.get(sr.model, 0) + 1

                else:
                    self._requeue_locked(r)

    # -----------------
    # Cost helpers
    # -----------------

    def _need_counts_locked(self) -> Dict[str, int]:
        c: Dict[str, int] = {}
        for r in self._reqs.values():
            if r.state in (ReqState.POTENTIAL, ReqState.WAITING):
                c[r.model] = c.get(r.model, 0) + 1
        return c

    def _cost_run_locked(self, r: Request, reserved: Set[int]) -> Tuple[float, Optional[Tuple[int, ...]]]:
        best, best_s = math.inf, None
        for s, inst in self.instances.items():
            if inst.state != InstState.ACTIVE or not inst.accept_new or inst.model != r.model:
                continue
            if any(g in reserved for g in s):
                continue
            c = float(inst.backlog_q) * float(inst.avg_time_s)
            if c < best:
                best, best_s = c, s
        return best, best_s

    def _best_switch_locked(self, r: Request, need: Dict[str, int], reserved: Set[int]) -> Tuple[float, Optional[SwitchPlan]]:
        info = self.models[r.model]
        best, best_plan = math.inf, None
        for s in self.gpu_sets:
            if len(s) < info.tp_min or any(g in reserved for g in s):
                continue
            c, p = self._simulate_switch_locked(s, r.model, need)
            if c < best:
                best, best_plan = c, p
        return best, best_plan

    def _simulate_switch_locked(self, target_set: Tuple[int, ...], target_model: str, need: Dict[str, int]) -> Tuple[float, Optional[SwitchPlan]]:
        overlapped: List[Tuple[int, ...]] = []
        displaced: Set[str] = set()

        for s2, inst2 in self.instances.items():
            if inst2.state in (InstState.ACTIVE, InstState.DRAINING, InstState.SWITCHING) and not self._disjoint(s2, target_set):
                overlapped.append(s2)
                if inst2.model:
                    displaced.add(inst2.model)

        # C_drain = max(Q*tau)
        drain = 0.0
        for s2 in overlapped:
            inst2 = self.instances[s2]
            drain = max(drain, float(inst2.backlog_q) * float(inst2.avg_time_s))

        # C_activate
        info_t = self.models[target_model]
        activate = info_t.t_wake_s if all(self.gpus[g].is_resident(target_model) for g in target_set) else info_t.t_load_s

        # C_evict
        evict_cost, evict_ops, evicted = self._evict_cost_locked(target_set, target_model, displaced, need)
        if not math.isfinite(evict_cost):
            return math.inf, None

        total = drain + activate + evict_cost
        return total, SwitchPlan(
            target_set=target_set,
            target_model=target_model,
            overlapped_sets=tuple(overlapped),
            displaced_models=tuple(sorted(displaced)),
            evict_ops=tuple(evict_ops),
            evicted_models=tuple(sorted(evicted)),
            est_drain_s=drain,
            est_activate_s=float(activate),
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
        # Base: sleeping displaced active models
        base = 0.0
        for u in displaced:
            if u in self.models:
                base += float(need.get(u, 0)) * float(self.models[u].t_wake_s)

        # Scenario 1: check if C1 holds without eviction
        for g in target_set:
            used = self._weight_used_locked(g) + (0.0 if self.gpus[g].is_resident(target_model) else self.models[target_model].shard_bytes)
            if used > self.gpus[g].weight_cap + 1e-6:
                break
        else:
            return base, [], set()

        # Scenario 2: must evict. Policy: smallest Need(.)
        ops: List[Tuple[int, str]] = []
        evicted: Set[str] = set()

        for g in target_set:
            used = self._weight_used_locked(g) + (0.0 if self.gpus[g].is_resident(target_model) else self.models[target_model].shard_bytes)
            excess = used - self.gpus[g].weight_cap
            if excess <= 1e-6:
                continue

            candidates = [m for m in self.gpus[g].resident.keys() if m != target_model]
            if not candidates:
                return math.inf, [], set()

            def key(m: str) -> Tuple[int, float, float]:
                # primary: smallest Need
                n = int(need.get(m, 0))
                # tie: smaller delta (Tload-Twake)
                delta = float(self.models[m].t_load_s - self.models[m].t_wake_s) if m in self.models else 0.0
                # tie: older LRU
                last = float(self.gpus[g].last_used.get(m, 0.0))
                return (n, delta, last)

            candidates.sort(key=key)

            for m in candidates:
                if excess <= 1e-6:
                    break
                shard = self.models[m].shard_bytes if m in self.models else 0.0
                if shard <= 0:
                    continue
                ops.append((g, m))
                evicted.add(m)
                excess -= shard

            if excess > 1e-6:
                return math.inf, [], set()

        # Extra penalty for eviction vs sleep
        delta_cost = 0.0
        for e in evicted:
            if e in self.models:
                delta_cost += float(need.get(e, 0)) * float(self.models[e].t_load_s - self.models[e].t_wake_s)

        return base + delta_cost, ops, evicted

    # -----------------
    # Apply actions
    # -----------------

    def _dispatch_locked(self, r: Request, s: Tuple[int, ...], now: float) -> None:
        inst = self.instances[s]
        if inst.state != InstState.ACTIVE or inst.model != r.model:
            self._requeue_locked(r)
            return
        inst.backlog_q += 1
        r.state = ReqState.RUNNING
        for g in s:
            self.gpus[g].set_state(r.model, Residence.ACTIVE, now)
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
        self._spawn(self._switch_then_run(r, plan, now))

    # -----------------
    # Background tasks
    # -----------------

    async def _run_infer(self, r: Request, s: Tuple[int, ...]) -> None:
        try:
            out = await self.controller.infer(r.model, r.payload, s)
        except Exception as e:
            out = {"error": str(e), "model": r.model, "gpu_set": list(s)}

        async with self._lock:
            inst = self.instances.get(s)
            if inst:
                inst.backlog_q = max(0, inst.backlog_q - 1)
            r.state = ReqState.DONE

        if r.on_completed:
            try:
                r.on_completed(r, out)
            except Exception:
                pass

    async def _switch_then_run(self, r: Request, plan: SwitchPlan, now0: float) -> None:
        # best-effort stop accepting new
        for s2 in plan.overlapped_sets:
            try:
                await self.controller.stop_accepting_new(s2)
            except Exception:
                pass

        # wait drain estimate
        await asyncio.sleep(max(0.0, plan.est_drain_s))

        async with self._lock:
            now = time.monotonic()

            # deactivate overlapped instances -> sleep their active model
            for s2 in plan.overlapped_sets:
                inst2 = self.instances[s2]
                if inst2.model:
                    u = inst2.model
                    for g in s2:
                        self.gpus[g].set_state(u, Residence.SLEPT, now)
                inst2.model, inst2.backlog_q = None, 0
                inst2.state, inst2.accept_new = InstState.IDLE, True

            # apply evictions
            for g, m in plan.evict_ops:
                self.gpus[g].evict(m)

            # ensure target weights are resident (but not serving yet)
            for g in plan.target_set:
                if not self.gpus[g].is_resident(plan.target_model):
                    self.gpus[g].set_state(plan.target_model, Residence.SLEPT, now)
                else:
                    self.gpus[g].set_state(plan.target_model, Residence.SLEPT, now)

            # keep target instance in SWITCHING until activation completes
            inst_t = self.instances[plan.target_set]
            inst_t.model = plan.target_model
            inst_t.avg_time_s = self.models[plan.target_model].avg_service_s
            inst_t.backlog_q = 1          # reserve one slot for r
            inst_t.state, inst_t.accept_new = InstState.SWITCHING, False

        # simulate wake/load time
        await asyncio.sleep(max(0.0, plan.est_activate_s))

        # now mark the instance active and weights active
        async with self._lock:
            now = time.monotonic()
            for g in plan.target_set:
                self.gpus[g].set_state(plan.target_model, Residence.ACTIVE, now)
            inst_t = self.instances[plan.target_set]
            inst_t.state, inst_t.accept_new = InstState.ACTIVE, True

        await self._run_infer(r, plan.target_set)

    # -----------------
    # Utils
    # -----------------

    def _weight_used_locked(self, gpu_id: int) -> float:
        g = self.gpus[gpu_id]
        return sum(self.models[m].shard_bytes for m in g.resident.keys() if m in self.models)

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
