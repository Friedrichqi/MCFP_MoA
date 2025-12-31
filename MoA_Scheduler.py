from __future__ import annotations

import asyncio, heapq, math, time, subprocess, re, json, os, signal, subprocess, httpx
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
    ACTIVE = "active"
    SLEPT = "slept"
    DRAINING = "draining"
    SWITCHING = "switching"
    DEAD = "dead"


class ReqState(str, Enum):
    POTENTIAL = "potential"
    WAITING = "waiting"
    RUNNING = "running"
    DONE = "done"


@dataclass
class ModelInfo:
    """Mutable model-card entry + online stats.

    EMAs:
      - t_wake_s   : wake a slept instance
      - t_sleep_s  : sleep an instance
      - t_load_s   : start vLLM from scratch until ready
      - t_offload_s: kill instance (offload/evict)

    Slept footprint is kept in MB per GPU, updated from nvidia-smi after sleep.

    Defaults:
      - slept_mem_MB_tp1   = 2048
      - slept_mem_MB_tpgt1 = 4096
      - t_wake_s=t_sleep_s=2, t_load_s=90, t_offload_s=3
    """

    name: str
    tp_min: int

    t_wake_s: float = 2.0
    t_sleep_s: float = 2.0
    t_load_s: float = 90.0
    t_offload_s: float = 3.0

    slept_mem_MB_tp1: float = 2048.0
    slept_mem_MB_tpgt1: float = 4096.0

    avg_service_s: float = 0.2

    def slept_mem_MB(self, tp: int) -> float:
        return float(self.slept_mem_MB_tp1 if int(tp) <= 1 else self.slept_mem_MB_tpgt1)

    @staticmethod
    def _ema(old: float, new: float, ema: float) -> float:
        return max(1e-3, (1.0 - ema) * float(old) + ema * float(new))

    def update_wake(self, observed_s: float, ema: float = 0.2) -> None:
        self.t_wake_s = self._ema(self.t_wake_s, observed_s, ema)

    def update_sleep(self, observed_s: float, ema: float = 0.2) -> None:
        self.t_sleep_s = self._ema(self.t_sleep_s, observed_s, ema)

    def update_load(self, observed_s: float, ema: float = 0.2) -> None:
        self.t_load_s = self._ema(self.t_load_s, observed_s, ema)

    def update_offload(self, observed_s: float, ema: float = 0.2) -> None:
        self.t_offload_s = self._ema(self.t_offload_s, observed_s, ema)

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

    resident: Dict[str, Residence] = field(default_factory=dict)  # model -> ACTIVE/SLEPT
    pid_by_model: Dict[str, int] = field(default_factory=dict)  # model -> pid on this GPU
    url_by_model: Dict[str, str] = field(default_factory=dict)  # model -> base_url
    tp_by_model: Dict[str, int] = field(default_factory=dict)  # model -> tp used on this GPU
    last_used: Dict[str, float] = field(default_factory=dict)  # model -> monotonic timestamp

    @property
    def weight_cap_MB(self) -> float:
        return self.alpha * float(self.vram_total_MB)

    def set_resident(
        self,
        model: str,
        st: Residence,
        now: float,
        *,
        pid: Optional[int] = None,
        tp: Optional[int] = None,
        url: Optional[str] = None,
    ) -> None:
        self.resident[model] = st
        self.last_used[model] = now
        if pid is not None:
            self.pid_by_model[model] = int(pid)
        if tp is not None:
            self.tp_by_model[model] = int(tp)
        if url is not None:
            self.url_by_model[model] = str(url)

    def evict_model(self, model: str) -> None:
        self.resident.pop(model, None)
        self.pid_by_model.pop(model, None)
        self.url_by_model.pop(model, None)
        self.tp_by_model.pop(model, None)
        self.last_used.pop(model, None)

    def is_resident(self, model: str) -> bool:
        return model in self.resident


@dataclass
class Instance:
    """A single vLLM server process bound to a GPU set and a model."""

    instance_id: str
    gpus: Tuple[int, ...]
    model: str
    base_url: str
    metrics_url: str
    state: InstState = InstState.ACTIVE
    accept_new: bool = True
    pid_by_gpu: Dict[int, int] = field(default_factory=dict)
    created_at: float = field(default_factory=lambda: time.monotonic())


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
    on_dispatched: Optional[Callable[["Request"], List["Request"]]] = None
    on_completed: Optional[Callable[["Request", Any], None]] = None

    def key(self) -> str:
        return f"{self.job_id}:{self.node_id}"


@dataclass(frozen=True)
class SwitchPlan:
    target_set: Tuple[int, ...]
    target_model: str
    overlapped_active_ids: Tuple[str, ...]
    displaced_models: Tuple[str, ...]
    evict_models: Tuple[str, ...]
    activate_kind: str  # "wake" or "load"
    est_drain_s: float
    est_activate_s: float
    est_sleep_now_s: float
    est_offload_now_s: float
    est_future_penalty_s: float
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
    async def metrics(self, inst: Instance) -> MetricsSnapshot:
        return MetricsSnapshot()

    async def pid_used_MB(self) -> Dict[Tuple[int, int], int]:
        return {}

    async def infer(self, inst: Instance, payload: Any) -> Any:
        raise NotImplementedError

    async def start(self, model: str, gpu_set: Tuple[int, ...], tp: int, gpu_mem_util: float) -> Instance:
        raise NotImplementedError

    async def sleep(self, inst: Instance) -> float:
        raise NotImplementedError

    async def wake(self, inst: Instance) -> float:
        raise NotImplementedError

    async def kill(self, inst: Instance) -> float:
        raise NotImplementedError

    async def drain_until_empty(self, inst: Instance, *, poll_s: float = 0.2, timeout_s: float = 600.0) -> float:
        """Wait until num_requests==0. Returns elapsed."""
        t0 = time.monotonic()
        while True:
            snap = await self.metrics(inst)
            if snap.num_requests <= 0:
                return time.monotonic() - t0
            if time.monotonic() - t0 > timeout_s:
                return time.monotonic() - t0
            await asyncio.sleep(poll_s)


# -----------------------------
# Managed vLLM controller (real servers)
# -----------------------------

class ManagedVLLMController(VLLMController):
    """Spawns and controls vLLM servers via subprocess + HTTP.

    Launch:
      CUDA_VISIBLE_DEVICES=<gpu-set> VLLM_SERVER_DEV_MODE=1 \
        vllm serve <model> --tensor-parallel-size <tp> --port <port> \
          --enable-sleep-mode --gpu-memory-utilization <util>

    Sleep:
      POST {base_url}/sleep?level=2
    Wake:
      POST {base_url}/wake_up

    Offload/Evict:
      kill -9 all known per-GPU pids; then wait until gone from nvidia-smi.

    Metrics parsing:
      vllm:num_requests
      vllm:e2e_request_latency_seconds_(sum|count)
    """

    def __init__(
        self,
        *,
        host: str = "127.0.0.1",
        port_base: int = 9000,
        request_timeout_s: float = 300.0,
        startup_timeout_s: float = 600.0,
        vllm_extra_args: Optional[List[str]] = None,
    ) -> None:
        self.host = host
        self.port_base = int(port_base)
        self.request_timeout_s = float(request_timeout_s)
        self.startup_timeout_s = float(startup_timeout_s)
        self.vllm_extra_args = list(vllm_extra_args or [])

        self._next_port = self.port_base
        self._procs: Dict[str, subprocess.Popen] = {}

    # ---------- HTTP helpers ----------

    async def _http_get_text(self, url: str) -> str:
        async with httpx.AsyncClient(timeout=self.request_timeout_s) as client:
            r = await client.get(url)
            r.raise_for_status()
            return r.text
        return await asyncio.to_thread(lambda: subprocess.check_output(["curl", "-s", url], text=True))

    async def _http_post_json(self, url: str, json_body: Dict[str, Any]) -> Any:
        async with httpx.AsyncClient(timeout=self.request_timeout_s) as client:
            r = await client.post(url, json=json_body)
            r.raise_for_status()
            return r.json()
        payload = json.dumps(json_body)
        out = await asyncio.to_thread(
            lambda: subprocess.check_output(
                ["curl", "-s", "-X", "POST", url, "-H", "Content-Type: application/json", "-d", payload],
                text=True,
            )
        )
        try:
            return json.loads(out)
        except Exception:
            return {"raw": out}

    async def _http_post_no_body(self, url: str) -> str:
        async with httpx.AsyncClient(timeout=self.request_timeout_s) as client:
            r = await client.post(url)
            r.raise_for_status()
            return r.text
        return await asyncio.to_thread(lambda: subprocess.check_output(["curl", "-s", "-X", "POST", url], text=True))

    # ---------- metrics parsing ----------

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

    async def metrics(self, inst: Instance) -> MetricsSnapshot:
        url = inst.metrics_url
        try:
            raw = await self._http_get_text(url)
        except Exception:
            return MetricsSnapshot()

        targets = {
            "vllm:num_requests",
            "vllm:e2e_request_latency_seconds_sum",
            "vllm:e2e_request_latency_seconds_count",
        }

        samples: Dict[str, List[Tuple[Dict[str, str], float]]] = {k: [] for k in targets}
        for ln in raw.splitlines():
            parsed = self._parse_prom_line(ln)
            if not parsed:
                continue
            name, labels, val = parsed
            if name in samples:
                samples[name].append((labels, val))

        def pick(name: str) -> float:
            arr = samples.get(name, [])
            if not arr:
                return 0.0
            for labels, v in arr:
                if labels.get("model_name") == inst.model:
                    return float(v)
            pids = {str(pid) for pid in inst.pid_by_gpu.values() if pid}
            for labels, v in arr:
                if labels.get("pid") in pids:
                    return float(v)
            return float(arr[0][1])

        return MetricsSnapshot(
            num_requests=pick("vllm:num_requests"),
            e2e_sum=pick("vllm:e2e_request_latency_seconds_sum"),
            e2e_count=pick("vllm:e2e_request_latency_seconds_count"),
        )

    # ---------- nvidia-smi pid memory ----------

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
                [
                    "nvidia-smi",
                    "--query-compute-apps=gpu_uuid,pid,used_memory",
                    "--format=csv,noheader,nounits",
                ],
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

    # ---------- lifecycle ----------

    def _alloc_port(self) -> int:
        while True:
            p = self._next_port
            self._next_port += 1
            if self._port_free(p):
                return p

    @staticmethod
    def _port_free(port: int) -> bool:
        import socket

        s = socket.socket()
        try:
            s.bind(("127.0.0.1", port))
            return True
        except Exception:
            return False
        finally:
            try:
                s.close()
            except Exception:
                pass

    @staticmethod
    def _gpu_used_total_MB(gpu_ids: Sequence[int]) -> Dict[int, Tuple[int, int]]:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,memory.used,memory.total", "--format=csv,noheader,nounits"],
            text=True,
        ).strip()
        m: Dict[int, Tuple[int, int]] = {}
        for ln in out.splitlines():
            idx_s, used_s, total_s = [x.strip() for x in ln.split(",")]
            idx = int(idx_s)
            if idx in gpu_ids:
                m[idx] = (int(float(used_s)), int(float(total_s)))
        return m

    @classmethod
    def compute_gpu_mem_util(cls, gpu_set: Tuple[int, ...]) -> float:
        try:
            stats = cls._gpu_used_total_MB(list(gpu_set))
            if not stats:
                return 0.9
            ratios = []
            for _, (used, total) in stats.items():
                free = max(0, total - used)
                ratios.append(float(free) / float(total) if total > 0 else 0.0)
            free_ratio = min(ratios) if ratios else 0.5
            util = max(0.05, min(0.95, free_ratio * 0.9))
            return float(util)
        except Exception:
            return 0.9

    @staticmethod
    def _snapshot_pids_by_gpu(gpu_set: Tuple[int, ...]) -> Dict[int, Set[int]]:
        try:
            out = subprocess.check_output(
                ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader,nounits"],
                text=True,
            ).strip()
        except Exception:
            return {g: set() for g in gpu_set}

        uuid_to_idx: Dict[str, int] = {}
        try:
            out2 = subprocess.check_output(["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"], text=True).strip()
            for ln in out2.splitlines():
                idx, uuid = [x.strip() for x in ln.split(",")]
                uuid_to_idx[uuid] = int(idx)
        except Exception:
            uuid_to_idx = {}

        m: Dict[int, Set[int]] = {g: set() for g in gpu_set}
        for ln in out.splitlines():
            parts = [x.strip() for x in ln.split(",")]
            if len(parts) != 2:
                continue
            gpu_uuid, pid_s = parts
            if gpu_uuid not in uuid_to_idx:
                continue
            gid = uuid_to_idx[gpu_uuid]
            if gid not in m:
                continue
            try:
                m[gid].add(int(pid_s))
            except Exception:
                continue
        return m

    async def start(self, model: str, gpu_set: Tuple[int, ...], tp: int, gpu_mem_util: float) -> Instance:
        port = self._alloc_port()
        base_url = f"http://{self.host}:{port}"
        metrics_url = f"{base_url}/metrics"
        instance_id = f"{','.join(map(str, gpu_set))}|{model}|{port}"

        before = await asyncio.to_thread(self._snapshot_pids_by_gpu, gpu_set)

        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = ",".join(str(x) for x in gpu_set)
        env["VLLM_SERVER_DEV_MODE"] = "1"

        cmd = [
            "vllm",
            "serve",
            model,
            "--host",
            self.host,
            "--port",
            str(port),
            "--tensor-parallel-size",
            str(tp),
            "--enable-sleep-mode",
            "--gpu-memory-utilization",
            str(gpu_mem_util),
        ] + self.vllm_extra_args

        proc = subprocess.Popen(
            cmd,
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        self._procs[instance_id] = proc

        t0 = time.monotonic()
        while True:
            if proc.poll() is not None:
                raise RuntimeError(f"vllm serve exited early for {instance_id} (rc={proc.returncode})")
            try:
                txt = await self._http_get_text(metrics_url)
                if len(txt) > 0:
                    break
            except Exception:
                pass
            if time.monotonic() - t0 > self.startup_timeout_s:
                raise TimeoutError(f"timeout waiting for vllm server {instance_id} ready")
            await asyncio.sleep(0.5)

        after = await asyncio.to_thread(self._snapshot_pids_by_gpu, gpu_set)
        pid_by_gpu: Dict[int, int] = {}
        for g in gpu_set:
            new = list(after.get(g, set()) - before.get(g, set()))
            pid_by_gpu[g] = int(new[0]) if new else int(proc.pid)

        return Instance(
            instance_id=instance_id,
            gpus=tuple(gpu_set),
            model=model,
            base_url=base_url,
            metrics_url=metrics_url,
            state=InstState.ACTIVE,
            accept_new=True,
            pid_by_gpu=pid_by_gpu,
        )

    async def sleep(self, inst: Instance) -> float:
        t0 = time.monotonic()
        await self._http_post_no_body(f"{inst.base_url}/sleep?level=2")
        return time.monotonic() - t0

    async def wake(self, inst: Instance) -> float:
        t0 = time.monotonic()
        await self._http_post_no_body(f"{inst.base_url}/wake_up")
        return time.monotonic() - t0

    async def kill(self, inst: Instance) -> float:
        t0 = time.monotonic()
        pids = {int(p) for p in inst.pid_by_gpu.values() if p}
        for pid in list(pids):
            try:
                os.kill(pid, signal.SIGKILL)
            except Exception:
                pass
        proc = self._procs.get(inst.instance_id)
        if proc and proc.poll() is None:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass

        while True:
            pid_map = await self.pid_used_MB()
            alive = any((g, pid) in pid_map for g in inst.gpus for pid in pids)
            if not alive:
                break
            if time.monotonic() - t0 > 120.0:
                break
            await asyncio.sleep(0.2)

        return time.monotonic() - t0

    async def infer(self, inst: Instance, payload: Any) -> Any:
        # endpoint selection
        if not isinstance(payload, dict):
            payload = {"messages": [{"role": "user", "content": str(payload)}]}

        endpoint = payload.pop("endpoint", None)
        if endpoint is None:
            if "messages" in payload:
                endpoint = "/v1/chat/completions"
            elif "prompt" in payload:
                endpoint = "/v1/completions"
            else:
                endpoint = "/v1/chat/completions"
                payload = {"messages": [{"role": "user", "content": json.dumps(payload)}]}

        payload.setdefault("model", inst.model)
        url = inst.base_url.rstrip("/") + endpoint
        return await self._http_post_json(url, payload)


# -----------------------------
# Scheduler core
# -----------------------------

class moa_scheduler:
    """MoA Scheduler with real vLLM management.

    - Dispatch cost uses online /metrics snapshot: C_run = num_requests * avg_latency.
    - Wait cost: C_wait = (now - t_arr) * beta.
    - Switch cost (your latest): C_switch = C_drain + C_activate + C_evict, implemented as:
        - C_drain: max backlog_cost among overlapped active instances.
        - C_activate: Twake if slept instance exists on that set; else Tload.
        - C_evict:
            * current-time: sum Tsleep(displaced but kept) + sum Toffload(evicted)
            * future penalty: sum Need(x)*Twake(x) for displaced kept + add Need(x)*(Tload-Twake) for evicted

    C1 is checked per-GPU, based on nvidia-smi pid memory when available, else model-card slept footprint.
    """

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

        self.instances: Dict[str, Instance] = {}
        self.active_ids: Set[str] = set()
        self.slept_ids: Set[str] = set()

        self._reqs: Dict[str, Request] = {}
        self._potential: List[Tuple[float, int, str]] = []
        self._waiting: List[Tuple[float, int, str]] = []
        self._seq = 0

        self.controller = controller or ManagedVLLMController()
        self._lock = asyncio.Lock()
        self._bg: Set[asyncio.Task] = set()
        self.max_decisions_per_step = int(max_decisions_per_step)

        self._pid_mem: Dict[Tuple[int, int], int] = {}

    # -----------------
    # External instance registration (optional)
    # -----------------

    async def register_existing_instance(self, inst: Instance) -> None:
        """Attach an already-running vLLM server."""
        now = time.monotonic()
        async with self._lock:
            self.instances[inst.instance_id] = inst
            inst.state = InstState.ACTIVE
            self.active_ids.add(inst.instance_id)
            self.slept_ids.discard(inst.instance_id)

            tp = len(inst.gpus)
            for g in inst.gpus:
                pid = inst.pid_by_gpu.get(g)
                self.gpus[g].set_resident(inst.model, Residence.ACTIVE, now, pid=pid, tp=tp, url=inst.base_url)

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
            active_insts = [self.instances[iid] for iid in self.active_ids if iid in self.instances]

        # snapshots outside lock
        metrics_map: Dict[str, MetricsSnapshot] = {}
        if active_insts:
            snaps = await asyncio.gather(*[self.controller.metrics(inst) for inst in active_insts], return_exceptions=True)
            for inst, snap in zip(active_insts, snaps):
                metrics_map[inst.instance_id] = snap if not isinstance(snap, Exception) else MetricsSnapshot()

        pid_mem = await self.controller.pid_used_MB()
        async with self._lock:
            if pid_mem:
                self._pid_mem = pid_mem

            reserved_gpus: Set[int] = set()

        for k in batch:
            async with self._lock:
                r = self._reqs.get(k)
                if not r or r.state != ReqState.WAITING:
                    continue

                if r.model not in self.models:
                    self._requeue_locked(r)
                    continue

                run_cost, run_iid = self._cost_run_locked(r, reserved_gpus, metrics_map)
                wait_cost = self.beta * max(0.0, now - r.t_arr)
                sw_cost, plan = self._best_switch_locked(r, need, reserved_gpus, metrics_map)

                action = min([("RUN", run_cost), ("WAIT", wait_cost), ("SWITCH", sw_cost)], key=lambda x: x[1])[0]

                if action == "RUN" and run_iid and math.isfinite(run_cost):
                    need[r.model] = max(0, need.get(r.model, 0) - 1)
                    # IMPORTANT: prepare payload + discover successors BEFORE sending request.
                    if r.on_dispatched:
                        self._discover_successors_locked(r, need)
                    self._dispatch_locked(r, run_iid, now)
                elif action == "SWITCH" and plan and math.isfinite(sw_cost):
                    reserved_gpus |= set(plan.target_set)
                    need[r.model] = max(0, need.get(r.model, 0) - 1)
                    # IMPORTANT: prepare payload + discover successors BEFORE sending request.
                    if r.on_dispatched:
                        self._discover_successors_locked(r, need)
                    self._schedule_switch_locked(r, plan, now)
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
        metrics_map: Dict[str, MetricsSnapshot],
    ) -> Tuple[float, Optional[str]]:
        best, best_id = math.inf, None
        for iid in self.active_ids:
            inst = self.instances.get(iid)
            if not inst or inst.model != r.model or inst.state != InstState.ACTIVE or not inst.accept_new:
                continue
            if any(g in reserved for g in inst.gpus):
                continue
            c = metrics_map.get(iid, MetricsSnapshot()).backlog_cost
            if c < best:
                best, best_id = c, iid
        return best, best_id

    def _best_switch_locked(
        self,
        r: Request,
        need: Dict[str, int],
        reserved: Set[int],
        metrics_map: Dict[str, MetricsSnapshot],
    ) -> Tuple[float, Optional[SwitchPlan]]:
        info = self.models[r.model]
        best, best_plan = math.inf, None
        for s in self.gpu_sets:
            if len(s) < info.tp_min:
                continue
            if any(g in reserved for g in s):
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
        metrics_map: Dict[str, MetricsSnapshot],
    ) -> Tuple[float, Optional[SwitchPlan]]:
        overlapped_ids: List[str] = []
        displaced_models: List[str] = []

        for iid in self.active_ids:
            inst = self.instances.get(iid)
            if not inst or inst.state != InstState.ACTIVE:
                continue
            if not self._disjoint(inst.gpus, target_set):
                overlapped_ids.append(iid)
                displaced_models.append(inst.model)

        # C_drain = max(Q*tau)
        drain_cost = 0.0
        for iid in overlapped_ids:
            drain_cost = max(drain_cost, metrics_map.get(iid, MetricsSnapshot()).backlog_cost)

        # C_activate (wake vs load)
        slept_iid = self._find_slept_instance_locked(target_model, target_set)
        activate_kind = "wake" if slept_iid else "load"
        info_t = self.models[target_model]
        activate_cost = info_t.t_wake_s if slept_iid else info_t.t_load_s

        future_penalty = 0.0
        for m in displaced_models:
            mi = self.models.get(m)
            if mi:
                future_penalty += float(need.get(m, 0)) * float(mi.t_wake_s)

        ok, to_evict = self._choose_evictions_locked(target_set, target_model, need)
        if not ok:
            return math.inf, None
        evict_set = set(to_evict)

        sleep_now_s = 0.0
        offload_now_s = 0.0

        for m in displaced_models:
            if m in evict_set:
                continue
            mi = self.models.get(m)
            if mi:
                sleep_now_s += float(mi.t_sleep_s)

        for m in evict_set:
            mi = self.models.get(m)
            if not mi:
                continue
            offload_now_s += float(mi.t_offload_s)
            future_penalty += float(need.get(m, 0)) * float(mi.t_load_s - mi.t_wake_s)

        total = drain_cost + activate_cost + sleep_now_s + offload_now_s + future_penalty

        return total, SwitchPlan(
            target_set=target_set,
            target_model=target_model,
            overlapped_active_ids=tuple(overlapped_ids),
            displaced_models=tuple(displaced_models),
            evict_models=tuple(sorted(evict_set)),
            activate_kind=activate_kind,
            est_drain_s=float(drain_cost),
            est_activate_s=float(activate_cost),
            est_sleep_now_s=float(sleep_now_s),
            est_offload_now_s=float(offload_now_s),
            est_future_penalty_s=float(future_penalty),
            total_cost_s=float(total),
        )

    def _find_slept_instance_locked(self, model: str, gpu_set: Tuple[int, ...]) -> Optional[str]:
        for iid in self.slept_ids:
            inst = self.instances.get(iid)
            if inst and inst.model == model and tuple(inst.gpus) == tuple(gpu_set) and inst.state == InstState.SLEPT:
                return iid
        return None

    def _choose_evictions_locked(
        self,
        target_set: Tuple[int, ...],
        target_model: str,
        need: Dict[str, int],
    ) -> Tuple[bool, List[str]]:
        candidates: Set[str] = set()
        for g in target_set:
            for m in self.gpus[g].resident.keys():
                if m != target_model:
                    candidates.add(m)

        evicted: List[str] = []

        def score(m: str) -> Tuple[int, float]:
            n = int(need.get(m, 0))
            # smaller last_used => older
            last = min(
                float(self.gpus[g].last_used.get(m, 0.0))
                for g in target_set
                if m in self.gpus[g].last_used
            ) if candidates else 0.0
            return (n, last)

        while not self._c1_satisfied_locked(target_set, target_model, evicted_extra=set(evicted)):
            if not candidates:
                return False, []
            m = sorted(candidates, key=score)[0]
            candidates.discard(m)
            evicted.append(m)

        return True, evicted

    def _c1_satisfied_locked(self, target_set: Tuple[int, ...], target_model: str, *, evicted_extra: Set[str]) -> bool:
        tp_target = len(target_set)
        for g in target_set:
            used = self._estimate_gpu_weight_used_MB_locked(g, evicted_extra=evicted_extra)
            if not self.gpus[g].is_resident(target_model):
                used += self._estimate_model_weight_MB_locked(target_model, tp_target, g)
            if used > self.gpus[g].weight_cap_MB + 1e-6:
                return False
        return True

    def _estimate_gpu_weight_used_MB_locked(self, gpu_id: int, *, evicted_extra: Set[str]) -> float:
        used = 0.0
        gi = self.gpus[gpu_id]
        for m in list(gi.resident.keys()):
            if m in evicted_extra:
                continue
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

    def _dispatch_locked(self, r: Request, iid: str, now: float) -> None:
        inst = self.instances.get(iid)
        if not inst or inst.state != InstState.ACTIVE or inst.model != r.model or not inst.accept_new:
            self._requeue_locked(r)
            return
        r.state = ReqState.RUNNING

        for g in inst.gpus:
            self.gpus[g].set_resident(r.model, Residence.ACTIVE, now, pid=inst.pid_by_gpu.get(g), tp=len(inst.gpus), url=inst.base_url)

        self._spawn(self._run_infer(r, iid))

    def _requeue_locked(self, r: Request) -> None:
        r.state = ReqState.WAITING
        self._push(self._waiting, r.t_arr, r.key())

    def _schedule_switch_locked(self, r: Request, plan: SwitchPlan, now: float) -> None:
        for iid in plan.overlapped_active_ids:
            inst = self.instances.get(iid)
            if inst and inst.state == InstState.ACTIVE:
                inst.accept_new = False
                inst.state = InstState.DRAINING

        r.state = ReqState.RUNNING
        self._spawn(self._switch_then_run(r, plan))

    # -----------------
    # Background tasks
    # -----------------

    async def _run_infer(self, r: Request, iid: str) -> None:
        async with self._lock:
            inst = self.instances.get(iid)
        if not inst:
            out = {"error": "instance not found", "instance_id": iid}
        else:
            try:
                out = await self.controller.infer(inst, r.payload)
            except Exception as e:
                out = {"error": str(e), "instance_id": iid, "model": r.model}

        async with self._lock:
            r.state = ReqState.DONE

        if r.on_completed:
            try:
                r.on_completed(r, out)
            except Exception:
                pass

    async def _switch_then_run(self, r: Request, plan: SwitchPlan) -> None:
        async with self._lock:
            overlapped = [self.instances[iid] for iid in plan.overlapped_active_ids if iid in self.instances]
            evict_set = set(plan.evict_models)

        # 1) Drain
        for inst in overlapped:
            try:
                await self.controller.drain_until_empty(inst)
            except Exception:
                pass

        # 2) Sleep or kill overlapped
        pid_mem = await self.controller.pid_used_MB()
        async with self._lock:
            if pid_mem:
                self._pid_mem = pid_mem

        for inst in overlapped:
            if inst.model in evict_set:
                t0 = time.monotonic()
                try:
                    elapsed = await self.controller.kill(inst)
                except Exception:
                    elapsed = time.monotonic() - t0
                async with self._lock:
                    mi = self.models.get(inst.model)
                    if mi:
                        mi.update_offload(elapsed, ema=self.timing_ema)
                    self._remove_instance_locked(inst)
                continue

            t0 = time.monotonic()
            try:
                elapsed = await self.controller.sleep(inst)
            except Exception:
                elapsed = time.monotonic() - t0

            pid_mem2 = await self.controller.pid_used_MB()
            async with self._lock:
                if pid_mem2:
                    self._pid_mem = pid_mem2

                mi = self.models.get(inst.model)
                if mi:
                    mi.update_sleep(elapsed, ema=self.timing_ema)
                    vals = []
                    for g in inst.gpus:
                        pid = inst.pid_by_gpu.get(g)
                        if pid is None:
                            continue
                        v = self._pid_mem.get((g, int(pid)))
                        if v is not None:
                            vals.append(int(v))
                    if vals:
                        mi.update_slept_mem(len(inst.gpus), float(sum(vals)) / float(len(vals)), ema=self.timing_ema)

                now = time.monotonic()
                inst.state = InstState.SLEPT
                inst.accept_new = False
                self.active_ids.discard(inst.instance_id)
                self.slept_ids.add(inst.instance_id)
                for g in inst.gpus:
                    self.gpus[g].set_resident(inst.model, Residence.SLEPT, now, pid=inst.pid_by_gpu.get(g), tp=len(inst.gpus), url=inst.base_url)

        # 3) Kill any additional evicted instances
        async with self._lock:
            extra_to_kill = [inst for inst in self.instances.values() if inst.model in evict_set and inst.state != InstState.DEAD]

        for inst in extra_to_kill:
            async with self._lock:
                if inst.instance_id not in self.instances:
                    continue
            t0 = time.monotonic()
            try:
                elapsed = await self.controller.kill(inst)
            except Exception:
                elapsed = time.monotonic() - t0
            async with self._lock:
                mi = self.models.get(inst.model)
                if mi:
                    mi.update_offload(elapsed, ema=self.timing_ema)
                self._remove_instance_locked(inst)

        # 4) Activate target
        target_inst: Optional[Instance] = None
        async with self._lock:
            slept_iid = self._find_slept_instance_locked(plan.target_model, plan.target_set)
            if slept_iid:
                target_inst = self.instances.get(slept_iid)

        if target_inst is not None:
            t0 = time.monotonic()
            try:
                elapsed = await self.controller.wake(target_inst)
            except Exception:
                elapsed = time.monotonic() - t0
            async with self._lock:
                mi = self.models.get(plan.target_model)
                if mi:
                    mi.update_wake(elapsed, ema=self.timing_ema)
                now = time.monotonic()
                target_inst.state = InstState.ACTIVE
                target_inst.accept_new = True
                self.slept_ids.discard(target_inst.instance_id)
                self.active_ids.add(target_inst.instance_id)
                for g in plan.target_set:
                    self.gpus[g].set_resident(plan.target_model, Residence.ACTIVE, now, pid=target_inst.pid_by_gpu.get(g), tp=len(plan.target_set), url=target_inst.base_url)
        else:
            tp = len(plan.target_set)
            gpu_util = ManagedVLLMController.compute_gpu_mem_util(plan.target_set) if isinstance(self.controller, ManagedVLLMController) else 0.9

            t0 = time.monotonic()
            new_inst = await self.controller.start(plan.target_model, plan.target_set, tp=tp, gpu_mem_util=gpu_util)
            elapsed = time.monotonic() - t0

            async with self._lock:
                mi = self.models.get(plan.target_model)
                if mi:
                    mi.update_load(elapsed, ema=self.timing_ema)

                self.instances[new_inst.instance_id] = new_inst
                self.active_ids.add(new_inst.instance_id)
                self.slept_ids.discard(new_inst.instance_id)
                now = time.monotonic()
                for g in new_inst.gpus:
                    self.gpus[g].set_resident(plan.target_model, Residence.ACTIVE, now, pid=new_inst.pid_by_gpu.get(g), tp=len(new_inst.gpus), url=new_inst.base_url)

            target_inst = new_inst

        # 5) Dispatch
        async with self._lock:
            if not target_inst:
                r.state = ReqState.DONE
                return
            iid = target_inst.instance_id
        await self._run_infer(r, iid)

    def _remove_instance_locked(self, inst: Instance) -> None:
        inst.state = InstState.DEAD
        self.active_ids.discard(inst.instance_id)
        self.slept_ids.discard(inst.instance_id)
        self.instances.pop(inst.instance_id, None)
        for g in inst.gpus:
            self.gpus[g].evict_model(inst.model)

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
