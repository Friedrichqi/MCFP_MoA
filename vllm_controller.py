"""
vllm_controller.py - vLLM server lifecycle management.

Provides an abstraction over vLLM server processes:
- Start: spawn vLLM serve process with appropriate configuration
- Sleep: POST /sleep?level=2 to offload weights to CPU
- Wake: POST /wake_up to reload weights to GPU
- Kill: SIGKILL + wait for cleanup
- Metrics: parse Prometheus /metrics endpoint for drain_latency
- Infer: forward requests to vLLM server
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import signal
import socket
import subprocess
import time
from typing import Any, Dict, List, Optional, Protocol, Sequence, Set, Tuple

try:
    import httpx
    HAS_HTTPX = True
except ImportError:
    HAS_HTTPX = False

from entities import Instance, InstState, DrainLatency


# -----------------------------
# Protocol / Interface
# -----------------------------

class VLLMController(Protocol):
    """Protocol for vLLM server control."""
    
    async def metrics(self, inst: Instance) -> DrainLatency:
        """Get metrics snapshot from instance."""
        ...
    
    async def pid_used_MB(self) -> Dict[Tuple[int, int], int]:
        """Get (gpu_id, pid) -> used_MB mapping from nvidia-smi."""
        ...
    
    async def start(
        self,
        model: str,
        gpu_set: Tuple[int, ...],
        tp: int,
        gpu_mem_util: float = 0.9,
    ) -> Instance:
        """Start a new vLLM server instance."""
        ...
    
    async def sleep(self, inst: Instance) -> float:
        """Sleep instance, returns elapsed time."""
        ...
    
    async def wake(self, inst: Instance) -> float:
        """Wake instance, returns elapsed time."""
        ...
    
    async def kill(self, inst: Instance) -> float:
        """Kill instance, returns elapsed time."""
        ...
    
    async def infer(self, inst: Instance, payload: Any) -> Any:
        """Send inference request to instance."""
        ...
    
    async def drain_until_empty(
        self,
        inst: Instance,
        poll_s: float = 0.2,
        timeout_s: float = 600.0,
    ) -> float:
        """Wait until instance has no pending requests."""
        ...


# -----------------------------
# Managed vLLM Controller
# -----------------------------

class ManagedVLLMController:
    """
    Real vLLM server management via subprocess + HTTP.
    
    Launch command:
        CUDA_VISIBLE_DEVICES=<gpu-set> VLLM_SERVER_DEV_MODE=1 \\
        vllm serve <model> --tensor-parallel-size <tp> --port <port> \\
            --enable-sleep-mode --gpu-memory-utilization <util>
    
    Sleep: POST {base_url}/sleep?level=2
    Wake: POST {base_url}/wake_up
    
    Metrics parsing:
        vllm:num_requests_running + vllm:num_requests_waiting
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
        self.port_base = port_base
        self.request_timeout_s = request_timeout_s
        self.startup_timeout_s = startup_timeout_s
        self.vllm_extra_args = list(vllm_extra_args or [])
        
        self._next_port = port_base
        self._procs: Dict[str, subprocess.Popen] = {}
    
    # ---------- HTTP Helpers ----------
    
    async def _http_get_text(self, url: str) -> str:
        """GET request returning text."""
        if HAS_HTTPX:
            async with httpx.AsyncClient(timeout=self.request_timeout_s) as client:
                r = await client.get(url)
                r.raise_for_status()
                return r.text
        else:
            # Fallback to curl
            return await asyncio.to_thread(
                lambda: subprocess.check_output(["curl", "-s", url], text=True)
            )
    
    async def _http_post_json(self, url: str, json_body: Dict[str, Any]) -> Any:
        """POST request with JSON body."""
        if HAS_HTTPX:
            async with httpx.AsyncClient(timeout=self.request_timeout_s) as client:
                r = await client.post(url, json=json_body)
                r.raise_for_status()
                return r.json()
        else:
            payload = json.dumps(json_body)
            out = await asyncio.to_thread(
                lambda: subprocess.check_output(
                    ["curl", "-s", "-X", "POST", url, 
                     "-H", "Content-Type: application/json", "-d", payload],
                    text=True,
                )
            )
            try:
                return json.loads(out)
            except Exception:
                return {"raw": out}
    
    async def _http_post_no_body(self, url: str) -> str:
        """POST request without body."""
        if HAS_HTTPX:
            async with httpx.AsyncClient(timeout=self.request_timeout_s) as client:
                r = await client.post(url)
                r.raise_for_status()
                return r.text
        else:
            return await asyncio.to_thread(
                lambda: subprocess.check_output(["curl", "-s", "-X", "POST", url], text=True)
            )
    
    # ---------- Metrics Parsing ----------
    
    @staticmethod
    def _parse_prom_line(line: str) -> Optional[Tuple[str, Dict[str, str], float]]:
        """Parse a Prometheus metrics line."""
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
    
    async def metrics(self, inst: Instance) -> DrainLatency:
        """Get metrics from instance."""
        try:
            raw = await self._http_get_text(inst.metrics_url)
        except Exception:
            return DrainLatency()
        
        targets = {
            "vllm:num_requests_running",
            "vllm:num_requests_waiting",
            "vllm:e2e_request_latency_seconds_sum",
            "vllm:e2e_request_latency_seconds_count",
            "vllm:e2e_request_latency_seconds_bucket",
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
            # Try to match by model name
            for labels, v in arr:
                if labels.get("model_name") == inst.model_id:
                    return float(v)
            # Try to match by PID
            pids = {str(pid) for pid in inst.pid_by_gpu.values() if pid}
            for labels, v in arr:
                if labels.get("pid") in pids:
                    return float(v)
            return float(arr[0][1])
        
        def pick_buckets() -> List[Tuple[float, float]]:
            """Extract histogram buckets as (le_bound, cumulative_count) pairs."""
            arr = samples.get("vllm:e2e_request_latency_seconds_bucket", [])
            if not arr:
                return []
            
            # Filter by model_name if possible
            model_buckets = [(labels, v) for labels, v in arr 
                             if labels.get("model_name") == inst.model_id]
            if not model_buckets:
                model_buckets = arr
            
            # Extract (le, count) pairs and sort by le
            buckets: List[Tuple[float, float]] = []
            for labels, count in model_buckets:
                le_str = labels.get("le", "")
                if le_str == "+Inf":
                    le_val = float('inf')
                else:
                    try:
                        le_val = float(le_str)
                    except ValueError:
                        continue
                buckets.append((le_val, float(count)))
            
            # Sort by le bound
            buckets.sort(key=lambda x: (x[0] == float('inf'), x[0]))
            return buckets
        
        num_requests = pick("vllm:num_requests_running") + pick("vllm:num_requests_waiting")
        return DrainLatency(
            num_requests=num_requests,
            latency_sum=pick("vllm:e2e_request_latency_seconds_sum"),
            latency_count=pick("vllm:e2e_request_latency_seconds_count"),
            latency_buckets=pick_buckets(),
        )
    
    # ---------- nvidia-smi Helpers ----------
    
    async def pid_used_MB(self) -> Dict[Tuple[int, int], int]:
        """Get memory usage by (gpu_id, pid) from nvidia-smi."""
        return await asyncio.to_thread(self._pid_used_MB_sync)
    
    @staticmethod
    def _pid_used_MB_sync() -> Dict[Tuple[int, int], int]:
        """Synchronous version of pid_used_MB."""
        # Get GPU UUID to index mapping
        uuid_to_idx: Dict[str, int] = {}
        try:
            out = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"],
                text=True,
            ).strip()
            for ln in out.splitlines():
                parts = [x.strip() for x in ln.split(",")]
                if len(parts) == 2:
                    uuid_to_idx[parts[1]] = int(parts[0])
        except Exception:
            pass
        
        # Get per-process memory usage
        try:
            out = subprocess.check_output(
                ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_memory",
                 "--format=csv,noheader,nounits"],
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
                    gpu_id = uuid_to_idx[gpu_uuid]
                    pid = int(pid_s)
                    used = int(float(used_s))
                    res[(gpu_id, pid)] = used
                except Exception:
                    continue
            return res
        except Exception:
            return {}
    
    @staticmethod
    def _gpu_used_total_MB(gpu_ids: Sequence[int]) -> Dict[int, Tuple[int, int]]:
        """Get (used_MB, total_MB) per GPU."""
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,memory.used,memory.total",
             "--format=csv,noheader,nounits"],
            text=True,
        ).strip()
        
        m: Dict[int, Tuple[int, int]] = {}
        for ln in out.splitlines():
            parts = [x.strip() for x in ln.split(",")]
            if len(parts) == 3:
                idx = int(parts[0])
                if idx in gpu_ids:
                    m[idx] = (int(float(parts[1])), int(float(parts[2])))
        return m
    
    @classmethod
    def compute_gpu_mem_util(cls, gpu_set: Tuple[int, ...]) -> float:
        """Compute safe gpu_memory_utilization for a GPU set."""
        try:
            stats = cls._gpu_used_total_MB(list(gpu_set))
            if not stats:
                return 0.9
            
            ratios = []
            for _, (used, total) in stats.items():
                free = max(0, total - used)
                ratios.append(free / total if total > 0 else 0.0)
            
            free_ratio = min(ratios) if ratios else 0.5
            util = max(0.05, min(0.95, free_ratio * 0.9))
            return util
        except Exception:
            return 0.9
    
    @staticmethod
    def _snapshot_pids_by_gpu(gpu_set: Tuple[int, ...]) -> Dict[int, Set[int]]:
        """Snapshot current PIDs on each GPU."""
        try:
            out = subprocess.check_output(
                ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid",
                 "--format=csv,noheader,nounits"],
                text=True,
            ).strip()
        except Exception:
            return {g: set() for g in gpu_set}
        
        # Get UUID mapping
        uuid_to_idx: Dict[str, int] = {}
        try:
            out2 = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"],
                text=True,
            ).strip()
            for ln in out2.splitlines():
                parts = [x.strip() for x in ln.split(",")]
                if len(parts) == 2:
                    uuid_to_idx[parts[1]] = int(parts[0])
        except Exception:
            pass
        
        m: Dict[int, Set[int]] = {g: set() for g in gpu_set}
        for ln in out.splitlines():
            parts = [x.strip() for x in ln.split(",")]
            if len(parts) != 2:
                continue
            gpu_uuid, pid_s = parts
            if gpu_uuid not in uuid_to_idx:
                continue
            gid = uuid_to_idx[gpu_uuid]
            if gid in m:
                try:
                    m[gid].add(int(pid_s))
                except Exception:
                    pass
        return m
    
    # ---------- Port Management ----------
    
    def _alloc_port(self) -> int:
        """Allocate next available port."""
        while True:
            p = self._next_port
            self._next_port += 1
            if self._port_free(p):
                return p
    
    @staticmethod
    def _port_free(port: int) -> bool:
        """Check if port is available."""
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
    
    # ---------- Lifecycle Methods ----------
    
    async def start(
        self,
        model: str,
        gpu_set: Tuple[int, ...],
        tp: int,
        gpu_mem_util: float = 0.9,
    ) -> Instance:
        """Start a new vLLM server instance."""
        port = self._alloc_port()
        base_url = f"http://{self.host}:{port}"
        metrics_url = f"{base_url}/metrics"
        instance_id = f"{','.join(map(str, gpu_set))}|{model}|{port}"
        
        # Snapshot PIDs before starting
        before = await asyncio.to_thread(self._snapshot_pids_by_gpu, gpu_set)
        
        # Prepare environment
        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = ",".join(str(x) for x in gpu_set)
        env["VLLM_SERVER_DEV_MODE"] = "1"
        
        # Build command
        cmd = [
            "vllm", "serve", model,
            "--host", self.host,
            "--port", str(port),
            "--tensor-parallel-size", str(tp),
            "--enable-sleep-mode",
            "--max-model-len", "16384",
            "--gpu-memory-utilization", str(gpu_mem_util),
        ] + self.vllm_extra_args
        
        # Start process
        proc = subprocess.Popen(
            cmd,
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        self._procs[instance_id] = proc
        
        # Wait for server to be ready
        t0 = time.monotonic()
        while True:
            if proc.poll() is not None:
                raise RuntimeError(
                    f"vllm serve exited early for {instance_id} (rc={proc.returncode})"
                )
            try:
                txt = await self._http_get_text(metrics_url)
                if len(txt) > 0:
                    break
            except Exception:
                pass
            if time.monotonic() - t0 > self.startup_timeout_s:
                raise TimeoutError(f"Timeout waiting for vllm server {instance_id}")
            await asyncio.sleep(0.5)
        
        # Detect PIDs on each GPU
        after = await asyncio.to_thread(self._snapshot_pids_by_gpu, gpu_set)
        pid_by_gpu: Dict[int, int] = {}
        for g in gpu_set:
            new_pids = list(after.get(g, set()) - before.get(g, set()))
            pid_by_gpu[g] = new_pids[0] if new_pids else proc.pid
        
        return Instance(
            instance_id=instance_id,
            model_id=model,
            gpus=tuple(gpu_set),
            base_url=base_url,
            metrics_url=metrics_url,
            state=InstState.ACTIVE,
            accept_new=True,
            pid_by_gpu=pid_by_gpu,
        )
    
    async def sleep(self, inst: Instance) -> float:
        """Put instance to sleep (offload weights to CPU)."""
        t0 = time.monotonic()
        await self._http_post_no_body(f"{inst.base_url}/sleep?level=2")
        return time.monotonic() - t0
    
    async def wake(self, inst: Instance) -> float:
        """Wake instance (reload weights to GPU)."""
        t0 = time.monotonic()
        await self._http_post_no_body(f"{inst.base_url}/wake_up")
        return time.monotonic() - t0
    
    async def kill(self, inst: Instance) -> float:
        """Kill instance and wait for GPU cleanup."""
        t0 = time.monotonic()
        
        pids = {int(p) for p in inst.pid_by_gpu.values() if p}
        
        # Kill known PIDs
        for pid in list(pids):
            try:
                os.kill(pid, signal.SIGKILL)
            except Exception:
                pass
        
        # Kill process group if we have the Popen
        proc = self._procs.pop(inst.instance_id, None)
        if proc and proc.poll() is None:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass
        
        # Wait for processes to disappear from nvidia-smi
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
        """Send inference request to instance."""
        if not isinstance(payload, dict):
            payload = {"messages": [{"role": "user", "content": str(payload)}]}
        
        # Determine endpoint
        endpoint = payload.pop("endpoint", None)
        if endpoint is None:
            if "messages" in payload:
                endpoint = "/v1/chat/completions"
            elif "prompt" in payload:
                endpoint = "/v1/completions"
            else:
                endpoint = "/v1/chat/completions"
                payload = {"messages": [{"role": "user", "content": json.dumps(payload)}]}
        
        payload.setdefault("model", inst.model_id)
        url = inst.base_url.rstrip("/") + endpoint
        
        return await self._http_post_json(url, payload)
    
    async def drain_until_empty(
        self,
        inst: Instance,
        poll_s: float = 0.2,
        timeout_s: float = 600.0,
    ) -> float:
        """Wait until instance has no pending requests."""
        inst.accept_new = False
        t0 = time.monotonic()
        while True:
            snap = await self.metrics(inst)
            if snap.num_requests <= 0:
                return time.monotonic() - t0
            if time.monotonic() - t0 > timeout_s:
                return time.monotonic() - t0
            await asyncio.sleep(poll_s)
