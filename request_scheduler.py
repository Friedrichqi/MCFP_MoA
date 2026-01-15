"""
request_scheduler.py - Request Routing Scheduler.

Runs every δt1 (default 1s) to process DAG-based jobs.

Per logic_devemopment_revised.md:
1. Activate ready nodes: move requests with indegree==0 from potential -> waiting
2. Route to active instances: send to instance with minimal drain_latency
3. Mark unmet demand: flag models that need loading via GPU scheduler
"""

from __future__ import annotations

import asyncio
import heapq
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

from entities import (
    GPUInfo, ModelCard, Instance, InstState,
    Request, ReqState, DrainLatency
)
from vllm_controller import VLLMController, ManagedVLLMController


logger = logging.getLogger(__name__)


@dataclass
class SchedulerStats:
    """Statistics tracking for the request scheduler."""
    
    total_requests_added: int = 0
    total_requests_completed: int = 0
    total_requests_dispatched: int = 0
    total_activated: int = 0
    
    # Timing
    avg_wait_time_s: float = 0.0
    _wait_time_sum: float = 0.0
    _wait_time_count: int = 0


class RequestScheduler:
    """
    Request Scheduler for DAG-based job execution.
    
    Manages two queues:
    - potential: requests awaiting DAG prerequisites (indegree > 0)
    - waiting: requests ready for execution (indegree == 0)
    
    Each step() call:
    1. Moves ready requests from potential -> waiting
    2. Routes waiting requests to active instances
    3. Returns set of models with unmet demand
    """
    
    def __init__(
        self,
        controller: Union[VLLMController, ManagedVLLMController],
        instances: Dict[str, Instance],
        active_ids: Set[str],
        model_cards: Dict[str, ModelCard],
        *,
        beta: float = 1.0,
        max_decisions_per_step: int = 1024,
    ) -> None:
        self.controller = controller
        self.instances = instances
        self.active_ids = active_ids
        self.model_cards = model_cards
        self.beta = beta
        self.max_decisions_per_step = max_decisions_per_step
        
        # Request storage
        self._requests: Dict[str, Request] = {}  # key -> Request
        
        # Priority queues: (t_arr, sequence, key)
        self._potential: List[Tuple[float, int, str]] = []
        self._waiting: List[Tuple[float, int, str]] = []
        self._seq = 0
        
        # Background tasks
        self._bg_tasks: Set[asyncio.Task] = set()
        
        # Stats
        self._stats = SchedulerStats()
        
        # Lock for concurrent access
        self._lock = asyncio.Lock()
    
    # -----------------
    # Queue Operations
    # -----------------
    
    async def add_request(self, req: Request) -> bool:
        """
        Add a new request to the scheduler.
        
        Returns True if inserted new, False if merged with existing.
        """
        async with self._lock:
            return self._add_request_locked(req)
    
    def _add_request_locked(self, req: Request) -> bool:
        """Add request while holding lock."""
        key = req.key
        
        if key in self._requests:
            # Merge with existing request
            existing = self._requests[key]
            existing.indegree = min(existing.indegree, req.indegree)
            existing.t_arr = min(existing.t_arr, req.t_arr)
            existing.succ = list(set(existing.succ + req.succ))
            return False
        
        # New request
        req.state = ReqState.POTENTIAL
        self._requests[key] = req
        self._push_heap(self._potential, req.t_arr, key)
        self._stats.total_requests_added += 1
        
        return True
    
    async def update_indegree(
        self,
        job_id: str,
        node_id: str,
        new_indegree: int,
    ) -> None:
        """Update indegree for a request (when predecessor completes)."""
        key = f"{job_id}:{node_id}"
        async with self._lock:
            if key in self._requests:
                self._requests[key].indegree = new_indegree
    
    async def get_potential_requests(self) -> List[Request]:
        """Get all requests in potential queue."""
        async with self._lock:
            return [
                self._requests[k]
                for _, _, k in self._potential
                if k in self._requests and self._requests[k].state == ReqState.POTENTIAL
            ]
    
    async def get_waiting_requests(self) -> List[Request]:
        """Get all requests in waiting queue."""
        async with self._lock:
            return [
                self._requests[k]
                for _, _, k in self._waiting
                if k in self._requests and self._requests[k].state == ReqState.WAITING
            ]
    
    # -----------------
    # Main Scheduling Loop
    # -----------------
    
    async def step(self) -> Set[str]:
        """
        Main scheduling step (called every δt1).
        
        1. Move ready requests from potential -> waiting
        2. Route waiting requests to active instances
        3. Return set of models with unmet demand
        
        Returns:
            Set of model_ids that have waiting requests but no active instance
        """
        now = time.monotonic()
        
        async with self._lock:
            # 1. Activate ready nodes
            activated = self._activate_ready_nodes_locked()
            self._stats.total_activated += activated
            
            # 2. Collect batch of waiting requests
            batch: List[str] = []
            while self._waiting and len(batch) < self.max_decisions_per_step:
                _, _, key = heapq.heappop(self._waiting)
                req = self._requests.get(key)
                if req and req.state == ReqState.WAITING:
                    batch.append(key)
        
        # 3. Fetch metrics from active instances
        metrics = await self._fetch_metrics()
        
        # 4. Process batch
        unmet_models: Set[str] = set()
        
        for key in batch:
            async with self._lock:
                req = self._requests.get(key)
                if not req or req.state != ReqState.WAITING:
                    continue
                
                # Try to route to active instance
                best_inst = self._find_best_instance_locked(req.model, metrics)
                
                if best_inst:
                    # Dispatch to instance
                    self._dispatch_request_locked(req, best_inst, now)
                else:
                    # No active instance - mark as unmet demand
                    unmet_models.add(req.model)
                    # Re-queue the request
                    self._push_heap(self._waiting, req.t_arr, key)
        
        return unmet_models
    
    def _activate_ready_nodes_locked(self) -> int:
        """
        Move requests with indegree==0 from potential to waiting.
        
        Returns number of requests activated.
        """
        now = time.monotonic()
        activated = 0
        new_potential: List[Tuple[float, int, str]] = []
        
        while self._potential:
            t, seq, key = heapq.heappop(self._potential)
            req = self._requests.get(key)
            
            if not req or req.state != ReqState.POTENTIAL:
                continue
            
            if req.indegree == 0:
                # Ready to execute
                req.state = ReqState.WAITING
                req.t_arr = now  # Reset arrival time to when it became ready
                self._push_heap(self._waiting, req.t_arr, key)
                activated += 1
            else:
                # Still waiting on dependencies
                new_potential.append((t, seq, key))
        
        self._potential = new_potential
        heapq.heapify(self._potential)
        
        return activated
    
    async def _fetch_metrics(self) -> Dict[str, DrainLatency]:
        """Fetch metrics from all active instances."""
        active_insts = [
            self.instances[iid]
            for iid in self.active_ids
            if iid in self.instances and self.instances[iid].state == InstState.ACTIVE
        ]
        
        if not active_insts:
            return {}
        
        results = await asyncio.gather(
            *[self.controller.metrics(inst) for inst in active_insts],
            return_exceptions=True
        )
        
        metrics: Dict[str, DrainLatency] = {}
        for inst, result in zip(active_insts, results):
            if isinstance(result, Exception):
                logger.warning(f"Failed to get metrics for {inst.instance_id}: {result}")
                metrics[inst.instance_id] = DrainLatency()
            else:
                metrics[inst.instance_id] = result
        
        return metrics
    
    def _find_best_instance_locked(
        self,
        model: str,
        metrics: Dict[str, DrainLatency],
    ) -> Optional[Instance]:
        """
        Find the best active instance to handle a request.
        
        Selection criteria: minimal drain_latency among active instances
        serving the requested model.
        """
        best_inst: Optional[Instance] = None
        best_cost = float('inf')
        
        for iid in self.active_ids:
            inst = self.instances.get(iid)
            if not inst:
                continue
            if inst.model_id != model:
                continue
            if inst.state != InstState.ACTIVE:
                continue
            if not inst.accept_new:
                continue
            
            # Get drain latency cost
            cost = metrics.get(iid, DrainLatency()).drain_latency
            
            if cost < best_cost:
                best_cost = cost
                best_inst = inst
        
        return best_inst
    
    def _dispatch_request_locked(
        self,
        req: Request,
        inst: Instance,
        now: float,
    ) -> None:
        """Dispatch a request to an instance."""
        req.state = ReqState.RUNNING
        
        # Track wait time
        wait_time = now - req.t_arr
        self._stats._wait_time_sum += wait_time
        self._stats._wait_time_count += 1
        self._stats.avg_wait_time_s = (
            self._stats._wait_time_sum / self._stats._wait_time_count
        )
        
        # Update instance last_used
        inst.last_used = now
        
        # Call on_dispatched callback and add successor requests
        if req.on_dispatched:
            try:
                new_requests = req.on_dispatched(req)
                if new_requests:
                    for new_req in new_requests:
                        self._add_request_locked(new_req)
            except Exception as e:
                logger.error(f"on_dispatched callback failed for {req.key}: {e}")
        
        # Spawn background task for inference
        self._spawn_task(self._run_inference(req, inst))
        self._stats.total_requests_dispatched += 1
    
    # -----------------
    # Inference Execution
    # -----------------
    
    async def _run_inference(self, req: Request, inst: Instance) -> None:
        """Execute inference for a request."""
        # Sanity check: instance may have been slept between dispatch and execution
        if inst.state != InstState.ACTIVE:
            logger.warning(
                f"Instance {inst.instance_id} is no longer active (state={inst.state}), "
                f"re-queueing request {req.key}"
            )
            async with self._lock:
                req.state = ReqState.WAITING
                self._push_heap(self._waiting, req.t_arr, req.key)
            return
        
        try:
            result = await self.controller.infer(inst, req.payload)
        except Exception as e:
            logger.error(f"Inference failed for {req.key}: {e}")
            result = {"error": str(e), "instance_id": inst.instance_id}
        
        # Mark as done and decrement successor indegrees
        async with self._lock:
            req.state = ReqState.DONE
            self._stats.total_requests_completed += 1
            
            # Decrement indegree of all successors
            for succ_node in req.succ:
                succ_key = f"{req.job_id}:{succ_node}"
                if succ_key in self._requests:
                    self._requests[succ_key].indegree = max(
                        0, self._requests[succ_key].indegree - 1
                    )
        
        # Call completion callback
        if req.on_completed:
            try:
                req.on_completed(req, result)
            except Exception as e:
                logger.error(f"Completion callback failed for {req.key}: {e}")
    
    # -----------------
    # Utilities
    # -----------------
    
    def _push_heap(
        self,
        heap: List[Tuple[float, int, str]],
        t: float,
        key: str,
    ) -> None:
        """Push item to heap with sequence number for stability."""
        self._seq += 1
        heapq.heappush(heap, (t, self._seq, key))
    
    def _spawn_task(self, coro: Any) -> None:
        """Spawn a background task."""
        task = asyncio.create_task(coro)
        self._bg_tasks.add(task)
        task.add_done_callback(lambda t: self._bg_tasks.discard(t))
    
    # -----------------
    # Status
    # -----------------
    
    def get_stats(self) -> Dict[str, Any]:
        """Get scheduler statistics."""
        return {
            "total_requests_added": self._stats.total_requests_added,
            "total_requests_dispatched": self._stats.total_requests_dispatched,
            "total_requests_completed": self._stats.total_requests_completed,
            "total_activated": self._stats.total_activated,
            "avg_wait_time_s": self._stats.avg_wait_time_s,
            "potential_queue_size": len(self._potential),
            "waiting_queue_size": len(self._waiting),
            "pending_tasks": len(self._bg_tasks),
        }
    
    async def get_queue_info(self) -> Dict[str, Any]:
        """Get detailed queue information."""
        async with self._lock:
            potential = []
            for _, _, key in self._potential:
                req = self._requests.get(key)
                if req and req.state == ReqState.POTENTIAL:
                    potential.append({
                        "key": key,
                        "model": req.model,
                        "indegree": req.indegree,
                        "wait_time": time.monotonic() - req.t_arr,
                    })
            
            waiting = []
            for _, _, key in self._waiting:
                req = self._requests.get(key)
                if req and req.state == ReqState.WAITING:
                    waiting.append({
                        "key": key,
                        "model": req.model,
                        "wait_time": time.monotonic() - req.t_arr,
                    })
            
            return {
                "potential": potential,
                "waiting": waiting,
            }


# -----------------------------
# Job Graph
# -----------------------------

@dataclass
class JobGraph:
    """
    Represents a DAG job with multiple nodes (stages).
    
    Each node is an LLM invocation. Edges represent data dependencies:
    - Edge u -> v means outputs of u concatenate into inputs of v
    """
    
    job_id: str
    graph: Dict[str, List[str]] = field(default_factory=dict)  # node -> successors
    node_models: Dict[str, str] = field(default_factory=dict)  # node -> model_id
    inputs: Dict[str, Any] = field(default_factory=dict)       # node -> base input
    
    # Computed state
    nodes: Set[str] = field(default_factory=set)
    indegree: Dict[str, int] = field(default_factory=dict)
    parents: Dict[str, List[str]] = field(default_factory=dict)
    
    # Execution state
    discovered: Set[str] = field(default_factory=set)
    completed: Set[str] = field(default_factory=set)
    outputs: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self) -> None:
        """Initialize computed state from graph definition."""
        # Collect all nodes
        all_nodes: Set[str] = set(self.graph.keys())
        for u, vs in self.graph.items():
            for v in vs:
                all_nodes.add(v)
        
        # Ensure all nodes have entries
        for n in all_nodes:
            self.graph.setdefault(n, [])
        self.nodes = all_nodes
        
        # Default node_models: node_id is model_id
        if not self.node_models:
            self.node_models = {n: n for n in self.nodes}
        else:
            for n in self.nodes:
                if n not in self.node_models:
                    raise ValueError(f"node_models missing mapping for node '{n}'")
        
        # Compute indegree and parents
        self.indegree = {n: 0 for n in self.nodes}
        self.parents = {n: [] for n in self.nodes}
        for u, vs in self.graph.items():
            for v in vs:
                self.indegree[v] += 1
                self.parents[v].append(u)
        
        # Validate DAG
        self._validate_dag()
    
    def _validate_dag(self) -> None:
        """Verify the graph is acyclic (topological sort)."""
        indeg = dict(self.indegree)
        queue = [n for n, d in indeg.items() if d == 0]
        seen = 0
        
        while queue:
            n = queue.pop()
            seen += 1
            for v in self.graph.get(n, []):
                indeg[v] -= 1
                if indeg[v] == 0:
                    queue.append(v)
        
        if seen != len(self.nodes):
            raise ValueError("Graph contains a cycle (not a valid DAG)")
    
    @property
    def done(self) -> bool:
        """Check if all nodes are completed."""
        return len(self.completed) == len(self.nodes)
    
    def model_of(self, node: str) -> str:
        """Get model_id for a node."""
        return self.node_models[node]
    
    def get_sources(self) -> List[str]:
        """Get source nodes (indegree == 0)."""
        return [n for n, d in self.indegree.items() if d == 0]
    
    def build_payload(self, node: str) -> Any:
        """
        Build request payload for a node.
        
        Combines base input with outputs from parent nodes.
        """
        import json
        
        base = self.inputs.get(node, {})
        parent_outputs = {p: self.outputs[p] for p in self.parents[node] if p in self.outputs}
        
        if isinstance(base, dict) and ("prompt" in base or "messages" in base):
            base = dict(base)
            base.setdefault("metadata", {})
            if isinstance(base.get("metadata"), dict):
                base["metadata"]["parents"] = parent_outputs
                base["metadata"]["node_id"] = node
            return base
        
        return {
            "messages": [{
                "role": "user",
                "content": json.dumps({
                    "node_id": node,
                    "base": base,
                    "parents": parent_outputs,
                }),
            }]
        }
    
    def make_request(self, node: str) -> Request:
        """Create a Request for a node."""
        return Request(
            job_id=self.job_id,
            node_id=node,
            model=self.model_of(node),
            t_arr=time.monotonic(),
            indegree=self.indegree[node],
            succ=list(self.graph.get(node, [])),
        )
