"""
gpu_scheduler.py - GPU Reconfiguration Scheduler.

Runs every δt2 (default 5s) to optimize GPU-to-model assignments
using the Min-Cost Flow algorithm.

Per logic_devemopment_revised.md:
1. Build 4-layer flow graph (Source -> GPUs -> Model Copies -> Sink)
2. Populate edge costs (drain, sleep/offload, wake/load, waiting relief)
3. Solve MCF to maximize flow with minimal cost
4. Generate and apply ReconfigPlan (load/activate/sleep/offload)
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from entities import (
    GPUInfo, GPUState, ModelCard, Instance, InstState,
    Residence, Request, ReqState, DrainLatency, ReconfigPlan
)
from mcf_solver import (
    MinCostFlowSolver, MCFSolution, MCFAssignment,
    compute_waiting_times, compute_request_counts, compute_drain_latencies
)
from vllm_controller import VLLMController, ManagedVLLMController


logger = logging.getLogger(__name__)


@dataclass
class GPUSchedulerState:
    """Internal state tracking for the GPU scheduler."""
    
    # Currently loading models (excluded from new decisions to avoid thrash)
    loading_models: Set[str] = field(default_factory=set)
    
    # Last reconfiguration time
    last_reconfig_time: float = 0.0
    
    # Accumulated metrics
    total_reconfigs: int = 0
    total_loads: int = 0
    total_wakes: int = 0
    total_sleeps: int = 0
    total_offloads: int = 0


class GPUScheduler:
    """
    GPU Reconfiguration Scheduler.
    
    Responsible for deciding which models to load/wake/sleep/offload
    based on current demand and GPU state.
    
    Uses Min-Cost Flow optimization on a 4-layer graph:
    - L1 (Source): Starting point of flow
    - L2 (GPUs): STABLE GPUs with capacity=1, cost=drain_latency
    - L3 (Model Copies): One copy per (GPU, needed_model) pair
    - L4 (Sink): End point with unlimited capacity
    """
    
    def __init__(
        self,
        gpus: Dict[int, GPUInfo],
        gpu_sets: List[Tuple[int, ...]],
        model_cards: Dict[str, ModelCard],
        instances: Dict[str, Instance],
        active_ids: Set[str],
        slept_ids: Set[str],
        controller: Union[VLLMController, ManagedVLLMController],
        *,
        timing_ema_alpha: float = 0.2,
    ) -> None:
        self.gpus = gpus
        self.gpu_sets = gpu_sets
        self.model_cards = model_cards
        self.instances = instances
        self.active_ids = active_ids
        self.slept_ids = slept_ids
        self.controller = controller
        self.timing_ema_alpha = timing_ema_alpha
        
        self._state = GPUSchedulerState()
        self._lock = asyncio.Lock()
        
        # Solver instance (reused)
        self._solver = MinCostFlowSolver()
    
    # -----------------
    # Main Entry Point
    # -----------------
    
    async def reconfigure(
        self,
        potential_requests: List[Request],
        waiting_requests: List[Request],
    ) -> ReconfigPlan:
        """
        Main entry point for GPU reconfiguration.
        
        1. Collect needed models (from waiting + potential, excluding loading)
        2. Fetch metrics from all active instances
        3. Build and solve MCF graph
        4. Generate and apply reconfiguration plan
        
        Args:
            potential_requests: Requests awaiting DAG dependencies
            waiting_requests: Requests ready for execution
            
        Returns:
            ReconfigPlan with actions taken
        """
        now = time.monotonic()
        
        async with self._lock:
            # 1. Compute needed models
            needed_models = self._compute_needed_models(
                potential_requests, waiting_requests
            )
            
            if not needed_models:
                logger.debug("No models needed, skipping reconfiguration")
                return ReconfigPlan()
            
            # 2. Fetch metrics from active instances
            metrics = await self._fetch_all_metrics()
            
            # 3. Compute derived values
            all_requests = potential_requests + waiting_requests
            waiting_times = compute_waiting_times(all_requests, now)
            request_counts = compute_request_counts(all_requests)
            drain_latencies = compute_drain_latencies(
                self.gpus, self.instances, metrics
            )
            
            # 4. Build and solve MCF graph
            self._solver.build_graph(
                gpus=self.gpus,
                needed_models=needed_models,
                model_cards=self.model_cards,
                instances=self.instances,
                drain_latencies=drain_latencies,
                waiting_times=waiting_times,
                request_counts=request_counts,
            )
            
            solution = self._solver.solve()
            
            # 5. Generate plan from solution
            plan = self._generate_plan(solution, needed_models)
            
            # 6. Apply plan
            if not plan.is_empty:
                await self._apply_plan(plan)
                self._state.last_reconfig_time = now
                self._state.total_reconfigs += 1
            
            return plan
    
    # -----------------
    # Helpers
    # -----------------
    
    def _compute_needed_models(
        self,
        potential: List[Request],
        waiting: List[Request],
    ) -> Set[str]:
        """
        Compute set of needed models.
        
        Needed = models required by (waiting + potential) requests,
        excluding models already being loaded.
        """
        needed: Set[str] = set()
        
        for req in potential + waiting:
            if req.state in (ReqState.POTENTIAL, ReqState.WAITING):
                if req.model not in self._state.loading_models:
                    needed.add(req.model)
        
        return needed
    
    async def _fetch_all_metrics(self) -> Dict[str, DrainLatency]:
        """Fetch metrics from all active instances."""
        active_insts = [
            self.instances[iid]
            for iid in self.active_ids
            if iid in self.instances
        ]
        
        if not active_insts:
            return {}
        
        # Get fallback latencies from ModelCards
        fallback_latencies = {
            inst.instance_id: self.model_cards.get(inst.model_id, ModelCard(model_id=inst.model_id)).avg_latency_s
            for inst in active_insts
        }
        
        results = await asyncio.gather(
            *[
                self.controller.metrics(inst, fallback_latencies.get(inst.instance_id, 60.0))
                for inst in active_insts
            ],
            return_exceptions=True
        )
        
        metrics: Dict[str, DrainLatency] = {}
        for inst, result in zip(active_insts, results):
            if isinstance(result, Exception):
                logger.warning(f"Failed to get metrics for {inst.instance_id}: {result}")
                metrics[inst.instance_id] = DrainLatency(
                    fallback_latency=fallback_latencies.get(inst.instance_id, 60.0)
                )
            else:
                metrics[inst.instance_id] = result
                # Update ModelCard EMA with observed avg_latency
                if result.latency_count > 0:
                    card = self.model_cards.get(inst.model_id)
                    if card:
                        observed_avg = result.latency_sum / result.latency_count
                        card.update_avg_latency(observed_avg, alpha=self.timing_ema_alpha)
        
        return metrics
    
    def _generate_plan(
        self,
        solution: MCFSolution,
        needed_models: Set[str],
    ) -> ReconfigPlan:
        """Generate reconfiguration plan from MCF solution."""
        plan = ReconfigPlan(estimated_cost_s=solution.total_cost)
        
        # Track which models are getting assigned
        assigned_models: Set[str] = set()
        used_gpus: Set[int] = set()
        
        for assignment in solution.assignments:
            gpu_id = assignment.gpu_id
            model_id = assignment.model_id
            
            if model_id in assigned_models:
                continue  # Already assigned to another GPU
            
            if gpu_id in used_gpus:
                continue  # GPU already used
            
            assigned_models.add(model_id)
            used_gpus.add(gpu_id)
            
            # Determine action based on current state
            gpu = self.gpus.get(gpu_id)
            if not gpu:
                continue
            
            # Check for existing instance of this model
            slept_inst = self._find_slept_instance(model_id, (gpu_id,))
            active_inst = self._find_active_instance(model_id, (gpu_id,))
            
            if active_inst:
                # Model already active on this GPU, no action needed
                continue
            
            # Need to displace current active model(s) on this GPU
            for inst_id in list(self.active_ids):
                inst = self.instances.get(inst_id)
                if inst and gpu_id in inst.gpus and inst.model_id != model_id:
                    # Check if we should sleep or offload
                    if self._should_offload(gpu, inst.model_id, model_id):
                        plan.to_offload.append(inst_id)
                    else:
                        plan.to_sleep.append(inst_id)
            
            if slept_inst:
                # Wake existing slept instance
                plan.to_wake.append(slept_inst.instance_id)
            else:
                # Need to load new instance
                gpu_set = self._find_gpu_set_for_model(model_id, gpu_id)
                if gpu_set:
                    plan.to_load.append((gpu_set, model_id))
        
        return plan
    
    def _find_slept_instance(
        self,
        model_id: str,
        gpu_set: Tuple[int, ...],
    ) -> Optional[Instance]:
        """Find a slept instance of model on given GPU set."""
        for inst_id in self.slept_ids:
            inst = self.instances.get(inst_id)
            if (inst and 
                inst.model_id == model_id and 
                tuple(inst.gpus) == tuple(gpu_set) and
                inst.state == InstState.SLEPT):
                return inst
        return None
    
    def _find_active_instance(
        self,
        model_id: str,
        gpu_set: Tuple[int, ...],
    ) -> Optional[Instance]:
        """Find an active instance of model on given GPU set."""
        for inst_id in self.active_ids:
            inst = self.instances.get(inst_id)
            if (inst and 
                inst.model_id == model_id and 
                tuple(inst.gpus) == tuple(gpu_set) and
                inst.state == InstState.ACTIVE):
                return inst
        return None
    
    def _find_gpu_set_for_model(self, model_id: str, assigned_gpu: int) -> Optional[Tuple[int, ...]]:
        """Find GPU set containing the assigned GPU for a model.
        
        Args:
            model_id: The model to find a GPU set for
            assigned_gpu: The GPU ID assigned by MCF solver
            
        Returns:
            GPU set tuple containing the assigned GPU, or None if not found
        """
        card = self.model_cards.get(model_id)
        tp_min = card.tp_min if card else 1
        
        # Find GPU set containing the assigned GPU that meets tp_min
        for gpu_set in sorted(self.gpu_sets, key=len):
            if assigned_gpu in gpu_set and len(gpu_set) >= tp_min:
                # Check if all GPUs are stable
                if all(self.gpus[g].is_stable for g in gpu_set if g in self.gpus):
                    return gpu_set
        
        # Fallback: return single GPU set if tp_min=1
        if tp_min == 1 and assigned_gpu in self.gpus and self.gpus[assigned_gpu].is_stable:
            return (assigned_gpu,)
        
        return None
    
    def _should_offload(
        self,
        gpu: GPUInfo,
        current_model: str,
        target_model: str,
    ) -> bool:
        """
        Determine if current model should be offloaded (vs slept).
        
        Offload if sleeping would exceed alpha * VRAM limit.
        """
        current_card = self.model_cards.get(current_model)
        target_card = self.model_cards.get(target_model)
        
        if not current_card or not target_card:
            return True  # Default to offload if unknown
        
        # Estimate slept memory usage
        tp = gpu.tp_by_model.get(current_model, 1)
        current_slept_mem = current_card.slept_mem_MB(tp)
        
        # Add existing slept models
        total_slept = current_slept_mem
        for m in gpu.get_slept_models():
            card = self.model_cards.get(m)
            if card:
                total_slept += card.slept_mem_MB(gpu.tp_by_model.get(m, 1))
        
        # Add target model's slept memory
        total_slept += target_card.slept_mem_MB(target_card.tp_min)
        
        return total_slept > gpu.weight_cap_MB
    
    # -----------------
    # Plan Execution
    # -----------------
    
    async def _apply_plan(self, plan: ReconfigPlan) -> None:
        """Execute the reconfiguration plan."""
        now = time.monotonic()
        
        # 1. Sleep instances (can be parallelized too if desired)
        await asyncio.gather(*[self._sleep_instance(inst_id) for inst_id in plan.to_sleep])
        
        # 2. Offload instances
        await asyncio.gather(*[self._offload_instance(inst_id) for inst_id in plan.to_offload])
        
        # 3. Wake instances
        await asyncio.gather(*[self._wake_instance(inst_id) for inst_id in plan.to_wake])
        
        # 4. Load new instances - RUN IN PARALLEL
        await asyncio.gather(*[
            self._load_instance(gpu_set, model_id) 
            for gpu_set, model_id in plan.to_load
        ])
    
    async def _sleep_instance(self, inst_id: str) -> None:
        """Put an instance to sleep."""
        inst = self.instances.get(inst_id)
        if not inst or inst.state != InstState.ACTIVE:
            return
        
        logger.info(f"Sleeping instance {inst_id} ({inst.model_id})")
        
        try:
            # Drain first
            await self.controller.drain_until_empty(inst, timeout_s=600.0)
            
            # Sleep
            t0 = time.monotonic()
            elapsed = await self.controller.sleep(inst)
            
            # Measure GPU memory consumption after sleeping
            # Wait a short time for memory to stabilize
            await asyncio.sleep(0.1)
            pid_mem = await self.controller.pid_used_MB()
            
            # Calculate slept memory per GPU for this instance
            tp = len(inst.gpus)
            slept_mem_samples = []
            for g in inst.gpus:
                pid = inst.pid_by_gpu.get(g)
                if pid:
                    mem_MB = pid_mem.get((g, pid), 0)
                    if mem_MB > 0:
                        slept_mem_samples.append(mem_MB)
            
            # Update EMA for timing
            card = self.model_cards.get(inst.model_id)
            if card:
                card.update_sleep(elapsed, alpha=self.timing_ema_alpha)
                
                # Update slept memory EMA if we got valid measurements
                if slept_mem_samples:
                    avg_slept_mem = sum(slept_mem_samples) / len(slept_mem_samples)
                    card.update_slept_mem(tp, avg_slept_mem, alpha=self.timing_ema_alpha)
                    logger.info(
                        f"Measured slept memory for {inst.model_id}: "
                        f"{avg_slept_mem:.0f} MB per GPU (tp={tp})"
                    )
            
            # Update state
            inst.state = InstState.SLEPT
            self.active_ids.discard(inst_id)
            self.slept_ids.add(inst_id)
            
            # Update GPU state
            now = time.monotonic()
            for g in inst.gpus:
                self.gpus[g].set_resident(
                    inst.model_id, Residence.SLEPT, now,
                    pid=inst.pid_by_gpu.get(g),
                    tp=len(inst.gpus),
                    url=inst.base_url
                )
            
            self._state.total_sleeps += 1
            
        except Exception as e:
            logger.error(f"Failed to sleep instance {inst_id}: {e}")
    
    async def _offload_instance(self, inst_id: str) -> None:
        """Kill and remove an instance."""
        inst = self.instances.get(inst_id)
        if not inst:
            return
        
        logger.info(f"Offloading instance {inst_id} ({inst.model_id})")
        
        try:
            t0 = time.monotonic()
            elapsed = await self.controller.kill(inst)
            
            # Update EMA
            card = self.model_cards.get(inst.model_id)
            if card:
                card.update_offload(elapsed, alpha=self.timing_ema_alpha)
            
            # Update state
            inst.state = InstState.VANISHING
            self.active_ids.discard(inst_id)
            self.slept_ids.discard(inst_id)
            self.instances.pop(inst_id, None)
            
            # Update GPU state
            for g in inst.gpus:
                self.gpus[g].evict_model(inst.model_id)
            
            self._state.total_offloads += 1
            
        except Exception as e:
            logger.error(f"Failed to offload instance {inst_id}: {e}")
    
    async def _wake_instance(self, inst_id: str) -> None:
        """Wake a slept instance."""
        inst = self.instances.get(inst_id)
        if not inst or inst.state != InstState.SLEPT:
            return
        
        logger.info(f"Waking instance {inst_id} ({inst.model_id})")
        
        try:
            t0 = time.monotonic()
            elapsed = await self.controller.wake(inst)
            
            # Update EMA
            card = self.model_cards.get(inst.model_id)
            if card:
                card.update_wake(elapsed, alpha=self.timing_ema_alpha)
            
            # Update state
            inst.state = InstState.ACTIVE
            inst.accept_new = True
            self.slept_ids.discard(inst_id)
            self.active_ids.add(inst_id)
            
            # Update GPU state
            now = time.monotonic()
            for g in inst.gpus:
                self.gpus[g].set_resident(
                    inst.model_id, Residence.ACTIVE, now,
                    pid=inst.pid_by_gpu.get(g),
                    tp=len(inst.gpus),
                    url=inst.base_url
                )
            
            self._state.total_wakes += 1
            
        except Exception as e:
            logger.error(f"Failed to wake instance {inst_id}: {e}")
    
    async def _load_instance(
        self,
        gpu_set: Tuple[int, ...],
        model_id: str,
    ) -> None:
        """Load a new instance."""
        logger.info(f"Loading instance for {model_id} on GPUs {gpu_set}")
        
        # Mark as loading to prevent duplicate decisions
        self._state.loading_models.add(model_id)
        
        try:
            tp = len(gpu_set)
            gpu_mem_util = min([
                (1 - self.gpus[g].alpha)
                for g in gpu_set
            ])
            # gpu_mem_util = (
            #     ManagedVLLMController.compute_gpu_mem_util(gpu_set)
            #     if isinstance(self.controller, ManagedVLLMController)
            #     else 0.9
            # )
            
            t0 = time.monotonic()
            inst = await self.controller.start(model_id, gpu_set, tp, gpu_mem_util)
            elapsed = time.monotonic() - t0
            
            # Update EMA
            card = self.model_cards.get(model_id)
            if card:
                card.update_load(elapsed, alpha=self.timing_ema_alpha)
            
            # Register instance
            self.instances[inst.instance_id] = inst
            self.active_ids.add(inst.instance_id)
            
            # Update GPU state
            now = time.monotonic()
            for g in inst.gpus:
                self.gpus[g].set_resident(
                    model_id, Residence.ACTIVE, now,
                    pid=inst.pid_by_gpu.get(g),
                    tp=len(inst.gpus),
                    url=inst.base_url
                )
            
            self._state.total_loads += 1
            
        except Exception as e:
            logger.error(f"Failed to load instance for {model_id}: {e}")
        finally:
            self._state.loading_models.discard(model_id)
    
    # -----------------
    # Status
    # -----------------
    
    def get_stats(self) -> Dict[str, Any]:
        """Get scheduler statistics."""
        return {
            "total_reconfigs": self._state.total_reconfigs,
            "total_loads": self._state.total_loads,
            "total_wakes": self._state.total_wakes,
            "total_sleeps": self._state.total_sleeps,
            "total_offloads": self._state.total_offloads,
            "loading_models": list(self._state.loading_models),
            "last_reconfig_time": self._state.last_reconfig_time,
        }
