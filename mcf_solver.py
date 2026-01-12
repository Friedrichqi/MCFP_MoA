"""
mcf_solver.py - Min-Cost Flow solver for GPU scheduling.

Implements the 4-layer flow graph as specified in logic_devemopment_revised.md:

Layer 1 (Source) -> Layer 2 (GPUs) -> Layer 3 (Model Copies) -> Layer 4 (Sink)

Edge costs capture the utilization vs latency trade-off:
- Source -> GPU: drain_latency of the GPU
- GPU -> Model Copy: sleep/offload cost + switch penalty
- Model Copy -> Sink: activation/loading cost - waiting relief
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

# Try to import scipy for linear sum assignment, fall back to pure Python
try:
    from scipy.optimize import linear_sum_assignment
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

# Try to import networkx for general MCF
try:
    import networkx as nx
    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False


from entities import (
    GPUInfo, GPUState, ModelCard, Instance, InstState,
    Residence, DrainLatency, ReconfigPlan
)


# -----------------------------
# Data Structures
# -----------------------------

@dataclass
class MCFEdge:
    """An edge in the min-cost flow graph."""
    source: str
    target: str
    capacity: int
    cost: float


@dataclass
class MCFAssignment:
    """A GPU-to-model assignment from the MCF solution."""
    gpu_id: int
    model_id: str
    action: str  # "load", "wake", "keep", "noop"
    cost: float


@dataclass
class MCFSolution:
    """Solution from the min-cost flow solver."""
    assignments: List[MCFAssignment]
    total_cost: float
    flow: Dict[Tuple[str, str], int] = field(default_factory=dict)


# -----------------------------
# Min-Cost Flow Solver
# -----------------------------

class MinCostFlowSolver:
    """
    Min-Cost Flow solver for GPU reconfiguration.
    
    The 4-layer graph structure:
    
    L1 (Source)
        |
        | capacity=1, cost=drain_latency
        v
    L2 (GPUs) -- only STABLE GPUs
        |
        | capacity=1, cost=sleep/offload + switch_penalty
        v
    L3 (Model Copies) -- one copy per (GPU, needed_model) pair
        |
        | capacity=∞, cost=activation/load - waiting_relief
        v
    L4 (Sink)
    
    The solver maximizes flow (GPU assignments) while minimizing total cost.
    """
    
    def __init__(self):
        self.edges: List[MCFEdge] = []
        self._gpu_nodes: Set[str] = set()
        self._model_copy_nodes: Set[str] = set()
    
    def build_graph(
        self,
        gpus: Dict[int, GPUInfo],
        needed_models: Set[str],
        model_cards: Dict[str, ModelCard],
        instances: Dict[str, Instance],
        drain_latencies: Dict[int, float],  # gpu_id -> drain_latency
        waiting_times: Dict[str, float],    # model_id -> sum of (now - t_arr)
        request_counts: Dict[str, int],     # model_id -> count of waiting requests
    ) -> None:
        """
        Build the 4-layer min-cost flow graph.
        
        Args:
            gpus: GPU inventory
            needed_models: Models needed (from waiting + potential, excluding loading)
            model_cards: Model timing/memory profiles
            instances: Current instances
            drain_latencies: Current drain latency per GPU
            waiting_times: Gross waiting relief per model
            request_counts: Number of requests needing each model
        """
        self.edges.clear()
        self._gpu_nodes.clear()
        self._model_copy_nodes.clear()
        
        source = "SOURCE"
        sink = "SINK"
        
        # Get STABLE GPUs only
        stable_gpus = [g for g in gpus.values() if g.state == GPUState.STABLE]
        
        if not stable_gpus or not needed_models:
            return
        
        # L1 -> L2: Source -> GPUs
        for gpu in stable_gpus:
            gpu_node = f"GPU_{gpu.gpu_id}"
            self._gpu_nodes.add(gpu_node)
            
            drain_cost = drain_latencies.get(gpu.gpu_id, 0.0)
            self.edges.append(MCFEdge(
                source=source,
                target=gpu_node,
                capacity=1,
                cost=drain_cost,
            ))
        
        # L2 -> L3: GPUs -> Model Copies
        for gpu in stable_gpus:
            gpu_node = f"GPU_{gpu.gpu_id}"
            
            for model_id in needed_models:
                model_card = model_cards.get(model_id)
                if not model_card:
                    continue
                
                # Check if GPU set is large enough for this model
                if model_card.tp_min > 1:
                    # For now, skip multi-GPU models on single GPU nodes
                    # Full implementation would need GPU set handling
                    continue
                
                model_copy_node = f"MODEL_{model_id}@GPU_{gpu.gpu_id}"
                self._model_copy_nodes.add(model_copy_node)
                
                # Compute GPU -> Model cost
                gpu_to_model_cost = self._compute_gpu_to_model_cost(
                    gpu, model_id, model_card, instances, request_counts
                )
                
                self.edges.append(MCFEdge(
                    source=gpu_node,
                    target=model_copy_node,
                    capacity=1,
                    cost=gpu_to_model_cost,
                ))
        
        # L3 -> L4: Model Copies -> Sink
        for gpu in stable_gpus:
            for model_id in needed_models:
                model_card = model_cards.get(model_id)
                if not model_card or model_card.tp_min > 1:
                    continue
                
                model_copy_node = f"MODEL_{model_id}@GPU_{gpu.gpu_id}"
                if model_copy_node not in self._model_copy_nodes:
                    continue
                
                # Compute Model -> Sink cost
                model_to_sink_cost = self._compute_model_to_sink_cost(
                    gpu, model_id, model_card, instances, waiting_times
                )
                
                self.edges.append(MCFEdge(
                    source=model_copy_node,
                    target=sink,
                    capacity=1,  # Could be higher for DP support
                    cost=model_to_sink_cost,
                ))
    
    def _compute_gpu_to_model_cost(
        self,
        gpu: GPUInfo,
        model_id: str,
        model_card: ModelCard,
        instances: Dict[str, Instance],
        request_counts: Dict[str, int],
    ) -> float:
        """
        Compute edge cost from GPU to Model Copy.
        
        Per document:
        - Case 1 (sleep fits): sleep_cost + switch_penalty
        - Case 2 (offload required): offload_cost + switch_penalty
        
        If active model on GPU == target model, cost = 0
        """
        # Check if this model is already active on this GPU
        if gpu.is_resident(model_id) and gpu.resident.get(model_id) == Residence.ACTIVE:
            return 0.0
        
        # Get currently active model on this GPU (if any)
        active_models = gpu.get_active_models()
        if not active_models:
            # No active model, no displacement needed
            return 0.0
        
        active_model = active_models[0]  # Assume single active model per GPU for simplicity
        active_card = model_card  # Would need to look up actual card
        
        # Compute sleepable memory
        # sleepable = slept_weights_on_g + slept_mem(active_model) + slept_mem(target_model)
        current_slept = sum(
            model_card.slept_mem_MB(gpu.tp_by_model.get(m, 1))
            for m in gpu.get_slept_models()
        )
        
        tp = gpu.tp_by_model.get(active_model, 1)
        active_slept_mem = model_card.slept_mem_MB(tp)
        target_slept_mem = model_card.slept_mem_MB(model_card.tp_min)
        
        sleepable = current_slept + active_slept_mem + target_slept_mem
        
        if sleepable <= gpu.weight_cap_MB:
            # Case 1: Sleep fits
            sleep_cost = model_card.t_sleep_s
            
            # Switch penalty: requests needing active model * wake time
            requests_for_active = request_counts.get(active_model, 0)
            switch_penalty = requests_for_active * model_card.t_wake_s
            
            return sleep_cost + switch_penalty
        else:
            # Case 2: Offload required
            # Choose model with lowest cost to offload
            offload_cost = model_card.t_offload_s
            
            # Switch penalty: requests needing offloaded model * load time
            requests_for_active = request_counts.get(active_model, 0)
            switch_penalty = requests_for_active * model_card.t_load_s
            
            return offload_cost + switch_penalty
    
    def _compute_model_to_sink_cost(
        self,
        gpu: GPUInfo,
        model_id: str,
        model_card: ModelCard,
        instances: Dict[str, Instance],
        waiting_times: Dict[str, float],
    ) -> float:
        """
        Compute edge cost from Model Copy to Sink.
        
        Per document:
        - If GPU has active instances of model: cost = 0
        - If GPU has slept weights of model: cost = t_wake_s
        - Else: cost = t_load_s
        
        Then subtract gross waiting relief:
        Edge cost = activation/loading cost - gross waiting relief
        """
        # Determine activation/loading cost
        if gpu.is_resident(model_id):
            if gpu.resident[model_id] == Residence.ACTIVE:
                activation_cost = 0.0
            else:  # SLEPT
                activation_cost = model_card.t_wake_s
        else:
            activation_cost = model_card.t_load_s
        
        # Waiting relief = sum of (now - t_arr) for requests needing this model
        waiting_relief = waiting_times.get(model_id, 0.0)
        
        # Edge cost (can be negative, which is good - we want to maximize relief)
        return activation_cost - waiting_relief
    
    def solve(self) -> MCFSolution:
        """
        Solve the min-cost flow problem.
        
        Uses scipy linear_sum_assignment for bipartite matching,
        or falls back to greedy assignment.
        """
        if not self.edges:
            return MCFSolution(assignments=[], total_cost=0.0)
        
        # Extract GPU -> Model assignments from edges
        # This is a bipartite matching problem
        
        gpu_ids = sorted([int(n.split("_")[1]) for n in self._gpu_nodes])
        model_ids = set()
        
        # Parse model copy nodes to get (gpu_id, model_id) pairs
        gpu_model_costs: Dict[Tuple[int, str], float] = {}
        
        for edge in self.edges:
            if edge.source.startswith("GPU_") and edge.target.startswith("MODEL_"):
                # Extract GPU id
                gpu_id = int(edge.source.split("_")[1])
                
                # Extract model_id from MODEL_{model_id}@GPU_{gpu_id}
                target = edge.target
                model_id = target.split("@")[0].replace("MODEL_", "")
                
                model_ids.add(model_id)
                
                # Add sink cost to get total cost for this assignment
                sink_edge = next(
                    (e for e in self.edges if e.source == target and e.target == "SINK"),
                    None
                )
                sink_cost = sink_edge.cost if sink_edge else 0.0
                
                gpu_model_costs[(gpu_id, model_id)] = edge.cost + sink_cost
        
        model_ids_list = sorted(model_ids)
        
        if HAS_SCIPY and len(gpu_ids) > 0 and len(model_ids_list) > 0:
            return self._solve_scipy(gpu_ids, model_ids_list, gpu_model_costs)
        else:
            return self._solve_greedy(gpu_ids, model_ids_list, gpu_model_costs)
    
    def _solve_scipy(
        self,
        gpu_ids: List[int],
        model_ids: List[str],
        costs: Dict[Tuple[int, str], float],
    ) -> MCFSolution:
        """Solve using scipy linear sum assignment."""
        import numpy as np
        
        n_gpus = len(gpu_ids)
        n_models = len(model_ids)
        
        # Create cost matrix (GPUs x Models)
        # Use large value for infeasible assignments
        INF = 1e9
        cost_matrix = np.full((n_gpus, n_models), INF)
        
        for i, gpu_id in enumerate(gpu_ids):
            for j, model_id in enumerate(model_ids):
                if (gpu_id, model_id) in costs:
                    cost_matrix[i, j] = costs[(gpu_id, model_id)]
        
        # Solve assignment problem
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        
        assignments: List[MCFAssignment] = []
        total_cost = 0.0
        
        for i, j in zip(row_ind, col_ind):
            cost = cost_matrix[i, j]
            if cost < INF:
                gpu_id = gpu_ids[i]
                model_id = model_ids[j]
                
                # Determine action (simplified)
                action = "load"  # Default to load
                
                assignments.append(MCFAssignment(
                    gpu_id=gpu_id,
                    model_id=model_id,
                    action=action,
                    cost=cost,
                ))
                total_cost += cost
        
        return MCFSolution(assignments=assignments, total_cost=total_cost)
    
    def _solve_greedy(
        self,
        gpu_ids: List[int],
        model_ids: List[str],
        costs: Dict[Tuple[int, str], float],
    ) -> MCFSolution:
        """Greedy fallback solver."""
        assignments: List[MCFAssignment] = []
        total_cost = 0.0
        
        used_gpus: Set[int] = set()
        assigned_models: Set[str] = set()
        
        # Sort all (gpu, model) pairs by cost
        all_pairs = [
            (gpu_id, model_id, costs.get((gpu_id, model_id), math.inf))
            for gpu_id in gpu_ids
            for model_id in model_ids
        ]
        all_pairs.sort(key=lambda x: x[2])
        
        for gpu_id, model_id, cost in all_pairs:
            if gpu_id in used_gpus:
                continue
            if model_id in assigned_models:
                continue
            if cost >= 1e8:
                continue
            
            used_gpus.add(gpu_id)
            assigned_models.add(model_id)
            
            assignments.append(MCFAssignment(
                gpu_id=gpu_id,
                model_id=model_id,
                action="load",
                cost=cost,
            ))
            total_cost += cost
        
        return MCFSolution(assignments=assignments, total_cost=total_cost)


# -----------------------------
# Helper Functions
# -----------------------------

def compute_waiting_times(
    requests: List[Any],  # List of Request
    now: float,
) -> Dict[str, float]:
    """
    Compute gross waiting relief per model.
    
    waiting_time[model] = sum of (now - t_arr) for all requests needing that model
    """
    waiting: Dict[str, float] = {}
    for req in requests:
        model = req.model
        wait = max(0.0, now - req.t_arr)
        waiting[model] = waiting.get(model, 0.0) + wait
    return waiting


def compute_request_counts(requests: List[Any]) -> Dict[str, int]:
    """Count requests per model."""
    counts: Dict[str, int] = {}
    for req in requests:
        counts[req.model] = counts.get(req.model, 0) + 1
    return counts


def compute_drain_latencies(
    gpus: Dict[int, GPUInfo],
    instances: Dict[str, Instance],
    metrics: Dict[str, DrainLatency],  # instance_id -> metrics
) -> Dict[int, float]:
    """
    Compute drain latency per GPU.
    
    For each GPU, take the max drain_latency of all active instances on it.
    """
    latencies: Dict[int, float] = {g: 0.0 for g in gpus}
    
    for inst_id, inst in instances.items():
        if inst.state != InstState.ACTIVE:
            continue
        
        drain = metrics.get(inst_id, DrainLatency()).drain_latency
        
        for gpu_id in inst.gpus:
            if gpu_id in latencies:
                latencies[gpu_id] = max(latencies[gpu_id], drain)
    
    return latencies
