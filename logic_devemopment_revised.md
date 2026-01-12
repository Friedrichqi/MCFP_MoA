# Scheduling & Hierarchical Resource Management (Revised)

## Problem Setup
- Goal: serve multi-agent DAG jobs on constrained GPUs running vLLM while minimizing end-to-end latency and maximizing GPU utilization.
- Cadence: two loops run continuously — request scheduling every `δt1` (default 1s) and GPU reconfiguration every `δt2` (default 5s).
- Inputs: GPU inventory with current loads, model cards with timing/memory profiles, active/slept instances, and a queue of agent requests derived from job DAGs.

## Entities & Variables
- **GPU (per GPU _g_)**
  - `gpu_id`, `vram_total_MB`.
  - `alpha`: max fraction of VRAM allowed for slept weights.
  - `state ∈ {STABLE, UNSTABLE}` (only STABLE participates in scheduling).
  - `resident`: map `model_id -> {ACTIVE | SLEPT}`.
  - `pid_by_model_id`, `url_by_model_id`, `last_used`.
- **Model Card (per model _m_)**
  - `model_id`.
  - `tp_min`: minimum tensor-parallel degree.
  - Timing: `t_wake_s`, `t_sleep_s`, `t_load_s`, `t_offload_s`.
  - Memory: `slept_mem_tp1_MB` (tp=1), `slept_mem_tpg1_MB` (tp>1 per GPU).
- **Instance (per logical deployment _i_ of model _m_)**
  - `instance_id`, `model_id`, `base_url`, `port`.
  - `state ∈ {LOADING, ACTIVE, SLEPT, VANISHING}`.
  - `pid_by_gpu`, `vram_by_gpu`, `created_time`, `last_used`.
- **Job & Requests**
  - Each job is a DAG; each node is an agent (LLM invocation). Multiple nodes may use the same LLM but are treated separately.
  - An edge `u -> v` means outputs of `u` concatenate into inputs of `v`.
  - **Lists**: `potential` holds edges/nodes awaiting prerequisites; `waiting_for_scheduling` holds ready requests with recorded arrival times.
- **Draining Latency (per active instance on GPU _g_)**
  - From vLLM metrics: `drain_latency = (num_requests * latency_sum) / latency_count`.
  - Used both for immediate routing and as edge cost in GPU selection.

## Scheduling Policy (Requests, every `δt1`)
1. **Activate ready nodes**: Move requests with `in_degree == 0` from `potential` into `waiting_for_scheduling`; stamp arrival time.
2. **Route to active instances**: For each request in `waiting_for_scheduling`, if an ACTIVE instance for its agent/model exists, send to the instance with minimal `drain_latency`.
3. **Mark unmet demand**: Requests without an ACTIVE target contribute required models to the GPU Scheduler for possible loading/activation.

## GPU Reconfiguration via Min-Cost Flow on new graphs (every `δt2`)
1. **Graph layers**
   - **L1 Source → L2 GPUs**: Include only STABLE GPUs. Capacity = 1 (a GPU can be assigned once per cycle). Cost = current `drain_latency` of the GPU.
   - **L2 GPUs → L3 Model Copies**: L3 contains one copy of each needed model (from `waiting_for_scheduling` + `potential`, excluding already-loading models) per GPU to capture GPU-specific costs.
   - **L3 Models → L4 Sink**: Unlimited capacity (TP/DP supported).
2. **Edge costs: GPU → Model (sleep vs offload)**
   - Compute `sleepable = slept_weights_on_g + slept_mem(active_model_on_g) + slept_mem_tp1_MB(model_j)`.
   - **Case 1: sleep fits** if `sleepable <= alpha * vram_total_MB_g`  
     - Sleep cost = `t_sleep_s(active_model_on_g)`.  
     - Switch penalty = `(requests needing active_model_on_g) * t_wake_s(active_model_on_g)`.
   - **Case 2: offload required** otherwise  
     - Offload cost = `t_offload_s(model_with_lowest_cost_on_g)` chosen among residents, weighted by their outstanding demand.  
     - Switch penalty = `(requests needing model_with_lowest_cost_on_g) * t_load_s(model_with_lowest_cost_on_g)`.
   - Total edge cost = sleep/offload cost + switch penalty if active_model_on_g != model_j else 0.
3. **Edge costs: Model → Sink (load/activate vs waiting relief)**
   - If GPU has active instances of `model_j`: activation/loading cost = 0
   - Elif GPU has slept weights of `model_j`: activation cost = `t_wake_s(model_j)`; else loading cost = `t_load_s(model_j)`.
   - Gross waiting relief = sum of `(now - arrival_time)` for all requests needing `model_j` in `waiting_for_scheduling`.
   - Edge cost = activation/loading cost − gross waiting relief.
4. **Solve min-cost-flow** to maximize flow (GPU assignments) with minimal total cost (captures utilization vs latency trade-off).
5. **Apply plan**
   - Load/activate/sleep/offload instances per flow solution.
   - Update `resident`, `pid_by_model_id`, URLs, and states on GPUs and instances.
   - Refresh model cards with EMA on observed times; update GPU/instance `state` and `last_used`.

## Operational Loop
- Run Request Scheduler (`δt1`) and GPU Reconfiguration (`δt2`) continually.
- Exclude currently loading models from new load decisions to avoid thrash.
- After each cycle, persist updated states and metrics for the next iteration.
