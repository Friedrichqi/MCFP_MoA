#!/usr/bin/env python3
# alt_reload_switch.py
import os
import time
from typing import Tuple
from vllm import LLM, SamplingParams

MODEL1 = "Qwen/Qwen3-0.6B"
MODEL2 = "Qwen/Qwen2.5-0.5B-Instruct"  # vision model

DEVICE = "0"
GPU_MEM_UTIL = 0.85
MAX_TOKENS = 32

def gen_once(llm: LLM, prompt: str) -> Tuple[str, float]:
    sp = SamplingParams(max_tokens=MAX_TOKENS, temperature=0.7, top_p=0.95)
    t0 = time.perf_counter()
    out = llm.generate(prompts=[prompt], sampling_params=sp)
    dt = time.perf_counter() - t0
    return out[0].outputs[0].text.strip(), dt

def time_call(fn, *args, **kwargs) -> Tuple[object, float]:
    t0 = time.perf_counter()
    ret = fn(*args, **kwargs)
    return ret, time.perf_counter() - t0

def load_model(name: str, use_sleep: bool=False) -> Tuple[LLM, float]:
    return time_call(
        LLM,
        model=name,
        gpu_memory_utilization=GPU_MEM_UTIL,
        **({"enable_sleep_mode": True} if use_sleep else {})
    )

def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = DEVICE
    print(f"[alt_reload_switch] CUDA device: {DEVICE}")

    totals = {}
    t_total0 = time.perf_counter()

    # --- LLM1: load → infer ---
    print(f"\nLoading LLM1: {MODEL1}")
    llm1, t_load1_a = load_model(MODEL1)
    print(f"LLM1 loaded in {t_load1_a:.3f}s")

    _, t_inf1_a = gen_once(llm1, f"Say a short greeting. Model: {MODEL1}")
    print(f"LLM1 inference (A): {t_inf1_a:.3f}s")

    # Delete LLM1
    del llm1
    import gc; gc.collect()

    # --- LLM2: load → infer ---
    print(f"\nLoading LLM2: {MODEL2}")
    llm2, t_load2_a = load_model(MODEL2)
    print(f"LLM2 loaded in {t_load2_a:.3f}s")

    _, t_inf2_a = gen_once(llm2, f"Say a short greeting. Model: {MODEL2}")
    print(f"LLM2 inference (A): {t_inf2_a:.3f}s")

    # Delete LLM2
    del llm2
    gc.collect()

    # --- LLM1 again: load → infer ---
    print(f"\nReloading LLM1: {MODEL1}")
    llm1b, t_load1_b = load_model(MODEL1)
    print(f"LLM1 reloaded in {t_load1_b:.3f}s")

    _, t_inf1_b = gen_once(llm1b, f"How are you after reload? Model: {MODEL1}")
    print(f"LLM1 inference (B): {t_inf1_b:.3f}s")

    del llm1b
    gc.collect()

    # --- LLM2 again: load → infer ---
    print(f"\nReloading LLM2: {MODEL2}")
    llm2b, t_load2_b = load_model(MODEL2)
    print(f"LLM2 reloaded in {t_load2_b:.3f}s")

    _, t_inf2_b = gen_once(llm2b, f"How are you after reload? Model: {MODEL2}")
    print(f"LLM2 inference (B): {t_inf2_b:.3f}s")

    del llm2b
    gc.collect()

    t_total = time.perf_counter() - t_total0

    totals["LLM1 load A"] = t_load1_a
    totals["LLM1 inf  A"] = t_inf1_a
    totals["LLM2 load A"] = t_load2_a
    totals["LLM2 inf  A"] = t_inf2_a
    totals["LLM1 load B"] = t_load1_b
    totals["LLM1 inf  B"] = t_inf1_b
    totals["LLM2 load B"] = t_load2_b
    totals["LLM2 inf  B"] = t_inf2_b
    totals["TOTAL (reload)"] = t_total

    print("\n=== Summary (Traditional Reload) ===")
    for k, v in totals.items():
        print(f"{k:18s}: {v:.3f}s")

    print("\n✅ Done (reload alternating LLM1→LLM2→LLM1→LLM2)")

if __name__ == "__main__":
    main()
