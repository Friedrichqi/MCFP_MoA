#!/usr/bin/env python3
# alt_sleep_switch.py (sleep mode; with Phi-3-vision)
import os
import time
from typing import Tuple
from vllm import LLM, SamplingParams

MODEL1 = "Qwen/Qwen3-0.6B"
MODEL2 = "Qwen/Qwen2.5-0.5B-Instruct"  # vision model

DEVICE = "0"
GPU_MEM_UTIL = 0.85
SLEEP_LEVEL = 2
WAKE_TAGS = ["weights", "kv_cache"]
MAX_TOKENS = 32

def gen_once(llm: LLM, prompt: str) -> Tuple[str, float]:
    sp = SamplingParams(max_tokens=MAX_TOKENS, temperature=0.7, top_p=0.95)
    t0 = time.perf_counter()
    out = llm.generate(prompts=[prompt], sampling_params=sp)
    dt = time.perf_counter() - t0
    return out[0].outputs[0].text.strip(), dt

def time_call(fn, *args, **kwargs):
    t0 = time.perf_counter()
    ret = fn(*args, **kwargs)
    return ret, time.perf_counter() - t0

def load_llm(model: str, enable_sleep: bool) -> Tuple[LLM, float]:
    # Add trust_remote_code=True for repos that require custom code
    kwargs = {
        "model": model,
        "gpu_memory_utilization": GPU_MEM_UTIL,
        "trust_remote_code": True if "Phi-3-vision" in model else False,
    }
    if enable_sleep:
        kwargs["enable_sleep_mode"] = True
    return time_call(LLM, **kwargs)

def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = DEVICE
    print(f"[alt_sleep_switch] CUDA device: {DEVICE}")

    totals = {}
    t0_total = time.perf_counter()

    # Load LLM1
    print(f"\nLoading LLM1: {MODEL1}")
    llm1, t_load1 = load_llm(MODEL1, enable_sleep=True)
    print(f"LLM1 loaded in {t_load1:.3f}s")
    _, t_inf1_a = gen_once(llm1, f"Say a short greeting. Model: {MODEL1}")
    print(f"LLM1 inference (A): {t_inf1_a:.3f}s")
    _, t_sleep1_a = time_call(llm1.sleep, level=SLEEP_LEVEL)
    print(f"LLM1 sleep L{SLEEP_LEVEL}: {t_sleep1_a:.3f}s")

    # Load LLM2 (vision; needs trust_remote_code)
    print(f"\nLoading LLM2: {MODEL2}")
    llm2, t_load2 = load_llm(MODEL2, enable_sleep=True)
    print(f"LLM2 loaded in {t_load2:.3f}s")
    _, t_inf2_a = gen_once(llm2, f"Say a short greeting. Model: {MODEL2}")
    print(f"LLM2 inference (A): {t_inf2_a:.3f}s")
    _, t_sleep2_a = time_call(llm2.sleep, level=SLEEP_LEVEL)
    print(f"LLM2 sleep L{SLEEP_LEVEL}: {t_sleep2_a:.3f}s")

    # Wake LLM1
    _, t_wake1_b = time_call(llm1.wake_up, tags=WAKE_TAGS)
    print(f"LLM1 wake: {t_wake1_b:.3f}s")
    _, t_inf1_b = gen_once(llm1, f"How are you after waking? Model: {MODEL1}")
    print(f"LLM1 inference (B): {t_inf1_b:.3f}s")
    _, t_sleep1_b = time_call(llm1.sleep, level=SLEEP_LEVEL)
    print(f"LLM1 sleep L{SLEEP_LEVEL}: {t_sleep1_b:.3f}s")

    # Wake LLM2
    _, t_wake2_b = time_call(llm2.wake_up, tags=WAKE_TAGS)
    print(f"LLM2 wake: {t_wake2_b:.3f}s")
    _, t_inf2_b = gen_once(llm2, f"How are you after waking? Model: {MODEL2}")
    print(f"LLM2 inference (B): {t_inf2_b:.3f}s")

    totals.update({
        "LLM1 load": t_load1, "LLM1 inf A": t_inf1_a, "LLM1 sleep A": t_sleep1_a,
        "LLM2 load": t_load2, "LLM2 inf A": t_inf2_a, "LLM2 sleep A": t_sleep2_a,
        "LLM1 wake B": t_wake1_b, "LLM1 inf B": t_inf1_b,
        "LLM2 wake B": t_wake2_b, "LLM2 inf B": t_inf2_b,
        "TOTAL (sleep-mode)": time.perf_counter() - t0_total
    })

    print("\n=== Summary (Sleep Mode) ===")
    for k, v in totals.items():
        print(f"{k:18s}: {v:.3f}s")

    del llm2, llm1
    import gc; gc.collect()
    print("\n✅ Done (sleep mode alt 1→2→1→2)")

if __name__ == "__main__":
    main()
