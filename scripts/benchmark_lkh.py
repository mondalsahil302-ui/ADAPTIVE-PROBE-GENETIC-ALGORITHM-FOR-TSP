"""
Reproducible Benchmark Suite for APGGS and Selective LKH.
Measures runtime, LKH overhead, generation throughput, route validity, and solution quality.
"""
from __future__ import annotations

import json
import os
import platform
import sys
import time
import tracemalloc
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.optimizer.core import run_generation, setup_state
from backend.optimizer.models import load_any_instance, valid_tour


def benchmark_instance(instance_path: str, seed: int = 42, max_gens: int = 35, patience: int = 30) -> dict:
    tracemalloc.start()
    t_start = time.perf_counter()

    inst = load_any_instance(instance_path)
    st = setup_state(inst, {"seed": seed, "max_generations": max_gens, "no_improvement_limit": patience})

    initial_cost = float(st["global_best_cost"])
    lkh_durations = []
    lkh_improved_global = 0
    lkh_improved_individual = 0

    gen_count = 0
    while gen_count < max_gens and not st.get("is_stopped", False):
        rec = run_generation(st)
        gen_count += 1

        lkh_dur = rec.get("stage_times", {}).get("selective_lkh", 0.0)
        if rec.get("lkh_applied", False):
            lkh_durations.append(lkh_dur)
            if rec.get("new_global_best", False) and "Selective LKH" in rec.get("best_method", ""):
                lkh_improved_global += 1
            if rec.get("lkh_improvements", 0) > 0:
                lkh_improved_individual += rec.get("lkh_improvements", 0)

        if st.get("converged", False) or st.get("no_improvement", 0) >= patience:
            break

    lkh_invocations = st.get("lkh_calls_total", 0)
    total_time = time.perf_counter() - t_start
    current_mem, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    final_cost = float(st["global_best_cost"])
    is_valid = valid_tour(st["global_best_tour"], inst.n)
    total_lkh_time = sum(lkh_durations)

    return {
        "instance": inst.name,
        "cities": inst.n,
        "seed": seed,
        "total_time_seconds": round(total_time, 4),
        "completed_generations": gen_count,
        "lkh_invocations": lkh_invocations,
        "total_lkh_time_seconds": round(total_lkh_time, 4),
        "avg_lkh_generation_duration": round(total_lkh_time / max(1, len(lkh_durations)), 4),
        "max_lkh_generation_duration": round(max(lkh_durations) if lkh_durations else 0.0, 4),
        "pct_time_in_lkh": round((total_lkh_time / max(1e-9, total_time)) * 100, 2),
        "lkh_improved_global": lkh_improved_global,
        "lkh_improved_individual": lkh_improved_individual,
        "initial_best_cost": initial_cost,
        "final_best_cost": final_cost,
        "route_valid": is_valid,
        "peak_memory_mb": round(peak_mem / (1024 * 1024), 2),
        "stop_reason": st.get("stop_reason", "Completed"),
    }


def run_benchmark(output_filename: str = "optimized_benchmark.json"):
    benchmark_dir = PROJECT_ROOT / "benchmarks"
    benchmark_dir.mkdir(exist_ok=True)

    instances = [
        PROJECT_ROOT / "tests" / "fixtures" / "demo10.tsp",
        PROJECT_ROOT / "tests" / "fixtures" / "bays29.tsp",
        PROJECT_ROOT / "tests" / "fixtures" / "berlin52.tsp",
        PROJECT_ROOT / "tests" / "fixtures" / "bier127.tsp",
    ]

    results = {}
    for p in instances:
        if not p.exists():
            continue
        print(f"[*] Benchmarking {p.name}...")
        res = benchmark_instance(str(p), seed=42, max_gens=35, patience=30)
        results[res["instance"]] = res
        print(
            f"    Total: {res['total_time_seconds']}s | LKH: {res['total_lkh_time_seconds']}s "
            f"({res['pct_time_in_lkh']}%) | Calls: {res['lkh_invocations']} | Final Cost: {res['final_best_cost']}"
        )

    summary = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "python_version": sys.version,
        "results": results,
    }

    out_path = benchmark_dir / output_filename
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\n[+] Benchmark saved to: {out_path}")
    return summary


if __name__ == "__main__":
    out_file = sys.argv[1] if len(sys.argv) > 1 else "optimized_benchmark.json"
    run_benchmark(out_file)

