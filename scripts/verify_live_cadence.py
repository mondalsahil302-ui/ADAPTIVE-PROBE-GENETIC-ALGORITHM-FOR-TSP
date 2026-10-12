"""
Verification script demonstrating live 32-generation run on data/bier127.tsp.
Captures generation logs, LKH invocations, and execution timing.
"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.algorithm_core import load_any_instance, setup_state, run_generation, make_config

def main():
    dataset_name = sys.argv[1] if len(sys.argv) > 1 else "data/a280.tsp"
    inst = load_any_instance(dataset_name)
    cfg = make_config(
        population_size=30,
        max_generations=32,
        no_improvement_limit=100,
        lkh_cadence=15,
        lkh_min_gap=15,
    )
    st = setup_state(inst, cfg)

    print(f"=== Verification Run on {inst.name} ({inst.n} cities) ===")
    print(f"Config: Cadence={cfg.lkh_cadence}, MinGap={cfg.lkh_min_gap}, MaxCalls={cfg.lkh_max_calls}")
    print("=" * 60)

    for g in range(1, 33):
        rec = run_generation(st)
        if st.get("is_stopped"):
            break

    print("=" * 60)
    print(f"Run completed at Generation {st['generation']}")
    print(f"Best Cost: {st['global_best_cost']:,.2f}")
    print(f"LKH Calls Total: {st.get('lkh_calls_total')}")
    print(f"LKH Last Duration: {st.get('lkh_last_duration', 0.0):.4f} seconds")
    print(f"LKH Total Duration: {st.get('lkh_total_duration', 0.0):.4f} seconds")
    print(f"LKH Final Status: {st.get('lkh_status')}")

if __name__ == "__main__":
    main()
