"""
Automated Unit and Integration Tests for APGGS Stopping Semantics and State Lifecycle.
Verifies all Phase 1.5 mandatory criteria:
- Consecutive non-improving generations increment counter
- Global-best improvements reset counter to 0
- Improvement on the 29th generation prevents premature termination
- Thirty consecutive non-improving generations terminate with stopped_stagnation
- No generations executed after stopping
- Max generations limit respected independently
- Manual cancellation terminates cleanly with stopped_by_user
- Best tour returned is historical best
"""
import sys
import unittest
from pathlib import Path
import numpy as np

# Add backend directory to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
backend_dir = PROJECT_ROOT / "backend"
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from algorithm_core import (
    GAConfig,
    Individual,
    TSPInstance,
    make_config,
    run_generation,
    setup_state,
    tour_length,
)


def create_deterministic_instance(n: int = 10) -> TSPInstance:
    """Creates a deterministic small TSP instance for fast, reproducible testing."""
    rng = np.random.RandomState(42)
    coords = rng.rand(n, 2) * 100.0
    d = coords[:, None, :] - coords[None, :, :]
    dist = np.sqrt((d ** 2).sum(axis=2))
    return TSPInstance("test10", n, list(range(n)), dist, coords, "EUC_2D")


class TestStoppingCriterion(unittest.TestCase):

    def setUp(self):
        self.instance = create_deterministic_instance(10)

    def test_improving_sequence_resets_counter(self):
        """Global-best improvement must reset the no-improvement counter to 0."""
        st = setup_state(self.instance, {"no_improvement_limit": 30, "max_generations": 50})
        self.assertEqual(st["no_improvement"], 0)

        # Run 5 generations
        for _ in range(5):
            run_generation(st)

        # If any generation improved global best, counter was reset
        # Let's artificially force a non-improving state then a genuine improvement
        st["no_improvement"] = 15
        prev_cost = st["global_best_cost"]

        # Inject an individual better than current global best by 10.0
        better_tour = st["global_best_tour"][:]
        # Create better individual
        dist = st["instance"].dist
        st["global_best_cost"] = prev_cost + 50.0  # make current global worse so pop[0] will beat it
        rec = run_generation(st)

        # Counter must be reset to 0
        self.assertEqual(st["no_improvement"], 0)
        self.assertTrue(st["global_best_cost"] < prev_cost + 50.0)

    def test_stagnation_terminates_at_threshold(self):
        """Exactly 30 consecutive generations without global-best improvement must stop the GA."""
        patience = 30
        st = setup_state(self.instance, {"no_improvement_limit": patience, "max_generations": 100})
        
        # Set global best cost to 0.0 so no generated tour can ever beat it (simulating stagnation)
        st["global_best_cost"] = 0.0

        for expected_streak in range(1, patience + 1):
            self.assertFalse(st.get("is_stopped", False), f"Premature stop at streak {expected_streak - 1}")
            rec = run_generation(st)
            self.assertEqual(st["no_improvement"], expected_streak)

        # At generation = patience (30), experiment must be stopped due to stagnation
        self.assertTrue(st.get("is_stopped", True))
        self.assertEqual(st.get("state"), "stopped_stagnation")
        self.assertIn(f"No improvement for {patience} consecutive generations", st.get("stop_reason", ""))

    def test_improvement_on_29th_generation_prevents_premature_stop(self):
        """An improvement on the 29th non-improving generation resets counter and continues."""
        patience = 30
        st = setup_state(self.instance, {"no_improvement_limit": patience, "max_generations": 100})
        
        # Simulate 28 non-improving generations
        st["no_improvement"] = 28
        st["global_best_cost"] = 999999.0  # high cost so next generation improves

        rec = run_generation(st)
        # Should have improved, resetting counter to 0 instead of reaching 29
        self.assertEqual(st["no_improvement"], 0)
        self.assertFalse(st.get("is_stopped", False))

        # Now simulate reaching 29 again
        st["no_improvement"] = 29
        # Set impossible best cost so it fails to improve on the 30th
        st["global_best_cost"] = 0.0
        rec30 = run_generation(st)

        # Now it hits 30 and stops
        self.assertEqual(st["no_improvement"], 30)
        self.assertTrue(st.get("is_stopped", True))
        self.assertEqual(st.get("state"), "stopped_stagnation")

    def test_no_generations_executed_after_stopping(self):
        """Terminated worker must not execute another generation if called again."""
        st = setup_state(self.instance, {"no_improvement_limit": 5, "max_generations": 50})
        st["global_best_cost"] = 0.0  # force stagnation

        for _ in range(5):
            run_generation(st)

        self.assertTrue(st.get("is_stopped", True))
        stopped_gen = st["generation"]
        records_count = len(st["records"])

        # Attempt to run another generation
        extra_rec = run_generation(st)

        # Generation must NOT increment
        self.assertEqual(st["generation"], stopped_gen)
        self.assertEqual(len(st["records"]), records_count)

    def test_max_generations_respected_independently(self):
        """Maximum generation limit stops experiment even when stagnation threshold is not reached."""
        max_gens = 10
        st = setup_state(self.instance, {"no_improvement_limit": 100, "max_generations": max_gens})

        for _ in range(max_gens):
            run_generation(st)

        self.assertTrue(st.get("is_stopped", True))
        self.assertEqual(st.get("state"), "stopped_max_generations")
        self.assertEqual(st["generation"], max_gens)

    def test_manual_cancellation_terminates_cleanly(self):
        """User cancellation requests stop at boundary and sets stopped_by_user."""
        st = setup_state(self.instance, {"no_improvement_limit": 50, "max_generations": 50})
        run_generation(st)
        self.assertFalse(st.get("is_stopped", False))

        # User requests stop
        st["stop_requested"] = True
        run_generation(st)

        self.assertTrue(st.get("is_stopped", True))
        self.assertEqual(st.get("state"), "stopped_by_user")
        self.assertIn("user request", st.get("stop_reason", "").lower())

    def test_preserves_best_historical_tour(self):
        """Final best solution returned is the global best found, not just final population best."""
        st = setup_state(self.instance, {"no_improvement_limit": 30, "max_generations": 50})
        # Record current best
        initial_best = st["global_best_cost"]
        initial_tour = st["global_best_tour"][:]

        # Run 5 generations
        for _ in range(5):
            run_generation(st)

        # The global best tour length must match global_best_cost
        calc_cost = tour_length(st["global_best_tour"], self.instance.dist)
        self.assertAlmostEqual(calc_cost, st["global_best_cost"], places=5)
        self.assertLessEqual(st["global_best_cost"], initial_best)


if __name__ == "__main__":
    unittest.main()

