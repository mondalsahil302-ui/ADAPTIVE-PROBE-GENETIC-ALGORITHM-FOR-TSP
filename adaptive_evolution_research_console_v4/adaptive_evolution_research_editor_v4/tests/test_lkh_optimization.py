"""
Comprehensive Unit and Integration Test Suite for Selective LKH Optimization,
Resource Bounds, Adaptive Triggering, Symmetric Caching, and Telemetry in APGGS.
"""
from __future__ import annotations

import copy
import sys
import time
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np

# Ensure backend is on sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR / "backend"))

import algorithm_core as core
from algorithm_core import (
    GAConfig,
    Individual,
    TSPInstance,
    canonicalize_tour,
    load_any_instance,
    make_config,
    run_generation,
    selective_lkh,
    selective_lkh_detailed,
    setup_state,
    solve_lkh_bounded,
    tour_length,
    valid_tour,
    LKH_CACHE,
)
import app as app_module


def create_mock_instance(n: int = 12) -> TSPInstance:
    """Creates a deterministic synthetic TSP instance for fast, reliable unit tests."""
    rng = np.random.RandomState(123)
    coords = rng.rand(n, 2) * 200.0
    d = coords[:, None, :] - coords[None, :, :]
    dist = np.sqrt((d ** 2).sum(axis=2))
    return TSPInstance(name="mock_test", n=n, nodes=list(range(n)), dist=dist, coords=coords, edge_weight_type="EUC_2D")


class TestLKHOptimization(unittest.TestCase):

    def setUp(self):
        self.instance = create_mock_instance(12)
        LKH_CACHE.clear()

    def tearDown(self):
        LKH_CACHE.clear()

    def test_canonicalize_tour_cyclic_and_reversal_symmetry(self):
        """Cyclic rotations and reversals of a tour must map to the exact same canonical tuple."""
        t1 = [0, 1, 2, 3, 4]
        t2 = [2, 3, 4, 0, 1]  # cyclic rotation
        t3 = [4, 3, 2, 1, 0]  # reversal
        t4 = [1, 0, 4, 3, 2]  # reversed and rotated

        c1 = canonicalize_tour(t1)
        c2 = canonicalize_tour(t2)
        c3 = canonicalize_tour(t3)
        c4 = canonicalize_tour(t4)

        self.assertEqual(c1, c2, "Cyclic shift failed canonical equivalence")
        self.assertEqual(c1, c3, "Reversal failed canonical equivalence")
        self.assertEqual(c1, c4, "Rotated reversal failed canonical equivalence")
        self.assertEqual(c1[0], 0, "Canonical tour must start with minimum node index (0)")

    def test_solve_lkh_bounded_success_returns_valid_permutation(self):
        """Normal bounded LKH execution must return a valid permutation of correct length and cost."""
        tour = list(range(self.instance.n))
        dist = self.instance.dist

        polished, polished_cost, status = solve_lkh_bounded(
            tour=tour,
            dist=dist,
            runs=1,
            max_trials=20,
            timeout_sec=5.0,
            instance_name="mock_test"
        )

        self.assertTrue(valid_tour(polished, self.instance.n), "Tour must be a valid permutation 0..N-1")
        self.assertEqual(len(polished), self.instance.n)
        self.assertEqual(len(set(polished)), self.instance.n)
        calculated_cost = tour_length(polished, dist)
        self.assertAlmostEqual(polished_cost, calculated_cost, places=6)
        self.assertIn(status, ["improved", "no_improvement", "cached", "fallback", "fallback_improved"])

    def test_solve_lkh_bounded_non_improving_preserves_tour(self):
        """When LKH does not improve, the tour is preserved without data corruption."""
        tour = list(range(self.instance.n))
        dist = self.instance.dist
        orig_cost = tour_length(tour, dist)

        # Mock solver to return exactly the same tour (cost equal to orig_cost)
        with patch.object(core, "_solve_elkai_internal", return_value=tour[:]):
            polished, polished_cost, status = solve_lkh_bounded(
                tour=tour,
                dist=dist,
                runs=1,
                max_trials=10,
                timeout_sec=5.0,
                instance_name="test_preserve"
            )
            self.assertEqual(polished, tour)
            self.assertAlmostEqual(polished_cost, orig_cost, places=6)
            self.assertEqual(status, "no_improvement")

    def test_solve_lkh_bounded_timeout_graceful(self):
        """Solver timeout must gracefully abort within budget without hanging and return original tour."""
        tour = list(range(self.instance.n))
        dist = self.instance.dist
        orig_cost = tour_length(tour, dist)

        def slow_solver(*args, **kwargs):
            time.sleep(0.5)
            return tour

        with patch.object(core, "_solve_elkai_internal", side_effect=slow_solver):
            t0 = time.perf_counter()
            polished, polished_cost, status = solve_lkh_bounded(
                tour=tour,
                dist=dist,
                runs=1,
                max_trials=10,
                timeout_sec=0.05,  # 50ms timeout
                instance_name="test_timeout"
            )
            elapsed = time.perf_counter() - t0

            # Must return reasonably close to timeout budget
            self.assertLess(elapsed, 0.4, "Timeout handling blocked longer than expected")
            self.assertEqual(polished, tour)
            self.assertAlmostEqual(polished_cost, orig_cost, places=6)
            self.assertEqual(status, "timed_out")

    def test_solve_lkh_bounded_missing_elkai_fallback(self):
        """If elkai is not available, solver falls back to 2-opt/Or-opt local search without error."""
        tour = list(range(self.instance.n))
        dist = self.instance.dist

        with patch.object(core, "HAS_ELKAI", False):
            polished, polished_cost, status = solve_lkh_bounded(
                tour=tour,
                dist=dist,
                runs=1,
                max_trials=10,
                timeout_sec=2.0,
                instance_name="test_missing_elkai"
            )
            self.assertTrue(valid_tour(polished, self.instance.n))
            self.assertIn(status, ["fallback", "fallback_improved"])

    def test_solve_lkh_bounded_malformed_output_rejected(self):
        """Malformed or wrong-length solver outputs must be safely discarded."""
        tour = list(range(self.instance.n))
        dist = self.instance.dist
        orig_cost = tour_length(tour, dist)

        # Mock solver returning wrong length
        with patch.object(core, "_solve_elkai_internal", return_value=[0, 1, 2]):
            polished, polished_cost, status = solve_lkh_bounded(
                tour=tour, dist=dist, runs=1, max_trials=10, timeout_sec=2.0, instance_name="test_wrong_len"
            )
            self.assertEqual(polished, tour, "Must retain original tour on malformed output")
            self.assertEqual(polished_cost, orig_cost)

        # Mock solver returning None or empty
        with patch.object(core, "_solve_elkai_internal", return_value=[]):
            polished, polished_cost, status = solve_lkh_bounded(
                tour=tour, dist=dist, runs=1, max_trials=10, timeout_sec=2.0, instance_name="test_empty"
            )
            self.assertEqual(polished, tour)
            self.assertEqual(polished_cost, orig_cost)

    def test_solve_lkh_bounded_duplicate_nodes_rejected(self):
        """Outputs with duplicate node IDs must be rejected by valid_tour check."""
        tour = list(range(self.instance.n))
        dist = self.instance.dist
        orig_cost = tour_length(tour, dist)

        duplicate_tour = list(range(self.instance.n))
        duplicate_tour[-1] = duplicate_tour[0]  # duplicate node 0, node N-1 missing

        with patch.object(core, "_solve_elkai_internal", return_value=duplicate_tour):
            polished, polished_cost, status = solve_lkh_bounded(
                tour=tour, dist=dist, runs=1, max_trials=10, timeout_sec=2.0, instance_name="test_dup"
            )
            self.assertEqual(polished, tour, "Must reject duplicate node permutations")
            self.assertEqual(polished_cost, orig_cost)

    def test_caching_prevents_redundant_solver_calls(self):
        """Identical or rotated/reversed tour queries must hit LKH_CACHE without invoking solver."""
        tour = list(range(self.instance.n))
        dist = self.instance.dist

        solver_mock = MagicMock(return_value=tour[:])
        with patch.object(core, "_solve_elkai_internal", solver_mock):
            # First call populates cache
            _, _, status1 = solve_lkh_bounded(
                tour=tour, dist=dist, runs=1, max_trials=10, timeout_sec=2.0,
                instance_name="test_cache", cache_enabled=True
            )
            self.assertEqual(solver_mock.call_count, 1)

            # Second call with rotated tour should hit cache
            rotated = tour[2:] + tour[:2]
            _, _, status2 = solve_lkh_bounded(
                tour=rotated, dist=dist, runs=1, max_trials=10, timeout_sec=2.0,
                instance_name="test_cache", cache_enabled=True
            )
            self.assertEqual(status2, "cached")
            self.assertEqual(solver_mock.call_count, 1, "Cache hit must not invoke solver again")

            # Third call with reversed tour should hit cache
            reversed_tour = tour[::-1]
            _, _, status3 = solve_lkh_bounded(
                tour=reversed_tour, dist=dist, runs=1, max_trials=10, timeout_sec=2.0,
                instance_name="test_cache", cache_enabled=True
            )
            self.assertEqual(status3, "cached")
            self.assertEqual(solver_mock.call_count, 1, "Reversed tour cache hit must not invoke solver")

    def test_lkh_max_calls_per_run_budget_cap(self):
        """Call count cap (lkh_max_calls_per_run) strictly enforces solver execution limit."""
        cfg_overrides = {
            "lkh_max_calls_per_run": 2,
            "lkh_min_generation_gap": 1,
            "lkh_cadence": 1,
            "lkh_trigger_stagnation": 1,
            "max_generations": 10,
            "no_improvement_limit": 20
        }
        st = setup_state(self.instance, cfg_overrides)

        # Run 8 generations
        for _ in range(8):
            if st.get("is_stopped", False):
                break
            run_generation(st)

        self.assertLessEqual(st["lkh_calls_total"], 2, "LKH total calls must never exceed budget cap")

        # After budget exhaustion, records should indicate skipped_budget_exhausted
        if len(st["records"]) > 3:
            statuses = [r.get("lkh_status") for r in st["records"][3:]]
            self.assertTrue(
                any(s in ("skipped_budget_exhausted", "idle") for s in statuses),
                f"Expected skipped_budget_exhausted status, got: {statuses}"
            )

    def test_lkh_min_generation_gap_enforced(self):
        """LKH must not execute on adjacent generations when lkh_min_generation_gap > 1."""
        min_gap = 4
        cfg_overrides = {
            "lkh_min_generation_gap": min_gap,
            "lkh_cadence": 1,
            "lkh_trigger_stagnation": 1,
            "lkh_max_calls_per_run": 10,
            "max_generations": 12,
            "no_improvement_limit": 30
        }
        st = setup_state(self.instance, cfg_overrides)

        lkh_gens = []
        for g in range(1, 11):
            if st.get("is_stopped", False):
                break
            rec = run_generation(st)
            if rec.get("lkh_applied", False):
                lkh_gens.append(g)

        for i in range(1, len(lkh_gens)):
            gap = lkh_gens[i] - lkh_gens[i - 1]
            self.assertGreaterEqual(gap, min_gap, f"LKH gap violation between gen {lkh_gens[i-1]} and {lkh_gens[i]}")

    def test_stagnation_trigger_invokes_lkh(self):
        """LKH must trigger when stagnation counter meets lkh_trigger_stagnation threshold."""
        stag_threshold = 3
        cfg_overrides = {
            "lkh_trigger_stagnation": stag_threshold,
            "lkh_cadence": 9999,  # disable cadence
            "lkh_min_generation_gap": 1,
            "lkh_max_calls_per_run": 5,
            "max_generations": 10,
            "no_improvement_limit": 30
        }
        st = setup_state(self.instance, cfg_overrides)

        # Force stagnation
        st["global_best_cost"] = 0.0  # impossible to beat
        lkh_triggered = False

        for g in range(1, 6):
            rec = run_generation(st)
            if rec.get("lkh_applied", False):
                lkh_triggered = True
                self.assertGreaterEqual(rec.get("no_improvement_count", 0), stag_threshold)
                break

        self.assertTrue(lkh_triggered, "LKH should have been triggered by stagnation counter")

    def test_stagnation_counter_resets_only_on_global_improvement(self):
        """Stagnation counter resets when global best improves, but increments when individual improvement fails to beat global."""
        st = setup_state(self.instance, {"no_improvement_limit": 30, "lkh_cadence": 9999})
        st["no_improvement"] = 5
        initial_streak = st["no_improvement"]
        prev_global = st["global_best_cost"]

        # Run 1 generation where population does not beat prev_global
        st["global_best_cost"] = 0.0  # unbeatably low
        run_generation(st)
        self.assertEqual(st["no_improvement"], initial_streak + 1, "Counter must increment when global best is not beaten")

        # Now simulate a global best improvement
        st["global_best_cost"] = 999999.0  # high so next pop definitely beats it
        run_generation(st)
        self.assertEqual(st["no_improvement"], 0, "Counter must reset to 0 when global best is beaten")

    def test_stopping_criterion_at_30_gens_with_lkh_active(self):
        """GA must terminate at exactly 30 consecutive non-improving generations even with LKH enabled."""
        patience = 30
        cfg_overrides = {
            "no_improvement_limit": patience,
            "lkh_enabled": True,
            "lkh_cadence": 5,
            "lkh_trigger_stagnation": 10,
            "lkh_min_generation_gap": 5,
            "max_generations": 60
        }
        st = setup_state(self.instance, cfg_overrides)
        st["global_best_cost"] = 0.0  # enforce stagnation

        for _ in range(patience):
            self.assertFalse(st.get("is_stopped", False))
            run_generation(st)

        self.assertTrue(st.get("is_stopped", False))
        self.assertEqual(st.get("state"), "stopped_stagnation")
        self.assertIn(f"{patience} consecutive generations", st.get("stop_reason", ""))

    def test_lkh_not_invoked_after_stopped(self):
        """LKH must never be executed if is_stopped is already True."""
        st = setup_state(self.instance)
        st["is_stopped"] = True
        st["lkh_calls_total"] = 0

        solver_mock = MagicMock()
        with patch.object(core, "_solve_elkai_internal", solver_mock):
            rec = run_generation(st)
            self.assertEqual(solver_mock.call_count, 0)
            self.assertEqual(st["lkh_calls_total"], 0)

    def test_lkh_exception_isolation_does_not_crash_ga(self):
        """Exceptions in LKH execution must be safely trapped and must not crash the GA loop."""
        cfg_overrides = {
            "lkh_cadence": 1,
            "lkh_min_generation_gap": 1,
            "max_generations": 5
        }
        st = setup_state(self.instance, cfg_overrides)

        def crashing_solver(*args, **kwargs):
            raise RuntimeError("Simulated LKH crash")

        with patch.object(core, "_solve_elkai_internal", side_effect=crashing_solver):
            # Run generation should complete safely without raising RuntimeError
            rec = run_generation(st)
            self.assertIsNotNone(rec)
            self.assertTrue(valid_tour(st["global_best_tour"], self.instance.n))

    def test_telemetry_in_app_payload(self):
        """FastAPI payload must expose LKH telemetry and status metrics."""
        st = setup_state(self.instance)
        st["lkh_status"] = "idle"
        st["lkh_calls_total"] = 1
        st["lkh_last_duration"] = 0.015

        data = app_module.payload(st, compact=False)
        self.assertIn("lkh", data)
        self.assertEqual(data["lkh"]["calls_total"], 1)
        self.assertEqual(data["lkh"]["status"], "idle")
        self.assertAlmostEqual(data["lkh"]["last_duration"], 0.015)
        self.assertIn("lkh_status", data["status"])
        self.assertIn("lkh_calls_total", data["status"])


if __name__ == "__main__":
    unittest.main()

