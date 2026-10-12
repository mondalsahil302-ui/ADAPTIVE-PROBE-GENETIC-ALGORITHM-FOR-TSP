"""
Master Automated Verification Suite for LKH-3 Generation Scheduling,
Execution Timer Measurement, Decoupled Cadence, and Telemetry Integration.

Fulfills all requirements of the Master Prompt:
- Test A: Scheduling verification on a 60-generation simulation
- Test B: Boundary and off-by-one verification
- Test C: Decoupling of cadence from stagnation
- Test D: Timer measurement accuracy (monotonic perf_counter)
- Test E: Safe error and timeout handling (preserving incumbent)
- Test F: Configuration enforcement (disabled & budget cap)
- Test G: Duplicate-execution prevention
- Test H: API telemetry payload and reset verification
- Test I: Configuration precedence (defaults vs env vs explicit kwargs)
"""
from __future__ import annotations

import os
import sys
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

# Setup paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
BACKEND_DIR = PROJECT_ROOT / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from backend.algorithm_core import (
    GAConfig,
    TSPInstance,
    make_config,
    setup_state,
    run_generation,
    LKH_CACHE,
)
from backend.services.experiment_service import service as global_service
from fastapi.testclient import TestClient
import backend.app as app_mod


def create_mock_instance(n: int = 15) -> TSPInstance:
    """Create a deterministic synthetic TSP instance for reproducible tests."""
    rng = np.random.RandomState(42)
    coords = rng.rand(n, 2) * 100.0
    d = coords[:, None, :] - coords[None, :, :]
    dist = np.sqrt((d ** 2).sum(axis=2))
    return TSPInstance(
        name="synthetic_master_15",
        n=n,
        nodes=list(range(n)),
        dist=dist,
        coords=coords,
        edge_weight_type="EUC_2D",
    )


class TestLKHMasterRequirements(unittest.TestCase):

    def setUp(self):
        self.instance = create_mock_instance(15)
        LKH_CACHE.clear()

    def tearDown(self):
        LKH_CACHE.clear()

    def test_a_scheduling_verification_60_generations(self):
        """Test A: LKH runs strictly at generations 15, 30, 45, 60.

        Verifies that LKH does NOT run at generations 1-14, 16-29, 31-44, 46-59,
        and specifically never at generation 5.
        """
        cfg = make_config(
            population_size=20,
            max_generations=60,
            no_improvement_limit=100,  # Prevent early stagnation stop
            lkh_enabled=True,
            lkh_cadence=15,
            lkh_min_gap=15,
            lkh_trigger_stagnation=9999,
            lkh_max_calls=20,
        )
        st = setup_state(self.instance, cfg)

        executed_gens = []

        with patch("backend.optimizer.core.selective_lkh_detailed") as mock_lkh:
            mock_lkh.side_effect = lambda *args, **kwargs: (
                list(args[0] if args else kwargs.get("tour", [])),
                False,
                "no_improvement",
                0.001
            )

            for g in range(1, 61):
                rec = run_generation(st)
                if rec.get("lkh_applied", False) or mock_lkh.call_count > len(executed_gens):
                    executed_gens.append(g)

        # Expected invocation generations: strictly 15, 30, 45, 60
        expected_gens = [15, 30, 45, 60]
        self.assertEqual(executed_gens, expected_gens, f"LKH executed at {executed_gens}, expected {expected_gens}")

        # Explicit check for generation 5
        self.assertNotIn(5, executed_gens, "LKH must NEVER execute at generation 5!")

        # Explicit checks for all non-scheduled generations
        non_scheduled = [g for g in range(1, 61) if g not in expected_gens]
        for ns in non_scheduled:
            self.assertNotIn(ns, executed_gens, f"LKH must not execute at non-cadence generation {ns}")

    def test_b_boundary_and_off_by_one_verification(self):
        """Test B: Strict boundary verification around cadence triggers.

        Specifically verifies generations 14, 15, 16 and 29, 30, 31.
        """
        cfg = make_config(
            population_size=20,
            max_generations=35,
            no_improvement_limit=100,
            lkh_enabled=True,
            lkh_cadence=15,
            lkh_min_gap=15,
            lkh_trigger_stagnation=9999,
            lkh_max_calls=20,
        )
        st = setup_state(self.instance, cfg)

        lkh_calls_by_gen = {}

        with patch("backend.optimizer.core.selective_lkh_detailed") as mock_lkh:
            mock_lkh.side_effect = lambda *args, **kwargs: (
                list(args[0] if args else kwargs.get("tour", [])),
                False,
                "no_improvement",
                0.001
            )

            for g in range(1, 35):
                prev_count = mock_lkh.call_count
                run_generation(st)
                lkh_calls_by_gen[g] = (mock_lkh.call_count > prev_count)

        # Gen 14: False, Gen 15: True, Gen 16: False
        self.assertFalse(lkh_calls_by_gen[14], "LKH must NOT run at generation 14")
        self.assertTrue(lkh_calls_by_gen[15], "LKH MUST run at generation 15")
        self.assertFalse(lkh_calls_by_gen[16], "LKH must NOT run at generation 16")

        # Gen 29: False, Gen 30: True, Gen 31: False
        self.assertFalse(lkh_calls_by_gen[29], "LKH must NOT run at generation 29")
        self.assertTrue(lkh_calls_by_gen[30], "LKH MUST run at generation 30")
        self.assertFalse(lkh_calls_by_gen[31], "LKH must NOT run at generation 31")

    def test_c1_decoupling_cadence_with_continual_improvement(self):
        """Test C1: Run with continual improvement.

        Verify LKH executes every 15 generations even when global best keeps improving.
        """
        cfg = make_config(
            population_size=20,
            max_generations=30,
            no_improvement_limit=100,
            lkh_enabled=True,
            lkh_cadence=15,
            lkh_min_gap=15,
            lkh_trigger_stagnation=9999,
            lkh_max_calls=20,
        )
        st = setup_state(self.instance, cfg)

        executed_gens = []

        with patch("backend.optimizer.core.selective_lkh_detailed") as mock_lkh:
            mock_lkh.side_effect = lambda *args, **kwargs: (
                list(args[0] if args else kwargs.get("tour", [])),
                False,
                "no_improvement",
                0.001
            )

            for g in range(1, 31):
                # Artificially keep improving cost so no_improvement stays 0
                st["global_best_cost"] = 1000.0 - g
                st["no_improvement"] = 0
                run_generation(st)
                if mock_lkh.call_count > len(executed_gens):
                    executed_gens.append(g)

        self.assertEqual(executed_gens, [15, 30], "Continual improvement must not suppress periodic 15-gen LKH")

    def test_c2_decoupling_cadence_with_stagnation(self):
        """Test C2: Run with stagnation.

        Verify LKH does NOT execute prematurely at stagnation 5 or 10,
        and strictly fires only at generation 15.
        """
        cfg = make_config(
            population_size=20,
            max_generations=20,
            no_improvement_limit=30,
            lkh_enabled=True,
            lkh_cadence=15,
            lkh_min_gap=15,
            lkh_trigger_stagnation=9999,
            lkh_max_calls=20,
        )
        st = setup_state(self.instance, cfg)

        executed_gens = []

        with patch("backend.optimizer.core.selective_lkh_detailed") as mock_lkh:
            mock_lkh.side_effect = lambda *args, **kwargs: (
                list(args[0] if args else kwargs.get("tour", [])),
                False,
                "no_improvement",
                0.001
            )

            for g in range(1, 16):
                # Force stagnation accumulation
                st["no_improvement"] = g
                run_generation(st)
                if mock_lkh.call_count > len(executed_gens):
                    executed_gens.append(g)

        self.assertEqual(executed_gens, [15], "Stagnation must NOT trigger LKH before generation 15")

    def test_d_timer_measurement_accuracy(self):
        """Test D: Monotonic execution time measurement accuracy.

        Verifies that actual solver wall-clock execution time is measured with
        perf_counter, stored in state, and distinct from overall experiment time.
        """
        cfg = make_config(
            population_size=20,
            max_generations=15,
            no_improvement_limit=30,
            lkh_enabled=True,
            lkh_cadence=15,
            lkh_min_gap=15,
            lkh_trigger_stagnation=9999,
            lkh_max_calls=20,
        )
        st = setup_state(self.instance, cfg)

        controlled_delay = 0.06

        def mock_sleep_solver(tour, dist, *args, **kwargs):
            time.sleep(controlled_delay)
            return list(tour), 500.0, "no_improvement"

        with patch("backend.optimizer.lkh.solve_lkh_bounded", side_effect=mock_sleep_solver):
            for g in range(1, 16):
                run_generation(st)

        # Check recorded duration
        last_dur = st.get("lkh_last_duration", 0.0)
        total_dur = st.get("lkh_total_duration", 0.0)

        self.assertGreaterEqual(last_dur, 0.05, f"LKH duration ({last_dur}s) should measure actual solver delay")
        self.assertLess(last_dur, 1.0, f"LKH duration ({last_dur}s) should not capture unrelated overhead")
        self.assertAlmostEqual(last_dur, total_dur, places=2)
        self.assertEqual(st.get("lkh_calls_total"), 1)

    def test_e_safe_error_and_timeout_handling(self):
        """Test E: Solver exceptions and timeouts are safely caught.

        Verifies that:
        - Exceptions do not crash the experiment.
        - Elapsed duration is recorded even on failure.
        - Status is set to 'failed' or 'timed_out'.
        - Incumbent best tour is preserved intact.
        """
        cfg = make_config(
            population_size=20,
            max_generations=15,
            no_improvement_limit=30,
            lkh_enabled=True,
            lkh_cadence=15,
            lkh_min_gap=15,
            lkh_trigger_stagnation=9999,
            lkh_max_calls=20,
        )
        st = setup_state(self.instance, cfg)
        incumbent_cost = float(st["global_best_cost"])
        incumbent_route = list(st["global_best_tour"])

        def mock_failing_solver(tour, dist, *args, **kwargs):
            time.sleep(0.02)
            raise RuntimeError("LKH binary crashed or memory fault")

        with patch("backend.optimizer.lkh.solve_lkh_bounded", side_effect=mock_failing_solver):
            for g in range(1, 16):
                run_generation(st)

        # Check state after failure
        self.assertTrue("error" in st.get("lkh_status", "").lower() or st.get("lkh_status") == "failed")
        self.assertGreaterEqual(st.get("lkh_last_duration", 0.0), 0.015)
        self.assertEqual(st.get("lkh_calls_failed"), 1)
        self.assertIn("crashed", st.get("lkh_last_error", ""))

        # Incumbent must remain intact
        self.assertEqual(st["global_best_cost"], incumbent_cost)
        self.assertEqual(list(st["global_best_tour"]), incumbent_route)

    def test_f_configuration_enforcement(self):
        """Test F: Enforce disabled setting and maximum call budget."""
        # Case 1: LKH Disabled
        cfg_disabled = make_config(
            population_size=20,
            max_generations=30,
            no_improvement_limit=50,
            lkh_enabled=False,
            lkh_cadence=15,
        )
        st1 = setup_state(self.instance, cfg_disabled)
        with patch("backend.optimizer.core.selective_lkh_detailed") as mock_lkh:
            for _ in range(30):
                run_generation(st1)
            mock_lkh.assert_not_called()
        self.assertEqual(st1.get("lkh_calls_total", 0), 0)

        # Case 2: LKH Call Budget = 1
        cfg_budget = make_config(
            population_size=20,
            max_generations=35,
            no_improvement_limit=50,
            lkh_enabled=True,
            lkh_cadence=15,
            lkh_min_gap=15,
            lkh_max_calls=1,
        )
        st2 = setup_state(self.instance, cfg_budget)
        with patch("backend.optimizer.core.selective_lkh_detailed") as mock_lkh:
            mock_lkh.side_effect = lambda *args, **kwargs: (
                list(args[0] if args else kwargs.get("tour", [])),
                False,
                "no_improvement",
                0.001
            )
            for _ in range(35):
                run_generation(st2)
            self.assertEqual(mock_lkh.call_count, 1, "LKH should only execute once due to max_calls=1")
        self.assertEqual(st2.get("lkh_calls_total"), 1)
        self.assertIn("budget_exhausted", st2.get("lkh_status"))

    def test_g_duplicate_execution_prevention(self):
        """Test G: Scheduler prevents duplicate solver invocations for same generation."""
        cfg = make_config(
            population_size=20,
            max_generations=15,
            no_improvement_limit=30,
            lkh_enabled=True,
            lkh_cadence=15,
            lkh_min_gap=15,
            lkh_max_calls=20,
        )
        st = setup_state(self.instance, cfg)

        call_records = []
        with patch("backend.optimizer.core.selective_lkh_detailed") as mock_lkh:
            mock_lkh.side_effect = lambda *args, **kwargs: (
                call_records.append(st["generation"] + 1),
                (list(args[0] if args else kwargs.get("tour", [])), False, "no_improvement", 0.001)
            )[1]

            for _ in range(15):
                run_generation(st)

        self.assertEqual(call_records, [15], "Exactly 1 call should happen at generation 15")

    def test_h_api_telemetry_payload_and_reset(self):
        """Test H: ExperimentService and FastAPI endpoints properly serialize LKH telemetry and handle reset."""
        client = TestClient(app_mod.app)

        res = client.get("/api/state")
        self.assertEqual(res.status_code, 200)
        payload = res.json()

        self.assertIn("lkh", payload)
        lkh_data = payload["lkh"]
        self.assertEqual(lkh_data["cadence"], 15)
        self.assertEqual(lkh_data["min_gap"], 15)
        self.assertIn("status", lkh_data)
        self.assertIn("last_duration", lkh_data)
        self.assertIn("total_duration", lkh_data)
        self.assertIn("calls_total", lkh_data)

        # Reset endpoint
        reset_res = client.post("/api/reset")
        self.assertEqual(reset_res.status_code, 200)
        reset_payload = reset_res.json()
        self.assertEqual(reset_payload["lkh"]["calls_total"], 0)
        self.assertEqual(reset_payload["lkh"]["cadence"], 15)

    def test_i_configuration_precedence(self):
        """Test I: Explicit kwargs override environment variables and default configuration."""
        # 1. Default config
        cfg_default = make_config()
        self.assertEqual(cfg_default.lkh_cadence, 15)
        self.assertEqual(cfg_default.lkh_min_generation_gap, 15)
        self.assertEqual(cfg_default.lkh_min_gap, 15)
        self.assertEqual(cfg_default.lkh_trigger_stagnation, 9999)

        # 2. Environment variable override
        with patch.dict(os.environ, {"LKH_CADENCE": "25", "LKH_MIN_GENERATION_GAP": "25"}):
            cfg_env = make_config()
            self.assertEqual(cfg_env.lkh_cadence, 25)
            self.assertEqual(cfg_env.lkh_min_generation_gap, 25)
            self.assertEqual(cfg_env.lkh_min_gap, 25)

            # 3. Explicit kwargs override environment variables
            cfg_override = make_config(lkh_cadence=15, lkh_min_gap=15)
            self.assertEqual(cfg_override.lkh_cadence, 15)
            self.assertEqual(cfg_override.lkh_min_generation_gap, 15)
            self.assertEqual(cfg_override.lkh_min_gap, 15)


if __name__ == "__main__":
    unittest.main()
