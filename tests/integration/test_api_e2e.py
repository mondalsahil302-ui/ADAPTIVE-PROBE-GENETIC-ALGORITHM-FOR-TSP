"""
End-to-end integration test for the APGGS Research Console v4.2.
Tests complete pipeline: GA execution, 30-gen stagnation / optimum stopping,
PDF report generation, artifact persistence, and FastAPI endpoints.
"""
import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path

# Add backend directory to sys.path
import sys
BASE_DIR = Path(__file__).resolve().parents[2]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))
sys.path.insert(0, str(BASE_DIR / "backend"))

from backend.algorithm_core import load_any_instance, setup_state, run_generation, make_config, best_known
from backend.pdf_report import persist_run_artifacts, create_run_zip, generate_pdf_report
from fastapi.testclient import TestClient
import backend.app as fastapi_app_module


class TestE2EPipeline(unittest.TestCase):

    def setUp(self):
        self.sample_file = BASE_DIR / "tests" / "fixtures" / "demo10.tsp"
        self.assertTrue(self.sample_file.exists(), f"Sample file not found: {self.sample_file}")

    def test_complete_run_and_artifact_generation(self):
        """Runs the GA to completion on demo10.tsp, verifies stopping and generates full artifacts."""
        inst = load_any_instance(str(self.sample_file))
        self.assertEqual(inst.n, 10)

        cfg = make_config(
            population_size=20,
            max_generations=100,
            no_improvement_limit=30,
            stagnation_patience=30,
            k_neighbors=5,
            mutation_rate=0.15,
            lkh_cadence=10
        )
        st = setup_state(inst, cfg)

        # Step until stopped
        max_safety_steps = 150
        steps = 0
        while not st.get("is_stopped", False) and steps < max_safety_steps:
            run_generation(st)
            steps += 1

        self.assertTrue(st["is_stopped"], "GA should have set is_stopped = True")
        self.assertIn(st["state"], ["completed", "stopped_stagnation", "stopped_max_generations"])

        # Check stopping reason
        if st["state"] == "stopped_stagnation":
            self.assertGreaterEqual(st["no_improvement"], 30)
            self.assertIn("30", st["stop_reason"])
        elif st["state"] == "completed":
            self.assertTrue(st["converged"])

        # Test artifact persistence
        run_id = f"test_e2e_run_{int(steps)}"
        runs_dir = BASE_DIR / "runs"
        test_run_dir = runs_dir / run_id

        try:
            artifacts = persist_run_artifacts(st, run_id, runs_base_dir=runs_dir)

            self.assertIn("summary", artifacts)
            self.assertIn("history", artifacts)
            self.assertIn("route", artifacts)
            self.assertIn("report", artifacts)
            self.assertIn("manifest", artifacts)

            self.assertTrue(artifacts["summary"].exists())
            self.assertTrue(artifacts["history"].exists())
            self.assertTrue(artifacts["route"].exists())
            self.assertTrue(artifacts["report"].exists())
            self.assertTrue(artifacts["manifest"].exists())

            # Check that PDF has non-zero size
            pdf_size = artifacts["report"].stat().st_size
            self.assertGreater(pdf_size, 5000, "Generated PDF should be at least 5KB")

            # Check summary json
            with open(artifacts["summary"], "r", encoding="utf-8") as f:
                summary = json.load(f)
            self.assertEqual(summary["run_id"], run_id)
            self.assertIn("results", summary)
            self.assertIn("config", summary)

            # Test ZIP packaging
            zip_path = runs_dir / f"{run_id}_artifacts.zip"
            created_zip = create_run_zip(test_run_dir, zip_path)
            self.assertTrue(created_zip.exists())
            self.assertGreater(created_zip.stat().st_size, 1000)

        finally:
            # Clean up test run directory and zip
            if test_run_dir.exists():
                shutil.rmtree(test_run_dir, ignore_errors=True)
            zip_file = runs_dir / f"{run_id}_artifacts.zip"
            if zip_file.exists():
                zip_file.unlink(missing_ok=True)

    def test_fastapi_endpoints(self):
        """Tests that FastAPI test client serves UI and handles requests properly."""
        client = TestClient(fastapi_app_module.app)

        # Health endpoint
        res = client.get("/api/health")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertTrue(data.get("ok"))

        # Frontend HTML
        res_ui = client.get("/")
        self.assertEqual(res_ui.status_code, 200)
        self.assertIn("Adaptive Evolution Research Console", res_ui.text)

        # Upload endpoint with demo10
        with open(self.sample_file, "rb") as f:
            res_up = client.post("/api/upload", files={"file": ("demo10.tsp", f, "text/plain")})
        self.assertEqual(res_up.status_code, 200)
        up_data = res_up.json()
        self.assertEqual(up_data["dataset"]["name"].lower(), "demo10")

        # State endpoint
        res_st = client.get("/api/state")
        self.assertEqual(res_st.status_code, 200)
        state_data = res_st.json()
        self.assertEqual(state_data["status"]["generation"], 0)
        self.assertEqual(state_data["status"]["stagnation_patience"], 30)

        # Live polling endpoint
        res_live = client.get("/api/live")
        self.assertEqual(res_live.status_code, 200)
        live_data = res_live.json()
        self.assertIn("stagnation_display", live_data["status"])
        self.assertEqual(live_data["status"]["stagnation_display"], "0 / 30")


if __name__ == "__main__":
    unittest.main()
