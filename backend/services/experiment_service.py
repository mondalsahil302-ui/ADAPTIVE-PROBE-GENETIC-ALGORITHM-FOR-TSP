"""
Experiment Service: State lifecycle, execution worker, background task runner, and telemetry.
"""
from __future__ import annotations

import io
import threading
import time
import traceback
import uuid
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
from fastapi import HTTPException

from backend.config import FIXTURES_DIR, RUNS_DIR
from backend.optimizer.core import run_generation, setup_state
from backend.optimizer.models import TSPInstance, best_known, gap_percent, load_any_instance
from backend.storage.file_store import persist_run_artifacts


class ExperimentService:
    def __init__(self):
        self.state: dict[str, Any] = {"st": None, "filename": None, "run_id": None}
        self.run: dict[str, Any] = {
            "active": False,
            "started": None,
            "finished": None,
            "error": None,
            "mode": None,
            "run_id": None,
        }
        self.lock = threading.RLock()
        self.init_default_dataset()

    def init_default_dataset(self):
        with self.lock:
            if self.state.get("st") is not None:
                return
            demo_file = FIXTURES_DIR / "demo10.tsp"
            if demo_file.exists():
                try:
                    inst = load_any_instance(str(demo_file))
                    self.state["st"] = setup_state(inst)
                    self.state["filename"] = "demo10.tsp"
                    self.state["run_id"] = f"{inst.name.lower()}_{int(time.time())}"
                except Exception as e:
                    print("Failed to auto-load default dataset:", e)

    def get_state(self) -> dict[str, Any]:
        with self.lock:
            if self.state["st"] is None:
                self.init_default_dataset()
            if self.state["st"] is None:
                raise HTTPException(400, "Please upload a TSPLIB or coordinate CSV dataset first.")
            return self.state["st"]

    @staticmethod
    def coords_for(inst: TSPInstance) -> np.ndarray:
        if inst.coords is not None:
            return inst.coords
        d = inst.dist
        n = inst.n
        J = np.eye(n) - np.ones((n, n)) / n
        B = -0.5 * J @ (d ** 2) @ J
        vals, vecs = np.linalg.eigh(B)
        idx = np.argsort(vals)[::-1][:2]
        vals = np.clip(vals[idx], 0, None)
        return vecs[:, idx] * np.sqrt(vals)

    @staticmethod
    def calculate_eta(st: dict[str, Any]) -> tuple[Optional[float], str]:
        records = st.get("records", [])
        if len(records) < 2:
            return None, "Initializing"
        recent_times = [r.get("execution_time", 0.0) for r in records[-5:] if r.get("execution_time", 0.0) > 0]
        if not recent_times:
            return None, "Calculating estimate"
        avg_time = sum(recent_times) / len(recent_times)
        cur_gen = st.get("generation", 0)
        max_gens = st["config"].max_generations
        patience = st["config"].no_improvement_limit
        no_imp = st.get("no_improvement", 0)

        gens_to_max = max(0, max_gens - cur_gen)
        gens_to_stag = max(0, patience - no_imp)
        remaining_gens = min(gens_to_max, gens_to_stag)
        eta_seconds = remaining_gens * avg_time
        if eta_seconds <= 0:
            return 0.0, "Near completion"
        mins = int(eta_seconds // 60)
        secs = int(eta_seconds % 60)
        return eta_seconds, f"~{mins:02d}:{secs:02d} remaining"

    def payload(self, st: dict[str, Any], compact: bool = False) -> dict[str, Any]:
        inst = st["instance"]
        opt = best_known(inst.name) if hasattr(inst, "name") else None
        c = self.coords_for(inst)
        route = st.get("global_best_tour", [])
        patience = int(st["config"].no_improvement_limit)
        no_imp = int(st.get("no_improvement", 0))
        eta_sec, eta_str = self.calculate_eta(st)

        with self.lock:
            records = list(st.get("records", []))
            gens = sorted(st.get("populations", {}).keys())
            latest = records[-1] if records else None

            coords_list = [{"city": int(i), "x": float(c[i, 0]), "y": float(c[i, 1])} for i in range(inst.n)]
            route_list = [{"city": int(city), "x": float(c[city, 0]), "y": float(c[city, 1])} for city in route]

            base_data = {
                "run_id": self.state.get("run_id"),
                "dataset": {
                    "name": inst.name,
                    "cities": inst.n,
                    "weight_type": inst.edge_weight_type,
                    "best_known": opt,
                    "has_coordinates": inst.coords is not None,
                    "filename": self.state["filename"],
                    "coordinates": coords_list,
                },
                "coordinates": coords_list,
                "route": route_list,
                "config": vars(st["config"]),
                "status": {
                    "generation": st.get("generation", 0),
                    "best_cost": float(st.get("global_best_cost", 0.0)),
                    "gap": gap_percent(st.get("global_best_cost", 0.0), opt),
                    "best_generation": st.get("best_generation", 0),
                    "no_improvement": no_imp,
                    "stagnation_patience": patience,
                    "stagnation_display": f"{no_imp} / {patience}",
                    "converged": bool(st.get("converged", False)),
                    "is_stopped": bool(st.get("is_stopped", False)),
                    "state": st.get("state", "idle" if not self.run["active"] else "running"),
                    "stop_reason": st.get("stop_reason") or ("In progress" if self.run["active"] else "Ready"),
                    "population_size": st["config"].population_size,
                    "eta_seconds": eta_sec,
                    "eta_text": eta_str,
                    "lkh_status": st.get("lkh_status", "idle"),
                    "lkh_calls_total": st.get("lkh_calls_total", 0),
                },
                "lkh": {
                    "calls_total": st.get("lkh_calls_total", 0),
                    "max_calls": getattr(st["config"], "lkh_max_calls_per_run", 8),
                    "last_duration": st.get("lkh_last_duration", 0.0),
                    "status": st.get("lkh_status", "idle"),
                    "last_generation": st.get("last_lkh_generation", -1),
                    "enabled": getattr(st["config"], "lkh_enabled", True),
                },
                "run": {
                    "active": self.run["active"],
                    "mode": self.run["mode"],
                    "started": self.run["started"],
                    "finished": self.run["finished"],
                    "error": self.run["error"],
                    "run_id": self.state.get("run_id"),
                },
                "history": [float(x) for x in st.get("history", [])],
                "generations_available": gens,
                "operator_activity": {
                    "plays": dict(st["selector"].plays),
                    "credit": dict(st["selector"].credit),
                },
                "memory": {
                    "trust": float(st["pmg"].trust),
                },
                "latest": latest,
                "diagnostics": st.get("diagnostics", [])[-15:],
                "records": records,
                "population_counts": {str(g): len(st["populations"][g]) for g in gens},
            }

            if compact:
                return base_data

            base_data.update({
                "initial": {
                    "greedy_route": st["pipe"].get("greedy", []),
                    "greedy_cost": float(st["pipe"].get("greedy_cost", 0.0)),
                    "greedy_time": float(st["pipe"].get("greedy_time", 0.0)),
                    "two_opt_local_route": st["pipe"].get("two_opt_local", []),
                    "two_opt_local_cost": float(st["pipe"].get("two_opt_local_cost", 0.0)),
                    "two_opt_local_improved": st["pipe"].get("two_opt_local_improved", False),
                    "two_opt_local_time": float(st["pipe"].get("two_opt_local_time", 0.0)),
                    "two_opt_refined_route": st["pipe"].get("two_opt_refined", []),
                    "two_opt_refined_cost": float(st["pipe"].get("two_opt_refined_cost", 0.0)),
                    "two_opt_refined_improved": st["pipe"].get("two_opt_refined_improved", False),
                    "two_opt_refined_time": float(st["pipe"].get("two_opt_refined_time", 0.0)),
                    "population_time": float(st["pipe"].get("population_time", 0.0)),
                },
                "route": route_list,
                "route_text": " → ".join(map(str, route[:80])) + ("..." if len(route) > 80 else ""),
                "records": records,
                "improvements": list(st.get("improvements", [])),
                "population_counts": {str(g): len(st["populations"][g]) for g in gens},
                "memory": {
                    "trust": float(st["pmg"].trust),
                    "top_edges": [
                        {"edge": [int(a), int(b)], "score": float(v)}
                        for (a, b), v in sorted(st["pmg"].edge_memory.items(), key=lambda z: -z[1])[:25]
                    ],
                },
                "feature_totals": {k: float(v) for k, v in st["feature_totals"].items()},
            })
            return base_data

    def run_background(self):
        try:
            while True:
                with self.lock:
                    st = self.state.get("st")
                    if st is None or st.get("is_stopped", False) or st.get("stop_requested", False):
                        break
                    patience = int(st["config"].no_improvement_limit)
                    if (
                        st["generation"] >= st["config"].max_generations
                        or st.get("converged", False)
                        or st.get("no_improvement", 0) >= patience
                    ):
                        st["is_stopped"] = True
                        break

                run_generation(st)

                with self.lock:
                    if st.get("is_stopped", False):
                        break
                time.sleep(0.001)

        except Exception as e:
            traceback.print_exc()
            with self.lock:
                self.run["error"] = str(e)
                if self.state.get("st"):
                    self.state["st"]["state"] = "failed"
                    self.state["st"]["stop_reason"] = f"Fatal Error: {e}"
                    self.state["st"]["diagnostics"].append(f"ERROR: {e}")
        finally:
            with self.lock:
                self.run["active"] = False
                self.run["finished"] = time.time()
                st = self.state.get("st")
                if st and not st.get("is_stopped", False):
                    st["is_stopped"] = True
                    if not st.get("stop_reason"):
                        st["stop_reason"] = "Finished"
                if st:
                    try:
                        persist_run_artifacts(st, self.state.get("run_id"), RUNS_DIR)
                    except Exception as err:
                        print("Auto-persistence note:", err)

    def load_csv(self, data: bytes, name: str) -> TSPInstance:
        df = pd.read_csv(io.BytesIO(data))
        cols = {str(c).strip().lower(): c for c in df.columns}
        x = cols.get("x") or cols.get("longitude") or cols.get("lon")
        y = cols.get("y") or cols.get("latitude") or cols.get("lat")
        if x is None or y is None:
            nums = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
            if len(nums) < 2:
                raise ValueError("CSV requires x/y coordinates or at least two numeric columns")
            x, y = nums[:2]
        coords = df[[x, y]].astype(float).to_numpy()
        n = len(coords)
        if n < 3:
            raise ValueError("Instance requires at least 3 cities")
        d = coords[:, None, :] - coords[None, :, :]
        dist = np.sqrt((d ** 2).sum(axis=2))
        return TSPInstance(Path(name).stem, n, list(range(n)), dist, coords, "EUCLIDEAN_CSV")


# Singleton instance
service = ExperimentService()
