from __future__ import annotations

import io
import json
import os
import tempfile
import time
import threading
import traceback
import uuid
from pathlib import Path
from typing import Any, Optional

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

try:
    from .algorithm_core import *
    from .pdf_report import generate_pdf_report, persist_run_artifacts, create_run_zip
except (ImportError, ValueError):
    from algorithm_core import *
    from pdf_report import generate_pdf_report, persist_run_artifacts, create_run_zip

ROOT = Path(__file__).resolve().parents[1]
FRONTEND = ROOT / "frontend"
RUNS_DIR = ROOT / "runs"
RUNS_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="Adaptive Evolution Research Console", version="4.2")
app.mount("/frontend", StaticFiles(directory=FRONTEND), name="frontend")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

STATE: dict[str, Any] = {"st": None, "filename": None, "run_id": None}
RUN: dict[str, Any] = {
    "active": False,
    "started": None,
    "finished": None,
    "error": None,
    "mode": None,
    "run_id": None,
}
LOCK = threading.RLock()


class RunConfig(BaseModel):
    values: dict[str, Any] = {}


def load_csv(data: bytes, name: str) -> TSPInstance:
    import pandas as pd
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


def init_default_dataset():
    with LOCK:
        if STATE.get("st") is not None:
            return
        demo_file = ROOT / "sample_data" / "demo10.tsp"
        if demo_file.exists():
            try:
                inst = load_any_instance(str(demo_file))
                STATE["st"] = setup_state(inst)
                STATE["filename"] = "demo10.tsp"
                STATE["run_id"] = f"{inst.name.lower()}_{int(time.time())}"
            except Exception as e:
                print("Failed to auto-load default dataset:", e)


# Auto-load default dataset on startup if available
init_default_dataset()


def get_state() -> dict[str, Any]:
    with LOCK:
        if STATE["st"] is None:
            init_default_dataset()
        if STATE["st"] is None:
            raise HTTPException(400, "Please upload a TSPLIB or coordinate CSV dataset first.")
        return STATE["st"]


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


def calculate_eta(st: dict[str, Any]) -> tuple[Optional[float], str]:
    """Calculates defensible rolling runtime estimate from recent completed generations."""
    records = st.get("records", [])
    if len(records) < 3 or not RUN["active"]:
        return None, "Calculating estimate"

    recent_times = [r.get("execution_time", 0.0) for r in records[-5:] if r.get("execution_time", 0.0) > 0]
    if not recent_times:
        return None, "Calculating estimate"

    avg_time = sum(recent_times) / len(recent_times)
    cur_gen = st.get("generation", 0)
    max_gens = st["config"].max_generations
    patience = st["config"].no_improvement_limit
    no_imp = st.get("no_improvement", 0)

    # Estimate based on which threshold is closer
    gens_to_max = max(0, max_gens - cur_gen)
    gens_to_stag = max(0, patience - no_imp)
    remaining_gens = min(gens_to_max, gens_to_stag)

    eta_seconds = remaining_gens * avg_time
    if eta_seconds <= 0:
        return 0.0, "Near completion"

    mins = int(eta_seconds // 60)
    secs = int(eta_seconds % 60)
    return eta_seconds, f"~{mins:02d}:{secs:02d} remaining"


def payload(st: dict[str, Any], compact: bool = False) -> dict[str, Any]:
    inst = st["instance"]
    opt = best_known(inst.name) if hasattr(inst, "name") else None
    c = coords_for(inst)
    route = st.get("global_best_tour", [])
    patience = int(st["config"].no_improvement_limit)
    no_imp = int(st.get("no_improvement", 0))

    eta_sec, eta_str = calculate_eta(st)

    with LOCK:
        records = list(st.get("records", []))
        gens = sorted(st.get("populations", {}).keys())
        latest = records[-1] if records else None

        coords_list = [{"city": int(i), "x": float(c[i, 0]), "y": float(c[i, 1])} for i in range(inst.n)]
        route_list = [{"city": int(city), "x": float(c[city, 0]), "y": float(c[city, 1])} for city in route]

        base_data = {
            "run_id": STATE.get("run_id"),
            "dataset": {
                "name": inst.name,
                "cities": inst.n,
                "weight_type": inst.edge_weight_type,
                "best_known": opt,
                "has_coordinates": inst.coords is not None,
                "filename": STATE["filename"],
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
                "state": st.get("state", "idle" if not RUN["active"] else "running"),
                "stop_reason": st.get("stop_reason") or ("In progress" if RUN["active"] else "Ready"),
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
                "active": RUN["active"],
                "mode": RUN["mode"],
                "started": RUN["started"],
                "finished": RUN["finished"],
                "error": RUN["error"],
                "run_id": STATE.get("run_id"),
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

        # If compact live poll, return lightweight response
        if compact:
            return base_data

        # Full payload includes coordinates, records, initial pipeline, etc.
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


def run_background():
    """Background worker executing generations up to stagnation or max_generations."""
    try:
        while True:
            with LOCK:
                st = STATE.get("st")
                if st is None:
                    break
                if st.get("is_stopped", False) or st.get("stop_requested", False):
                    break
                patience = int(st["config"].no_improvement_limit)
                if (
                    st["generation"] >= st["config"].max_generations
                    or st.get("converged", False)
                    or st.get("no_improvement", 0) >= patience
                ):
                    st["is_stopped"] = True
                    break

            rec = run_generation(st)

            with LOCK:
                if st.get("is_stopped", False):
                    break

            # Micro-yield ensures FastAPI server remains fully responsive
            time.sleep(0.001)

    except Exception as e:
        traceback.print_exc()
        with LOCK:
            RUN["error"] = str(e)
            if STATE.get("st"):
                STATE["st"]["state"] = "failed"
                STATE["st"]["stop_reason"] = f"Fatal Error: {e}"
                STATE["st"]["diagnostics"].append(f"ERROR: {e}")
    finally:
        with LOCK:
            RUN["active"] = False
            RUN["finished"] = time.time()
            st = STATE.get("st")
            if st and not st.get("is_stopped", False):
                st["is_stopped"] = True
                if not st.get("stop_reason"):
                    st["stop_reason"] = "Finished"

            # Automatically persist run artifacts & PDF report on completion
            if st:
                try:
                    persist_run_artifacts(st, STATE.get("run_id"))
                except Exception as pe:
                    print("Error persisting run artifacts:", pe)


@app.get("/")
def home():
    return FileResponse(
        FRONTEND / "index.html",
        headers={"Cache-Control": "no-cache, no-store, must-revalidate"}
    )


@app.get("/style.css")
@app.get("/frontend/style.css")
def get_style():
    return FileResponse(
        FRONTEND / "style.css",
        media_type="text/css",
        headers={"Cache-Control": "no-cache, no-store, must-revalidate"}
    )


@app.get("/app.js")
@app.get("/frontend/app.js")
def get_script():
    return FileResponse(
        FRONTEND / "app.js",
        media_type="application/javascript",
        headers={"Cache-Control": "no-cache, no-store, must-revalidate"}
    )


@app.get("/api/health")
def health():
    return {"ok": True, "elkai": HAS_ELKAI, "tsplib95": tsplib95 is not None, "version": "4.2"}


@app.get("/api/state")
def state():
    return payload(get_state(), compact=False)


@app.get("/api/live")
def live():
    """High-frequency lightweight polling endpoint."""
    return payload(get_state(), compact=True)


@app.get("/api/generation/{generation}")
def get_generation(generation: int):
    st = get_state()
    with LOCK:
        if generation not in st["populations"]:
            raise HTTPException(404, f"Generation {generation} not available yet.")
        pop = st["populations"][generation]
        record = st["records"][generation] if generation < len(st["records"]) else {}
        route = st["route_history"].get(generation, st["global_best_tour"])
        inst = st["instance"]
        c = coords_for(inst)
        route_coords = [{"city": int(city), "x": float(c[city, 0]), "y": float(c[city, 1])} for city in route]
        return {
            "generation": generation,
            "record": record,
            "population": pop,
            "population_count": len(pop),
            "route": route,
            "route_coords": route_coords,
        }


@app.post("/api/upload")
async def upload(file: UploadFile = File(...)):
    data = await file.read()
    name = file.filename or "dataset"
    try:
        if name.lower().endswith(".csv"):
            inst = load_csv(data, name)
        elif name.lower().endswith((".tsp", ".atsp")):
            with tempfile.NamedTemporaryFile(suffix=".tsp", delete=False) as f:
                f.write(data)
                p = f.name
            try:
                inst = load_any_instance(p)
            finally:
                Path(p).unlink(missing_ok=True)
        else:
            raise ValueError("Supported formats are .tsp, .atsp, and coordinate .csv files.")

        new_run_id = f"{inst.name.lower()}_{int(time.time())}_{uuid.uuid4().hex[:6]}"

        with LOCK:
            STATE["st"] = setup_state(inst)
            STATE["filename"] = name
            STATE["run_id"] = new_run_id
            RUN.update(active=False, started=None, finished=None, error=None, mode=None, run_id=new_run_id)

        return payload(STATE["st"], compact=False)
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(400, f"Dataset upload failed: {e}")


@app.post("/api/reset")
def reset():
    st = get_state()
    with LOCK:
        if RUN["active"]:
            raise HTTPException(409, "Please stop the active experiment before resetting.")
        new_run_id = f"{st['instance'].name.lower()}_{int(time.time())}_{uuid.uuid4().hex[:6]}"
        STATE["st"] = setup_state(st["instance"], vars(st["config"]))
        STATE["run_id"] = new_run_id
        RUN.update(active=False, started=None, finished=None, error=None, mode=None, run_id=new_run_id)
        return payload(STATE["st"], compact=False)


@app.post("/api/research-config")
def research_config(cfg: RunConfig):
    st = get_state()
    with LOCK:
        if RUN["active"]:
            raise HTTPException(409, "Please stop the active experiment before modifying parameters.")
        STATE["st"] = setup_state(st["instance"], cfg.values)
        return payload(STATE["st"], compact=False)


@app.post("/api/run-generation")
def run_single_gen():
    st = get_state()
    with LOCK:
        if RUN["active"]:
            raise HTTPException(409, "A live experiment is currently running.")
        if st.get("is_stopped", False):
            return {"stopped": True, "message": st.get("stop_reason"), **payload(st, compact=False)}
        rec = run_generation(st)
        return {"record": rec, **payload(st, compact=False)}


@app.post("/api/run-all/start")
def start_run_all():
    with LOCK:
        st = get_state()
        if RUN["active"]:
            return JSONResponse({"started": False, "reason": "already_running", **payload(st, compact=True)}, status_code=409)
        if st.get("is_stopped", False):
            return {"started": False, "reason": "already_stopped", "message": st.get("stop_reason"), **payload(st, compact=True)}

        RUN["active"] = True
        RUN["mode"] = "run_all"
        RUN["started"] = time.time()
        RUN["finished"] = None
        RUN["error"] = None
        RUN["run_id"] = STATE.get("run_id")
        st["stop_requested"] = False
        st["state"] = "running"

        thread = threading.Thread(target=run_background, daemon=True)
        thread.start()
        return {"started": True, **payload(st, compact=True)}


@app.post("/api/run-all/stop")
def stop_run_all():
    """Graceful stop request: completes current generation then terminates cleanly."""
    with LOCK:
        if not RUN["active"]:
            return {"stopped": False, "message": "Experiment is not currently running.", **payload(get_state(), compact=True)}
        st = get_state()
        st["stop_requested"] = True
        RUN["mode"] = "stopping"
        return {"stopped": True, "message": "Stop requested. Experiment will terminate at next generation boundary.", **payload(st, compact=True)}


@app.post("/api/report/generate")
def trigger_report_generation():
    """Generates the comprehensive ReportLab PDF report on demand."""
    st = get_state()
    run_id = STATE.get("run_id") or f"run_{int(time.time())}"
    run_dir = RUNS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = run_dir / "report.pdf"

    try:
        generate_pdf_report(st, run_id, pdf_path)
        return {"generated": True, "path": f"/api/report/pdf?run_id={run_id}", "run_id": run_id}
    except Exception as exc:
        traceback.print_exc()
        raise HTTPException(500, f"PDF report generation failed: {exc}")


@app.get("/api/report/pdf")
def get_report_pdf(run_id: Optional[str] = Query(None)):
    """Serves the generated PDF report for viewing in the side panel or browser."""
    target_run = run_id or STATE.get("run_id")
    if not target_run:
        raise HTTPException(400, "No experiment has been run yet.")

    pdf_path = RUNS_DIR / target_run / "report.pdf"
    if not pdf_path.exists():
        # Generate on the fly if state is available
        st = STATE.get("st")
        if st is not None:
            try:
                generate_pdf_report(st, target_run, pdf_path)
            except Exception as e:
                raise HTTPException(500, f"Report generation failed: {e}")
        else:
            raise HTTPException(404, f"Report for run {target_run} not found.")

    return FileResponse(
        pdf_path,
        media_type="application/pdf",
        filename=f"APGGS_Report_{target_run}.pdf",
    )


@app.get("/api/artifacts/zip")
def download_artifacts_zip(run_id: Optional[str] = Query(None)):
    """Creates and returns a ZIP archive of all experiment artifacts."""
    target_run = run_id or STATE.get("run_id")
    if not target_run:
        raise HTTPException(400, "No active or selected experiment.")

    run_dir = RUNS_DIR / target_run
    if not run_dir.exists():
        raise HTTPException(404, f"Artifacts directory for run {target_run} not found.")

    # Ensure artifacts exist
    st = STATE.get("st")
    if st:
        persist_run_artifacts(st, target_run)

    zip_path = RUNS_DIR / f"{target_run}_artifacts.zip"
    create_run_zip(run_dir, zip_path)
    return FileResponse(
        zip_path,
        media_type="application/zip",
        filename=f"APGGS_Artifacts_{target_run}.zip",
    )


@app.get("/api/runs")
def list_runs():
    """Lists all saved experimental runs."""
    runs = []
    if RUNS_DIR.exists():
        for d in sorted(RUNS_DIR.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
            if d.is_dir():
                summary_file = d / "summary.json"
                info = {"run_id": d.name, "created": d.stat().st_mtime}
                if summary_file.exists():
                    try:
                        with open(summary_file, "r", encoding="utf-8") as f:
                            info["summary"] = json.load(f)
                    except Exception:
                        pass
                runs.append(info)
    return {"runs": runs}
