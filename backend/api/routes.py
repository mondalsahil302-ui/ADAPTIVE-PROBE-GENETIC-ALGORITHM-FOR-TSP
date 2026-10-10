"""
FastAPI Router with all research console API endpoints.
"""
from __future__ import annotations

import shutil
import tempfile
import threading
import time
import traceback
import uuid
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse

from backend.config import BASE_DIR, RUNS_DIR
from backend.optimizer.core import run_generation, setup_state
from backend.optimizer.lkh import HAS_ELKAI
from backend.optimizer.models import load_any_instance, tsplib95
from backend.reports.pdf_generator import generate_pdf_report
from backend.schemas.models import HealthResponse, RunConfig
from backend.services.experiment_service import service
from backend.storage.file_store import create_run_zip, list_saved_runs, persist_run_artifacts

router = APIRouter(prefix="/api", tags=["API"])


@router.get("/health", response_model=HealthResponse)
def health():
    return {
        "ok": True,
        "elkai": HAS_ELKAI,
        "tsplib95": tsplib95 is not None,
        "version": "4.2",
    }


@router.get("/state")
def get_state_payload():
    return service.payload(service.get_state(), compact=False)


@router.get("/live")
def get_live_payload():
    return service.payload(service.get_state(), compact=True)


@router.get("/generation/{generation}")
def get_generation(generation: int):
    st = service.get_state()
    with service.lock:
        if generation not in st["populations"]:
            raise HTTPException(404, f"Generation {generation} not available yet.")
        pop = st["populations"][generation]
        record = st["records"][generation] if generation < len(st["records"]) else {}
        route = st["route_history"].get(generation, st["global_best_tour"])
        inst = st["instance"]
        c = service.coords_for(inst)
        route_coords = [{"city": int(city), "x": float(c[city, 0]), "y": float(c[city, 1])} for city in route]
        return {
            "generation": generation,
            "record": record,
            "population": pop,
            "population_count": len(pop),
            "route": route,
            "route_coords": route_coords,
        }


@router.post("/upload")
async def upload_dataset(file: UploadFile = File(...)):
    data = await file.read()
    name = file.filename or "dataset"
    try:
        if name.lower().endswith(".csv"):
            inst = service.load_csv(data, name)
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

        with service.lock:
            service.state["st"] = setup_state(inst)
            service.state["filename"] = name
            service.state["run_id"] = new_run_id
            service.run.update(active=False, started=None, finished=None, error=None, mode=None, run_id=new_run_id)

        return service.payload(service.state["st"], compact=False)
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(400, f"Dataset upload failed: {e}")


@router.post("/reset")
def reset_experiment():
    st = service.get_state()
    with service.lock:
        if service.run["active"]:
            raise HTTPException(409, "Please stop the active experiment before resetting.")
        new_run_id = f"{st['instance'].name.lower()}_{int(time.time())}_{uuid.uuid4().hex[:6]}"
        service.state["st"] = setup_state(st["instance"])
        service.state["run_id"] = new_run_id
        service.run.update(active=False, started=None, finished=None, error=None, mode=None, run_id=new_run_id)
        return service.payload(service.state["st"], compact=False)


@router.post("/run-all/start")
@router.post("/run/start")
def start_experiment(req: RunConfig = None):
    req = req or RunConfig()
    st = service.get_state()
    with service.lock:
        if service.run["active"]:
            raise HTTPException(409, "An experiment is already actively running.")

        if req.values:
            for k, v in req.values.items():
                if hasattr(st["config"], k):
                    val = bool(v) if isinstance(getattr(st["config"], k), bool) else type(getattr(st["config"], k))(v)
                    setattr(st["config"], k, val)

        if st.get("is_stopped", False) or st.get("converged", False):
            inst = st["instance"]
            new_run_id = f"{inst.name.lower()}_{int(time.time())}_{uuid.uuid4().hex[:6]}"
            service.state["st"] = setup_state(inst, st["config"])
            service.state["run_id"] = new_run_id
            st = service.state["st"]

        service.run.update(active=True, started=time.time(), finished=None, error=None, mode="continuous", run_id=service.state["run_id"])
        st["is_stopped"] = False
        st["stop_requested"] = False
        st["state"] = "running"
        st["stop_reason"] = "In progress"

    worker = threading.Thread(target=service.run_background, daemon=True)
    worker.start()
    return {"ok": True, "message": "Experiment started in continuous mode."}


@router.post("/run/step")
def step_experiment(req: RunConfig = None):
    req = req or RunConfig()
    st = service.get_state()
    with service.lock:
        if service.run["active"]:
            raise HTTPException(409, "Cannot step manually while continuous run is active.")

        if req.values:
            for k, v in req.values.items():
                if hasattr(st["config"], k):
                    val = bool(v) if isinstance(getattr(st["config"], k), bool) else type(getattr(st["config"], k))(v)
                    setattr(st["config"], k, val)

        if st.get("is_stopped", False) or st.get("converged", False):
            raise HTTPException(400, f"Experiment has already completed: {st.get('stop_reason')}")

        service.run.update(mode="single_step")
        rec = run_generation(st)
        return {"ok": True, "record": rec, "payload": service.payload(st, compact=True)}


@router.post("/run-all/stop")
@router.post("/run/stop")
def stop_experiment():
    with service.lock:
        st = service.state.get("st")
        if st:
            st["stop_requested"] = True
            st["is_stopped"] = True
            st["state"] = "stopped_by_user"
            st["stop_reason"] = "Stopped by user request"
        service.run["active"] = False
        service.run["finished"] = time.time()
        return {"ok": True, "message": "Stop requested."}


@router.get("/report/status")
def get_report_status(run_id: Optional[str] = Query(None)):
    target_run = run_id or service.state.get("run_id")
    if not target_run:
        return {"available": False, "run_id": None, "reason": "No run_id specified"}
    
    pdf_path = RUNS_DIR / target_run / "report.pdf"
    if pdf_path.exists() and pdf_path.stat().st_size > 0:
        return {"available": True, "run_id": target_run, "size_bytes": pdf_path.stat().st_size}
    
    hist_pdf = BASE_DIR / "research" / "historical_runs" / target_run / "report.pdf"
    if hist_pdf.exists() and hist_pdf.stat().st_size > 0:
        return {"available": True, "run_id": target_run, "size_bytes": hist_pdf.stat().st_size}
    
    st = service.state.get("st")
    if st and service.state.get("run_id") == target_run:
        return {"available": True, "can_generate": True, "run_id": target_run}
    
    return {"available": False, "run_id": target_run}


@router.post("/report/generate")
def generate_report_endpoint(run_id: Optional[str] = Query(None)):
    target_run = run_id or service.state.get("run_id")
    if not target_run:
        raise HTTPException(400, "No active or selected experiment.")
    
    run_dir = RUNS_DIR / target_run
    run_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = run_dir / "report.pdf"
    
    st = service.state.get("st")
    if not st:
        raise HTTPException(400, "No experiment state available to generate report.")
    
    generate_pdf_report(st, target_run, pdf_path)
    return {
        "ok": True,
        "run_id": target_run,
        "path": str(pdf_path),
        "size_bytes": pdf_path.stat().st_size,
    }


@router.get("/report/pdf")
@router.get("/pdf")
def download_pdf(run_id: Optional[str] = Query(None)):
    target_run = run_id or service.state.get("run_id")
    if not target_run:
        raise HTTPException(400, "No active or selected experiment.")

    run_dir = RUNS_DIR / target_run
    pdf_path = run_dir / "report.pdf"

    # 1. Existing report in run directory
    if pdf_path.exists() and pdf_path.stat().st_size > 0:
        return FileResponse(
            pdf_path,
            media_type="application/pdf",
            filename=f"APGGS_Report_{target_run}.pdf",
        )

    # 2. Check if preserved in historical_runs
    hist_pdf = BASE_DIR / "research" / "historical_runs" / target_run / "report.pdf"
    if hist_pdf.exists() and hist_pdf.stat().st_size > 0:
        run_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(hist_pdf, pdf_path)
        return FileResponse(
            pdf_path,
            media_type="application/pdf",
            filename=f"APGGS_Report_{target_run}.pdf",
        )

    # 3. Generate on-demand if in-memory state matches
    st = service.state.get("st")
    if st and service.state.get("run_id") == target_run:
        run_dir.mkdir(parents=True, exist_ok=True)
        generate_pdf_report(st, target_run, pdf_path)
        if pdf_path.exists() and pdf_path.stat().st_size > 0:
            return FileResponse(
                pdf_path,
                media_type="application/pdf",
                filename=f"APGGS_Report_{target_run}.pdf",
            )

    # 4. If neither, return informative 404
    raise HTTPException(404, f"No PDF report found for run '{target_run}'.")


@router.get("/artifacts/zip")
def download_artifacts_zip(run_id: Optional[str] = Query(None)):
    target_run = run_id or service.state.get("run_id")
    if not target_run:
        raise HTTPException(400, "No active or selected experiment.")

    run_dir = RUNS_DIR / target_run
    if not run_dir.exists():
        raise HTTPException(404, f"Artifacts directory for run {target_run} not found.")

    st = service.state.get("st")
    if st:
        persist_run_artifacts(st, target_run, RUNS_DIR)

    zip_path = RUNS_DIR / f"{target_run}_artifacts.zip"
    create_run_zip(run_dir, zip_path)
    return FileResponse(
        zip_path,
        media_type="application/zip",
        filename=f"APGGS_Artifacts_{target_run}.zip",
    )


@router.get("/runs")
def list_runs():
    return {"runs": list_saved_runs(RUNS_DIR)}

