"""
Experiment Output Persistence, File Storage, and ZIP Packaging.
"""
from __future__ import annotations

import json
import time
import zipfile
from pathlib import Path
from typing import Any, Optional

from backend.config import RUNS_DIR
from backend.optimizer.models import best_known, gap_percent


def persist_run_artifacts(
    st: dict[str, Any],
    run_id: Optional[str] = None,
    runs_base_dir: Optional[Path] = None
) -> dict[str, Path]:
    """
    Saves complete experiment outputs to runs/<run_id>/ directory:
    - summary.json
    - generation_history.jsonl
    - best_route.csv
    - report.pdf (if generator available)
    - manifest.json
    """
    if not run_id:
        run_id = f"run_{int(time.time())}"

    base = Path(runs_base_dir) if runs_base_dir else RUNS_DIR
    runs_dir = base / run_id
    runs_dir.mkdir(parents=True, exist_ok=True)

    artifacts: dict[str, Path] = {}
    inst = st["instance"]
    cfg = st["config"]
    records = st.get("records", [])

    optimum = best_known(inst.name) if hasattr(inst, "name") else None
    final_cost = float(st["global_best_cost"])

    # 1. summary.json
    summary = {
        "run_id": run_id,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "dataset": {
            "name": inst.name,
            "cities": inst.n,
            "edge_weight_type": inst.edge_weight_type,
            "best_known": optimum,
        },
        "config": vars(cfg),
        "results": {
            "best_cost": final_cost,
            "gap_percent": gap_percent(final_cost, optimum),
            "best_generation": st.get("best_generation", 0),
            "completed_generations": st.get("generation", 0),
            "no_improvement_streak": st.get("no_improvement", 0),
            "stagnation_patience": cfg.no_improvement_limit,
            "converged": bool(st.get("converged", False)),
            "is_stopped": bool(st.get("is_stopped", False)),
            "state": st.get("state", "completed"),
            "stop_reason": st.get("stop_reason", ""),
            "total_execution_time": sum(r.get("execution_time", 0.0) for r in records),
        },
        "diagnostics": st.get("diagnostics", []),
    }
    summary_path = runs_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    artifacts["summary"] = summary_path

    # 2. generation_history.jsonl
    history_path = runs_dir / "generation_history.jsonl"
    with open(history_path, "w", encoding="utf-8") as f:
        for r in records:
            clean_r = {k: v for k, v in r.items() if k != "route"}
            f.write(json.dumps(clean_r) + "\n")
    artifacts["history"] = history_path

    # 3. best_route.csv
    route_path = runs_dir / "best_route.csv"
    best_tour = st.get("global_best_tour", [])
    coords = inst.coords
    with open(route_path, "w", encoding="utf-8") as f:
        if coords is not None:
            f.write("order,city_index,x,y\n")
            for order, city in enumerate(best_tour):
                f.write(f"{order},{city},{coords[city, 0]:.6f},{coords[city, 1]:.6f}\n")
        else:
            f.write("order,city_index\n")
            for order, city in enumerate(best_tour):
                f.write(f"{order},{city}\n")
    artifacts["route"] = route_path

    # 4. report.pdf (optional invocation)
    try:
        from backend.reports.pdf_generator import generate_pdf_report
        pdf_path = runs_dir / "report.pdf"
        generate_pdf_report(st, run_id, pdf_path)
        artifacts["report"] = pdf_path
    except Exception as exc:
        print(f"Notice: PDF generation in persist_run_artifacts: {exc}")

    # 5. manifest.json
    manifest = {
        "run_id": run_id,
        "created_at": time.time(),
        "files": {name: str(p.relative_to(runs_dir)) for name, p in artifacts.items()},
    }
    manifest_path = runs_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    artifacts["manifest"] = manifest_path

    return artifacts


def create_run_zip(run_dir: Path, zip_path: Path) -> Path:
    """Creates a zip archive containing all files in run_dir."""
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in run_dir.iterdir():
            if f.is_file() and not f.name.endswith(".zip"):
                zf.write(f, arcname=f.name)
    return zip_path


def list_saved_runs(runs_base_dir: Optional[Path] = None) -> list[dict[str, Any]]:
    base = Path(runs_base_dir) if runs_base_dir else RUNS_DIR
    runs = []
    if base.exists():
        for d in sorted(base.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
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
    return runs
