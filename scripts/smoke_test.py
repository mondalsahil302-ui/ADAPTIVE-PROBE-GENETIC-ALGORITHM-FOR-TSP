"""
End-to-End Smoke Test for Backend, FastAPI Endpoints, Experiment Execution, and PDF Export.
"""
from __future__ import annotations

import io
import sys
import zipfile
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi.testclient import TestClient
from backend.app import app


def run_smoke_test():
    print("=" * 65)
    print("  Adaptive Evolution Research Console - Smoke Test")
    print("=" * 65)

    client = TestClient(app)

    # 1. Health check
    print("[1/7] Testing GET /api/health...")
    res = client.get("/api/health")
    assert res.status_code == 200, f"Health check failed: {res.status_code}"
    data = res.json()
    assert data.get("ok") is True
    print(f"      OK! Version: {data.get('version')} | elkai: {data.get('elkai')}")

    # 2. Frontend root
    print("[2/7] Testing GET / (HTML Dashboard)...")
    res = client.get("/")
    assert res.status_code == 200
    assert "Adaptive Evolution Research Console" in res.text
    print("      OK! Glassmorphism dashboard rendered successfully.")

    # 3. Static assets
    print("[3/7] Testing static CSS & JS delivery...")
    css_res = client.get("/style.css")
    assert css_res.status_code == 200
    js_res = client.get("/app.js")
    assert js_res.status_code == 200
    print("      OK! /style.css and /app.js served with HTTP 200.")

    # 4. State endpoint
    print("[4/7] Testing GET /api/state...")
    res = client.get("/api/state")
    assert res.status_code == 200
    state = res.json()
    assert "status" in state
    assert "route" in state
    assert "lkh" in state
    print(f"      OK! Current instance: {state['dataset']['name']} ({state['dataset']['cities']} cities)")

    # 4b. Upload berlin52 for active stepping
    print("[4b] Uploading berlin52.tsp to test active stepping...")
    b52_path = PROJECT_ROOT / "tests" / "fixtures" / "berlin52.tsp"
    with open(b52_path, "rb") as f:
        up_res = client.post("/api/upload", files={"file": ("berlin52.tsp", f, "application/octet-stream")})
    assert up_res.status_code == 200, f"Upload failed: {up_res.text}"
    print("      OK! berlin52.tsp uploaded successfully.")

    # 5. Step execution
    print("[5/7] Stepping 3 generations via POST /api/run/step...")
    for step_i in range(1, 4):
        step_res = client.post("/api/run/step", json={"values": {}})
        assert step_res.status_code == 200, f"Step {step_i} failed: {step_res.text}"
        step_data = step_res.json()
        assert step_data.get("ok") is True
        gen = step_data["payload"]["status"]["generation"]
        cost = step_data["payload"]["status"]["best_cost"]
        print(f"      Step {step_i}: Generation {gen} | Best Cost: {cost:,.2f}")

    # 6. PDF Report Generation & Download
    print("[6/7] Testing PDF report generation via GET /api/pdf...")
    pdf_res = client.get("/api/pdf")
    assert pdf_res.status_code == 200, f"PDF generation failed: {pdf_res.status_code}"
    pdf_bytes = pdf_res.content
    assert pdf_bytes.startswith(b"%PDF-"), "Invalid PDF signature"
    print(f"      OK! Generated valid PDF ({len(pdf_bytes):,} bytes).")

    # 7. Artifacts ZIP Download
    print("[7/7] Testing ZIP export via GET /api/artifacts/zip...")
    zip_res = client.get("/api/artifacts/zip")
    assert zip_res.status_code == 200, f"ZIP export failed: {zip_res.status_code}"
    zip_bytes = zip_res.content
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        names = zf.namelist()
        assert "summary.json" in names
        assert "best_route.csv" in names
    print(f"      OK! Valid ZIP with manifest files: {names}")

    print("=" * 65)
    print("  [SUCCESS] All 7 smoke tests passed completely!")
    print("=" * 65)


if __name__ == "__main__":
    run_smoke_test()
