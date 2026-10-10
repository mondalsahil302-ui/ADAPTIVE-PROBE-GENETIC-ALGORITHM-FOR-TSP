"""
Authoritative Production FastAPI Application Entrypoint.
Adaptive Evolution Research Console v4.2.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from backend.api.routes import router as api_router
from backend.config import FRONTEND_DIR, RUNS_DIR
from backend.services.experiment_service import service

app = FastAPI(
    title="Adaptive Evolution Research Console",
    description="Accelerated Adaptive Genetic Algorithm for TSP with Bounded Selective LKH and Crystal Glassmorphism UI",
    version="4.2",
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static file mounts
if FRONTEND_DIR.exists():
    app.mount("/frontend", StaticFiles(directory=FRONTEND_DIR), name="frontend")
    assets_dir = FRONTEND_DIR / "assets"
    if assets_dir.exists():
        app.mount("/assets", StaticFiles(directory=assets_dir), name="assets")

# Include API Router
app.include_router(api_router)


# Frontend root and asset routes
@app.get("/", tags=["Frontend"])
def home():
    index_file = FRONTEND_DIR / "index.html"
    return FileResponse(
        index_file,
        media_type="text/html",
        headers={"Cache-Control": "no-cache, no-store, must-revalidate"},
    )


@app.get("/style.css", tags=["Frontend"])
@app.get("/frontend/style.css", tags=["Frontend"])
@app.get("/assets/css/style.css", tags=["Frontend"])
def get_style():
    style_file = FRONTEND_DIR / "assets" / "css" / "style.css"
    if not style_file.exists():
        style_file = FRONTEND_DIR / "style.css"
    return FileResponse(
        style_file,
        media_type="text/css",
        headers={"Cache-Control": "no-cache, no-store, must-revalidate"},
    )


@app.get("/app.js", tags=["Frontend"])
@app.get("/frontend/app.js", tags=["Frontend"])
@app.get("/assets/js/app.js", tags=["Frontend"])
def get_script():
    js_file = FRONTEND_DIR / "assets" / "js" / "app.js"
    if not js_file.exists():
        js_file = FRONTEND_DIR / "app.js"
    return FileResponse(
        js_file,
        media_type="application/javascript",
        headers={"Cache-Control": "no-cache, no-store, must-revalidate"},
    )


# Compatibility aliases for legacy tests and external callers
STATE = service.state
RUN = service.run
LOCK = service.lock
payload = service.payload
get_state = service.get_state
calculate_eta = service.calculate_eta
coords_for = service.coords_for

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    host = os.environ.get("HOST", "0.0.0.0")
    uvicorn.run("backend.app:app", host=host, port=port, reload=False)

