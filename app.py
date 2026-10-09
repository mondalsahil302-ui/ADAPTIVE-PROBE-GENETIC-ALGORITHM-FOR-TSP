"""
Root ASGI Entrypoint for Render, Railway, Heroku, and Cloud Deployments.
Forwards FastAPI ASGI instance from backend.app.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# Add app root to sys.path
APP_DIR = Path(__file__).resolve().parent / "adaptive_evolution_research_console_v4" / "adaptive_evolution_research_editor_v4"
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from backend.app import app  # noqa: F401

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app:app", host="0.0.0.0", port=port)

