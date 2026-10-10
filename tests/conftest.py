"""
Pytest configuration and shared test fixtures for APGGS and Selective LKH.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
import pytest
import numpy as np

# Ensure project root is at the front of sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Also add backend for direct imports
BACKEND_DIR = PROJECT_ROOT / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))


@pytest.fixture(scope="session")
def fixtures_dir() -> Path:
    return PROJECT_ROOT / "tests" / "fixtures"


@pytest.fixture(scope="session")
def demo10_path(fixtures_dir: Path) -> Path:
    return fixtures_dir / "demo10.tsp"


@pytest.fixture
def deterministic_instance_10():
    from backend.optimizer.models import TSPInstance
    rng = np.random.RandomState(42)
    coords = rng.rand(10, 2) * 100.0
    d = coords[:, None, :] - coords[None, :, :]
    dist = np.sqrt((d ** 2).sum(axis=2))
    return TSPInstance("test10", 10, list(range(10)), dist, coords, "EUC_2D")

