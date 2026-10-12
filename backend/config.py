"""
Centralized Configuration and Path Management for APGGS and Selective LKH.
"""
from __future__ import annotations

import copy
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

# Base directories
BASE_DIR = Path(__file__).resolve().parent.parent
BACKEND_DIR = BASE_DIR / "backend"
FRONTEND_DIR = BASE_DIR / "frontend"
DATA_DIR = Path(os.environ.get("DATA_DIR", BASE_DIR / "data"))
RUNS_DIR = Path(os.environ.get("RUNS_DIR", DATA_DIR / "runs"))
UPLOADS_DIR = Path(os.environ.get("UPLOADS_DIR", DATA_DIR / "uploads"))
CACHE_DIR = Path(os.environ.get("CACHE_DIR", DATA_DIR / "cache"))
ARTIFACTS_DIR = BASE_DIR / "artifacts"
FIXTURES_DIR = BASE_DIR / "tests" / "fixtures"

# Ensure runtime directories exist
for p in (DATA_DIR, RUNS_DIR, UPLOADS_DIR, CACHE_DIR, ARTIFACTS_DIR):
    p.mkdir(parents=True, exist_ok=True)


@dataclass
class GAConfig:
    # Population & Evolutionary Parameters
    population_size: int = 70
    nn_count: int = 17
    greedy_count: int = 11
    random_count: int = 42
    tournament_size: int = 4
    elite_count: int = 4
    mutation_rate: int | float = 0.15
    swap_fraction: float = 0.02
    segment_crossover_probability: float = 0.65
    offspring_local_search_probability: float = 0.20
    fast_two_opt_passes: int = 1
    k_neighbors: int = 15
    adaptive_elite_count: int = 5
    memory_decay: float = 0.96
    memory_elite_fraction: float = 0.20
    stagnation_window: int = 15
    stagnation_epsilon: float = 1e-4
    diversity_floor: float = 0.05
    seed: int = 42
    max_generations: int = 100
    no_improvement_limit: int = 30
    stagnation_patience: int = 30
    improvement_tolerance: float = 1e-9

    # Bounded Selective LKH Parameters
    lkh_enabled: bool = True
    lkh_cadence: int = 15
    lkh_runs: int = 1
    lkh_individuals: int = 1
    lkh_max_seconds_per_call: float = 5.0
    lkh_min_generation_gap: int = 15   # matches cadence; prevents double-firing
    lkh_trigger_stagnation: int = 9999  # effectively disabled; cadence-only triggering
    lkh_max_calls_per_run: int = 20    # enough for 300 gens ÷ 15 cadence = 20 calls
    lkh_max_trials: int = 50
    lkh_cache_enabled: bool = True

    @property
    def lkh_min_gap(self) -> int:
        return self.lkh_min_generation_gap

    @lkh_min_gap.setter
    def lkh_min_gap(self, val: int):
        self.lkh_min_generation_gap = int(val)

    @property
    def lkh_max_calls(self) -> int:
        return self.lkh_max_calls_per_run

    @lkh_max_calls.setter
    def lkh_max_calls(self, val: int):
        self.lkh_max_calls_per_run = int(val)

    @property
    def lkh_timeout_seconds(self) -> float:
        return self.lkh_max_seconds_per_call

    @lkh_timeout_seconds.setter
    def lkh_timeout_seconds(self, val: float):
        self.lkh_max_seconds_per_call = float(val)


def make_config(overrides: Optional[dict[str, Any] | GAConfig] = None, **kwargs) -> GAConfig:
    """Builds a GAConfig instance with priority given to kwargs/overrides > env vars > defaults."""
    if isinstance(overrides, GAConfig):
        cfg = copy.copy(overrides)
        combined = kwargs
    else:
        cfg = GAConfig()
        combined = dict(overrides or {})
        combined.update(kwargs)

    # 1. Environment variable overrides (intermediate priority over defaults)
    if "LKH_ENABLED" in os.environ:
        cfg.lkh_enabled = os.environ["LKH_ENABLED"].lower() in ("true", "1", "yes")
    if "LKH_CADENCE" in os.environ:
        try:
            cfg.lkh_cadence = int(os.environ["LKH_CADENCE"])
        except ValueError:
            pass
    if "LKH_MAX_SECONDS_PER_CALL" in os.environ:
        try:
            cfg.lkh_max_seconds_per_call = float(os.environ["LKH_MAX_SECONDS_PER_CALL"])
        except ValueError:
            pass
    if "LKH_MIN_GENERATION_GAP" in os.environ:
        try:
            cfg.lkh_min_generation_gap = int(os.environ["LKH_MIN_GENERATION_GAP"])
        except ValueError:
            pass
    if "LKH_TRIGGER_STAGNATION" in os.environ:
        try:
            cfg.lkh_trigger_stagnation = int(os.environ["LKH_TRIGGER_STAGNATION"])
        except ValueError:
            pass
    if "LKH_MAX_CALLS_PER_RUN" in os.environ:
        try:
            cfg.lkh_max_calls_per_run = int(os.environ["LKH_MAX_CALLS_PER_RUN"])
        except ValueError:
            pass
    if "LKH_MAX_TRIALS" in os.environ:
        try:
            cfg.lkh_max_trials = int(os.environ["LKH_MAX_TRIALS"])
        except ValueError:
            pass
    if "LKH_RUNS" in os.environ:
        try:
            cfg.lkh_runs = int(os.environ["LKH_RUNS"])
        except ValueError:
            pass
    if "LKH_CACHE_ENABLED" in os.environ:
        cfg.lkh_cache_enabled = os.environ["LKH_CACHE_ENABLED"].lower() in ("true", "1", "yes")

    # 2. Explicit overrides and kwargs (highest priority)
    if "stagnation_patience" in combined and "no_improvement_limit" not in combined:
        combined["no_improvement_limit"] = int(combined["stagnation_patience"])
    if "lkh_min_gap" in combined and "lkh_min_generation_gap" not in combined:
        combined["lkh_min_generation_gap"] = int(combined["lkh_min_gap"])
    if "lkh_max_calls" in combined and "lkh_max_calls_per_run" not in combined:
        combined["lkh_max_calls_per_run"] = int(combined["lkh_max_calls"])
    if "lkh_timeout_seconds" in combined and "lkh_max_seconds_per_call" not in combined:
        combined["lkh_max_seconds_per_call"] = float(combined["lkh_timeout_seconds"])

    for k, v in combined.items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)

    cfg.no_improvement_limit = int(cfg.no_improvement_limit)
    cfg.stagnation_patience = cfg.no_improvement_limit
    cfg.population_size = max(4, int(cfg.population_size))
    cfg.nn_count = min(int(cfg.nn_count), cfg.population_size)
    cfg.greedy_count = min(int(cfg.greedy_count), max(0, cfg.population_size - cfg.nn_count))
    cfg.random_count = max(0, cfg.population_size - cfg.nn_count - cfg.greedy_count)

    return cfg

