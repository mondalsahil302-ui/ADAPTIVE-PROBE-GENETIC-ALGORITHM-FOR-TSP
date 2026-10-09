"""
Bounded Selective LKH Optimization with Timeout, Canonical Tour Caching,
and Defensive Permutation Validation.
"""
from __future__ import annotations

import concurrent.futures
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Optional

import numpy as np

from backend.optimizer.models import build_knn_lists, tour_length, valid_tour

try:
    import elkai
    HAS_ELKAI = True
    _elkai = getattr(elkai, "_elkai", None)
except Exception:
    HAS_ELKAI = False
    _elkai = None

# Symmetric tour cache
LKH_CACHE: dict[tuple, tuple[list[int], float]] = {}


def canonicalize_tour(tour: list[int]) -> tuple[int, ...]:
    """
    Normalizes a TSP tour under cyclic rotations and reversals to an invariant representation.
    Rotates the tour so the minimum city index is at position 0, then selects
    the orientation where city 1 is smaller than city -1.
    """
    n = len(tour)
    if n <= 1:
        return tuple(tour)
    min_idx = tour.index(min(tour))
    rotated = tour[min_idx:] + tour[:min_idx]
    if rotated[1] > rotated[-1]:
        rotated = [rotated[0]] + rotated[1:][::-1]
    return tuple(rotated)


def _solve_elkai_internal(matrix: list[list[int]], runs: int, max_trials: int, timeout_sec: float) -> Optional[list[int]]:
    """Invokes elkai / _elkai C solver with strict parameters."""
    if not HAS_ELKAI:
        return None
    try:
        dim = len(matrix)
        params = {
            "PROBLEM_FILE": "matrix",
            "MAX_TRIALS": max(10, int(max_trials)),
            "RUNS": max(1, int(runs)),
            "TIME_LIMIT": max(1, int(timeout_sec)),
            "SPECIAL": "",
        }
        res = _elkai.solve_problem(params, matrix)
        if res and len(res) == dim:
            return list(map(int, res))
    except Exception:
        try:
            res = elkai.solve_int_matrix(matrix, runs=runs)
            if res and len(res) == len(matrix):
                return list(map(int, res))
        except Exception:
            return None
    return None


def _solve_lkh_external(tour: list[int], dist: np.ndarray, exe_path: str, runs: int, max_trials: int, timeout_sec: float) -> Optional[list[int]]:
    """Fallback runner for external official LKH binary."""
    n = len(tour)
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        tsp_file = tmp / "problem.tsp"
        par_file = tmp / "problem.par"
        tour_file = tmp / "solution.tour"

        with open(tsp_file, "w", encoding="utf-8") as f:
            f.write(f"NAME: problem\nTYPE: TSP\nDIMENSION: {n}\nEDGE_WEIGHT_TYPE: EXPLICIT\nEDGE_WEIGHT_FORMAT: FULL_MATRIX\nEDGE_WEIGHT_SECTION\n")
            for i in range(n):
                row = [str(int(round(dist[i, j]))) for j in range(n)]
                f.write(" ".join(row) + "\n")
            f.write("EOF\n")

        with open(par_file, "w", encoding="utf-8") as f:
            f.write(f"PROBLEM_FILE = {tsp_file}\nOUTPUT_TOUR_FILE = {tour_file}\nRUNS = {runs}\nMAX_TRIALS = {max_trials}\nTIME_LIMIT = {int(timeout_sec)}\n")

        proc = subprocess.run([exe_path, str(par_file)], stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout_sec + 2.0)
        if proc.returncode == 0 and tour_file.exists():
            lines = tour_file.read_text().splitlines()
            read_nodes = False
            result_tour = []
            for line in lines:
                line = line.strip()
                if line == "TOUR_SECTION":
                    read_nodes = True
                    continue
                if line in ("-1", "EOF"):
                    break
                if read_nodes and line:
                    result_tour.append(int(line) - 1)
            if valid_tour(result_tour, n):
                return result_tour
    return None


def solve_lkh_bounded(
    tour: list[int],
    dist: np.ndarray,
    runs: int = 1,
    max_trials: int = 50,
    timeout_sec: float = 5.0,
    instance_name: str = "tsp",
    cache_enabled: bool = True,
    int_matrix: Optional[list[list[int]]] = None,
    lkh_exe: Optional[str] = None
) -> tuple[list[int], float, str]:
    """
    Solves TSP route using bounded LKH with ThreadPoolExecutor timeout,
    symmetric tour caching, and defensive validation.
    """
    from backend.optimizer.heuristics import or_opt, two_opt

    original = tour[:]
    n = len(tour)
    orig_cost = tour_length(original, dist)

    if n < 4:
        return original, orig_cost, "too_small"

    canon = canonicalize_tour(original) if cache_enabled else None
    cache_key = (instance_name, n, canon, runs, max_trials) if cache_enabled else None
    if cache_enabled and cache_key in LKH_CACHE:
        cached_tour, cached_cost = LKH_CACHE[cache_key]
        if cached_cost + 1e-9 < orig_cost:
            return cached_tour[:], cached_cost, "cached"
        return original, orig_cost, "cached"

    result = None
    status = "no_improvement"

    # 1. External LKH executable check
    exe = lkh_exe or os.environ.get("LKH_EXE")
    if exe and os.path.isfile(exe):
        try:
            result = _solve_lkh_external(original, dist, exe, runs, max_trials, timeout_sec)
        except Exception:
            result = None

    # 2. Elkai execution in isolated thread
    if result is None and HAS_ELKAI:
        try:
            matrix = int_matrix if int_matrix is not None else np.rint(dist).astype(int).tolist()
            executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
            try:
                future = executor.submit(_solve_elkai_internal, matrix, runs, max_trials, timeout_sec)
                try:
                    result = future.result(timeout=max(0.01, timeout_sec))
                except concurrent.futures.TimeoutError:
                    status = "timed_out"
                    result = None
                    executor.shutdown(wait=False, cancel_futures=True)
                except Exception as ex:
                    status = f"error: {ex}"
                    result = None
                    executor.shutdown(wait=False, cancel_futures=True)
                else:
                    executor.shutdown(wait=False)
            except Exception as ex:
                status = f"error: {ex}"
                result = None
                executor.shutdown(wait=False, cancel_futures=True)
        except Exception as ex:
            status = f"error: {ex}"
            result = None

    # 3. Fallback local search if LKH is unavailable
    if result is None and status not in ("timed_out",) and not HAS_ELKAI and not exe:
        try:
            k_neighbors = min(15, n - 1)
            nb = build_knn_lists(dist, k_neighbors)
            cand, _ = two_opt(original, dist, nb, max_passes=2)
            cand, _ = or_opt(cand, dist, nb)
            result = cand
            status = "fallback"
        except Exception:
            result = original

    # 4. Validation
    if result is not None and valid_tour(result, n):
        res_cost = tour_length(result, dist)
        if res_cost + 1e-9 < orig_cost:
            if cache_enabled and cache_key:
                LKH_CACHE[cache_key] = (result[:], res_cost)
            return result[:], res_cost, "improved" if status != "fallback" else "fallback_improved"
        else:
            if cache_enabled and cache_key:
                LKH_CACHE[cache_key] = (original[:], orig_cost)
            return original, orig_cost, status

    return original, orig_cost, status


def selective_lkh(
    tour: list[int],
    dist: np.ndarray,
    neighbors=None,
    runs: int = 1,
    max_trials: int = 50,
    timeout_sec: float = 5.0,
    instance_name: str = "tsp",
    cache_enabled: bool = True,
    int_matrix=None,
    lkh_exe=None
) -> tuple[list[int], bool]:
    orig_cost = tour_length(tour, dist)
    polished, polished_cost, _ = solve_lkh_bounded(
        tour=tour, dist=dist, runs=runs, max_trials=max_trials,
        timeout_sec=timeout_sec, instance_name=instance_name,
        cache_enabled=cache_enabled, int_matrix=int_matrix, lkh_exe=lkh_exe
    )
    return polished, (polished_cost + 1e-9 < orig_cost)


def selective_lkh_detailed(
    tour: list[int],
    dist: np.ndarray,
    neighbors=None,
    runs: int = 1,
    max_trials: int = 50,
    timeout_sec: float = 5.0,
    instance_name: str = "tsp",
    cache_enabled: bool = True,
    int_matrix=None,
    lkh_exe=None
) -> tuple[list[int], bool, str, float]:
    t0 = time.perf_counter()
    orig_cost = tour_length(tour, dist)
    polished, polished_cost, status = solve_lkh_bounded(
        tour=tour, dist=dist, runs=runs, max_trials=max_trials,
        timeout_sec=timeout_sec, instance_name=instance_name,
        cache_enabled=cache_enabled, int_matrix=int_matrix, lkh_exe=lkh_exe
    )
    dur = time.perf_counter() - t0
    improved = (polished_cost + 1e-9 < orig_cost)
    return polished, improved, status, dur
