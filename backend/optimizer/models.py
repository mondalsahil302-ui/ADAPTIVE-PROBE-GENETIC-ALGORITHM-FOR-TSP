"""
Domain Models and Core Utilities for TSP Instances and Tours.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

try:
    import tsplib95
except ImportError:
    tsplib95 = None

BEST_KNOWN_OPTIMA: dict[str, float] = {
    "DEMO10": 102.0,
    "BAYS29": 2020.0,
    "BERLIN52": 7542.0,
    "BIER127": 118282.0,
    "A280": 2579.0,
    "PR1002": 259045.0,
    "LIN318": 42029.0,
    "EIL76": 538.0,
    "KROA100": 21282.0,
    "CH130": 6110.0,
    "D657": 48912.0,
    "RAT783": 8806.0,
    "RD400": 15281.0,
    "U574": 36905.0,
}


@dataclass
class TSPInstance:
    name: str
    n: int
    nodes: list[int]
    dist: np.ndarray
    coords: Optional[np.ndarray] = None
    edge_weight_type: str = "EXPLICIT"


@dataclass
class Individual:
    id: int
    tour: list[int]
    cost: float
    fitness: float
    method: str
    generation: int


def best_known(name: str) -> Optional[float]:
    if not name:
        return None
    return BEST_KNOWN_OPTIMA.get(str(name).strip().upper())


def gap_percent(cost: float, opt: Optional[float]) -> Optional[float]:
    if opt is None or opt <= 0:
        return None
    return round(float(((cost - opt) / opt) * 100.0), 3)


def tour_length(tour: list[int], dist: np.ndarray) -> float:
    n = len(tour)
    if n == 0:
        return 0.0
    return float(sum(dist[tour[i], tour[(i + 1) % n]] for i in range(n)))


def valid_tour(tour: list[int], n: int) -> bool:
    return len(tour) == n and len(set(tour)) == n and min(tour) == 0 and max(tour) == n - 1


def fitness_from_cost(cost: float) -> float:
    return 1.0 / (cost + 1e-9)


def position_map(tour: list[int]) -> dict[int, int]:
    return {city: idx for idx, city in enumerate(tour)}


def build_knn_lists(dist: np.ndarray, k: int = 15) -> list[list[int]]:
    n = dist.shape[0]
    effective_k = max(1, min(k, n - 1))
    neighbors = []
    for i in range(n):
        row = dist[i].copy()
        row[i] = np.inf
        ranked = np.argsort(row)[:effective_k].tolist()
        neighbors.append(ranked)
    return neighbors


def edge_set(tour: list[int]) -> set[tuple[int, int]]:
    n = len(tour)
    return {tuple(sorted((tour[i], tour[(i + 1) % n]))) for i in range(n)}


def population_diversity(population: list[Individual], rng: random.Random, sample_pairs: int = 40) -> float:
    if len(population) < 2:
        return 0.0
    total = 0.0
    count = 0
    for _ in range(sample_pairs):
        a, b = rng.sample(population, 2)
        ea = edge_set(a.tour)
        eb = edge_set(b.tour)
        n = len(a.tour)
        diff = len(ea.symmetric_difference(eb)) / (2.0 * n)
        total += diff
        count += 1
    return total / count if count else 0.0


def _plain_tsplib(path: str | Path) -> TSPInstance:
    """Fallback parser for standard TSPLIB files."""
    txt = Path(path).read_text(encoding="utf-8", errors="ignore")
    lines = [x.strip() for x in txt.splitlines()]
    hdr = {}
    sec = None
    coords = {}
    weights = []
    for line in lines:
        if not line:
            continue
        u = line.upper()
        if u.startswith("NODE_COORD_SECTION"):
            sec = "coords"
            continue
        if u.startswith("EDGE_WEIGHT_SECTION"):
            sec = "weights"
            continue
        if u.startswith("EOF"):
            break
        if ":" in line and sec is None:
            k, v = line.split(":", 1)
            hdr[k.strip().upper()] = v.strip()
            continue
        if sec == "coords":
            p = line.split()
            if len(p) >= 3:
                try:
                    coords[int(p[0])] = (float(p[1]), float(p[2]))
                except ValueError:
                    pass
        elif sec == "weights":
            for p in line.split():
                try:
                    weights.append(float(p))
                except ValueError:
                    pass

    n = int(hdr.get("DIMENSION", len(coords)))
    nodes = list(sorted(coords)) if coords else list(range(1, n + 1))
    dist = np.zeros((n, n), dtype=float)
    if coords:
        arr = np.array([coords[node] for node in nodes], dtype=float)
        d = arr[:, None, :] - arr[None, :, :]
        et = str(hdr.get("EDGE_WEIGHT_TYPE", "EUC_2D")).upper()
        if et in {"EUC_2D", "EUC_3D"}:
            dist = np.sqrt((d ** 2).sum(axis=2))
            if et == "EUC_2D":
                dist = np.floor(dist + 0.5)
        else:
            dist = np.sqrt((d ** 2).sum(axis=2))
    elif weights:
        typ = str(hdr.get("EDGE_WEIGHT_FORMAT", "FULL_MATRIX")).upper()
        a = np.array(weights, float)
        if typ == "FULL_MATRIX" and len(a) >= n * n:
            dist = a[: n * n].reshape(n, n)
        else:
            dist = np.zeros((n, n), float)
            k = 0
            if typ == "LOWER_DIAG_ROW":
                for i in range(n):
                    for j in range(i + 1):
                        dist[i, j] = dist[j, i] = a[k]
                        k += 1
            elif typ == "UPPER_ROW":
                for i in range(n - 1):
                    for j in range(i + 1, n):
                        dist[i, j] = dist[j, i] = a[k]
                        k += 1
            else:
                raise ValueError("Unsupported EDGE_WEIGHT_FORMAT in fallback parser")

    return TSPInstance(
        hdr.get("NAME", Path(path).stem),
        n,
        nodes,
        dist,
        np.array([coords[x] for x in nodes]) if coords else None,
        str(hdr.get("EDGE_WEIGHT_TYPE", "UNKNOWN")),
    )


def load_tsplib_instance(path: str | Path) -> TSPInstance:
    if tsplib95 is None:
        return _plain_tsplib(path)
    problem = tsplib95.load(str(path))
    nodes = list(problem.get_nodes())
    n = len(nodes)
    node_to_idx = {node: idx for idx, node in enumerate(nodes)}
    coords = None
    if getattr(problem, "node_coords", None):
        coords = np.array([problem.node_coords[node] for node in nodes], dtype=float)

    dist = np.zeros((n, n), dtype=float)
    graph = problem.get_graph()
    for u in nodes:
        i = node_to_idx[u]
        for v in nodes:
            j = node_to_idx[v]
            if i != j:
                weight = graph[u][v].get("weight")
                if weight is None:
                    weight = problem.get_weight(u, v)
                dist[i, j] = float(weight)

    return TSPInstance(
        name=getattr(problem, "name", Path(path).stem),
        n=n,
        nodes=nodes,
        dist=dist,
        coords=coords,
        edge_weight_type=getattr(problem, "edge_weight_type", "EXPLICIT"),
    )


def load_any_instance(path: str | Path) -> TSPInstance:
    if tsplib95 is not None:
        try:
            return load_tsplib_instance(path)
        except Exception:
            pass
    return _plain_tsplib(path)

