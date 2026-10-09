from __future__ import annotations
import os, random, time, math, copy
import concurrent.futures, subprocess, tempfile, shutil
from dataclasses import dataclass
from typing import Optional
from collections import Counter, defaultdict, deque
from pathlib import Path
import numpy as np
try:
    import tsplib95
except Exception:
    tsplib95 = None
try:
    import elkai
    HAS_ELKAI = True
except Exception:
    HAS_ELKAI = False


TSPLIB_BEST_KNOWN = {
    "a280": 2579,
    "ali535": 202339,
    "att48": 10628,
    "att532": 27686,
    "bayg29": 1610,
    "bays29": 2020,
    "berlin52": 7542,
    "bier127": 118282,
    "brazil58": 25395,
    "brd14051": 469385,
    "brg180": 1950,
    "burma14": 3323,
    "ch130": 6110,
    "ch150": 6528,
    "d198": 15780,
    "d493": 35002,
    "d657": 48912,
    "d1291": 50801,
    "d1655": 62128,
    "d2103": 80450,
    "d15112": 1573084,
    "d18512": 645238,
    "dantzig42": 699,
    "dsj1000": 18660188,
    "eil51": 426,
    "eil76": 538,
    "eil101": 629,
    "fl417": 11861,
    "fl1400": 20127,
    "fl1577": 22249,
    "fl3795": 28772,
    "fnl4461": 182566,
    "fri26": 937,
    "gil262": 2378,
    "gr17": 2085,
    "gr21": 2707,
    "gr24": 1272,
    "gr48": 5046,
    "gr96": 55209,
    "gr120": 6942,
    "gr137": 69853,
    "gr202": 40160,
    "gr229": 134602,
    "gr431": 171414,
    "gr666": 294358,
    "hk48": 11461,
    "kroa100": 21282,
    "krob100": 22141,
    "kroc100": 20749,
    "krod100": 21294,
    "kroe100": 22068,
    "kroa150": 26524,
    "krob150": 26130,
    "kroa200": 29368,
    "krob200": 29437,
    "lin105": 14379,
    "lin318": 42029,
    "linhp318": 41345,
    "nrw1379": 56638,
    "p654": 34643,
    "pa561": 2763,
    "pcb442": 50778,
    "pcb1173": 56892,
    "pcb3038": 137694,
    "pr76": 108159,
    "pr107": 44303,
    "pr124": 59030,
    "pr136": 96772,
    "pr144": 58537,
    "pr152": 73682,
    "pr226": 80369,
    "pr264": 49135,
    "pr299": 48191,
    "pr439": 107217,
    "pr1002": 259045,
    "pr2392": 378032,
    "rat99": 1211,
    "rat195": 2323,
    "rat575": 6773,
    "rat783": 8806,
    "rd100": 7910,
    "rd400": 15281,
    "rl1304": 252948,
    "rl1323": 270199,
    "rl1889": 316536,
    "rl5915": 565530,
    "rl5934": 556045,
    "rl11849": 923288,
    "si175": 21407,
    "si535": 48450,
    "si1032": 92650,
    "st70": 675,
    "swiss42": 1273,
    "ts225": 126643,
    "tsp225": 3916,
    "u159": 42080,
    "u574": 36905,
    "u724": 41910,
    "u1060": 224094,
    "u1432": 152970,
    "u1817": 57201,
    "u2152": 64253,
    "u2319": 234256,
    "ulysses16": 6859,
    "ulysses22": 7013,
    "usa13509": 19982859,
    "vm1084": 239297,
    "vm1748": 336556,
}

@dataclass
class GAConfig:

    # --------------------------------------------------------
    # POPULATION DISTRIBUTION
    # --------------------------------------------------------
    population_size: int = 70

    nn_count: int = 17
    greedy_count: int = 11
    random_count: int = 42

    # --------------------------------------------------------
    # MAIN EVOLUTION
    # --------------------------------------------------------
    max_generations: int = 300

    # Stop if global best does not improve
    # for 30 consecutive generations
    no_improvement_limit: int = 30
    stagnation_patience: int = 30
    improvement_tolerance: float = 1e-9

    # --------------------------------------------------------
    # K-NEAREST NEIGHBORS
    # --------------------------------------------------------
    k_neighbors: int = 15

    # --------------------------------------------------------
    # GENETIC OPERATORS
    # --------------------------------------------------------
    tournament_size: int = 4

    mutation_rate: float = 0.20

    swap_fraction: float = 0.02

    # --------------------------------------------------------
    # CROSSOVER CONTROL
    # --------------------------------------------------------
    segment_crossover_probability: float = 0.40

    # --------------------------------------------------------
    # FAST LOCAL EVOLUTION
    # --------------------------------------------------------
    offspring_local_search_probability: float = 0.35

    fast_two_opt_passes: int = 1

    # --------------------------------------------------------
    # ELITISM
    # --------------------------------------------------------
    elite_count: int = 2

    # --------------------------------------------------------
    # PREDICTIVE MEMORY GRAPH
    # --------------------------------------------------------
    memory_elite_fraction: float = 0.20

    memory_decay: float = 0.92

    # --------------------------------------------------------
    # ADAPTIVE OPTIMIZATION
    # EXACTLY 14 ELITES WILL BE USED
    # --------------------------------------------------------
    adaptive_elite_count: int = 14

    # --------------------------------------------------------
    # DIVERSITY / STAGNATION
    # --------------------------------------------------------
    stagnation_window: int = 15

    stagnation_epsilon: float = 1e-4

    diversity_floor: float = 0.05

    # --------------------------------------------------------
    # SELECTIVE LKH & RESOURCE LIMITS
    # --------------------------------------------------------
    lkh_enabled: bool = True
    lkh_cadence: int = 15
    lkh_runs: int = 1
    lkh_individuals: int = 1
    lkh_max_seconds_per_call: float = 5.0
    lkh_min_generation_gap: int = 10
    lkh_trigger_stagnation: int = 10
    lkh_max_calls_per_run: int = 8
    lkh_max_trials: int = 50
    lkh_cache_enabled: bool = True

    # --------------------------------------------------------
    # RANDOM SEED
    # --------------------------------------------------------
    seed: int = 1

@dataclass
class TSPInstance:
    name: str
    n: int
    nodes: list
    dist: np.ndarray
    coords: Optional[np.ndarray]
    edge_weight_type: str

    def tour_length(self, tour):
        return float(
            sum(
                self.dist[tour[i], tour[(i + 1) % self.n]]
                for i in range(self.n)
            )
        )

def load_tsplib_instance(path):

    problem = tsplib95.load(path)

    nodes = list(problem.get_nodes())
    n = len(nodes)

    index = {node: i for i, node in enumerate(nodes)}

    # -------------------------------------------
    # COMPLETE DISTANCE MATRIX
    # -------------------------------------------

    dist = np.zeros((n, n), dtype=np.float64)

    for a in nodes:
        ia = index[a]

        for b in nodes:

            ib = index[b]

            if a != b:
                dist[ia, ib] = problem.get_weight(a, b)

    # -------------------------------------------
    # COORDINATES
    # -------------------------------------------

    coords = None

    if getattr(problem, "node_coords", None):

        try:
            coords = np.array(
                [problem.node_coords[node] for node in nodes],
                dtype=np.float64
            )

        except Exception:
            coords = None

    name = (
        problem.name
        if problem.name
        else os.path.splitext(os.path.basename(path))[0]
    )

    edge_weight_type = str(
        getattr(problem, "edge_weight_type", "UNKNOWN")
    )

    return TSPInstance(
        name=name,
        n=n,
        nodes=nodes,
        dist=dist,
        coords=coords,
        edge_weight_type=edge_weight_type
    )

def build_knn_lists(dist, k=15):

    n = dist.shape[0]

    k = min(k, n - 1)

    neighbors = []

    for city in range(n):

        row = dist[city].copy()

        row[city] = np.inf

        idx = np.argpartition(row, k)[:k]

        idx = idx[np.argsort(row[idx])]

        neighbors.append(idx.tolist())

    return neighbors

def format_route(tour, one_based=False):

    offset = 1 if one_based else 0

    return "-".join(
        str(city + offset)
        for city in tour
    )

def valid_tour(tour, n):

    return (
        len(tour) == n
        and len(set(tour)) == n
        and set(tour) == set(range(n))
    )

def tour_length(tour, dist):

    n = len(tour)

    return float(
        sum(
            dist[tour[i], tour[(i + 1) % n]]
            for i in range(n)
        )
    )

def fitness_from_cost(cost):

    return 1.0 / max(cost, 1e-12)

def best_known(name):

    return TSPLIB_BEST_KNOWN.get(
        name.lower()
    )

def gap_percent(cost, optimum):

    if optimum is None:
        return None

    return (
        (cost - optimum)
        / optimum
        * 100.0
    )

def position_map(tour):

    pos = [0] * len(tour)

    for i, city in enumerate(tour):
        pos[city] = i

    return pos

def nearest_neighbor_tour(dist, start=0):

    n = len(dist)

    unvisited = set(range(n))

    unvisited.remove(start)

    tour = [start]

    current = start

    while unvisited:

        nxt = min(
            unvisited,
            key=lambda city: dist[current, city]
        )

        tour.append(nxt)

        unvisited.remove(nxt)

        current = nxt

    return tour

def greedy_edge_tour(dist, rng):

    n = len(dist)

    edges = []

    for i in range(n):
        for j in range(i + 1, n):

            edges.append(
                (dist[i, j], i, j)
            )

    edges.sort(
        key=lambda x: x[0]
    )

    degree = [0] * n

    parent = list(range(n))

    adjacency = [[] for _ in range(n)]


    def find(x):

        while parent[x] != x:

            parent[x] = parent[parent[x]]

            x = parent[x]

        return x


    used = 0

    for _, a, b in edges:

        if degree[a] >= 2 or degree[b] >= 2:
            continue

        ra = find(a)
        rb = find(b)

        # Prevent premature cycle
        if ra == rb and used != n - 1:
            continue

        parent[ra] = rb

        adjacency[a].append(b)
        adjacency[b].append(a)

        degree[a] += 1
        degree[b] += 1

        used += 1

        if used == n:
            break


    # Repair incomplete construction if required
    ends = [
        city
        for city in range(n)
        if degree[city] < 2
    ]

    while len(ends) > 2:

        a = ends[0]

        candidates = [
            b
            for b in ends[1:]
            if find(a) != find(b)
        ]

        if not candidates:
            candidates = ends[1:]

        b = min(
            candidates,
            key=lambda x: dist[a, x]
        )

        adjacency[a].append(b)
        adjacency[b].append(a)

        parent[find(a)] = find(b)

        degree[a] += 1
        degree[b] += 1

        ends = [
            city
            for city in range(n)
            if degree[city] < 2
        ]


    if len(ends) == 2:

        a, b = ends

        adjacency[a].append(b)
        adjacency[b].append(a)


    # Convert adjacency cycle into permutation
    tour = [0]

    visited = {0}

    previous = -1
    current = 0

    while len(tour) < n:

        candidates = [
            x
            for x in adjacency[current]
            if x != previous
            and x not in visited
        ]

        if not candidates:

            candidates = [
                x
                for x in range(n)
                if x not in visited
            ]

        nxt = min(
            candidates,
            key=lambda x: dist[current, x]
        )

        tour.append(nxt)

        visited.add(nxt)

        previous, current = current, nxt

    return tour

def random_tour(n, rng):

    tour = list(range(n))

    rng.shuffle(tour)

    return tour

def two_opt(
    tour,
    dist,
    neighbors,
    max_passes=1
):

    n = len(tour)

    tour = tour[:]

    improved_any = False

    for _ in range(max_passes):

        improved = False

        pos = position_map(tour)

        for i in range(n):

            a = tour[i]
            b = tour[(i + 1) % n]

            old_ab = dist[a, b]

            for c in neighbors[a]:

                j = pos[c]

                d = tour[(j + 1) % n]

                if c == b or d == a:
                    continue

                old = (
                    old_ab
                    + dist[c, d]
                )

                new = (
                    dist[a, c]
                    + dist[b, d]
                )

                if new + 1e-9 < old:

                    left = i + 1
                    right = j

                    if left <= right:

                        tour[left:right + 1] = (
                            tour[left:right + 1][::-1]
                        )

                    else:

                        left = j + 1
                        right = i

                        tour[left:right + 1] = (
                            tour[left:right + 1][::-1]
                        )

                    improved = True
                    improved_any = True

                    break

            if improved:
                break

        if not improved:
            break

    return tour, improved_any

def or_opt(
    tour,
    dist,
    neighbors,
    segment_lengths=(1, 2, 3),
    priority_edges=None
):

    n = len(tour)

    tour = tour[:]

    improved_any = False

    priority_edges = (
        priority_edges
        if priority_edges is not None
        else set()
    )

    improvement_found = True

    while improvement_found:

        improvement_found = False

        pos = position_map(tour)

        for L in segment_lengths:

            for start in range(n):

                # Keep implementation simple and safe:
                # do not move wrapped segments.
                if start + L > n:
                    continue

                segment = tour[start:start + L]

                segment_set = set(segment)

                before = tour[(start - 1) % n]
                after = tour[(start + L) % n]

                removed_cost = (
                    dist[before, segment[0]]
                    + dist[segment[-1], after]
                    - dist[before, after]
                )

                candidates = (
                    list(neighbors[segment[0]])
                    + list(neighbors[segment[-1]])
                )

                # PMG-priority candidates first
                candidates = sorted(
                    set(candidates),
                    key=lambda city:
                    0
                    if (
                        tuple(sorted((city, segment[0])))
                        in priority_edges
                        or
                        tuple(sorted((city, segment[-1])))
                        in priority_edges
                    )
                    else 1
                )

                for city in candidates:

                    if city in segment_set:
                        continue

                    j = pos[city]

                    nxt = tour[(j + 1) % n]

                    if nxt in segment_set:
                        continue

                    forward = (
                        dist[city, segment[0]]
                        + dist[segment[-1], nxt]
                        - dist[city, nxt]
                    )

                    reverse = (
                        dist[city, segment[-1]]
                        + dist[segment[0], nxt]
                        - dist[city, nxt]
                    )

                    if min(forward, reverse) + 1e-9 >= removed_cost:
                        continue

                    remaining = (
                        tour[:start]
                        + tour[start + L:]
                    )

                    insertion = remaining.index(city) + 1

                    piece = (
                        segment
                        if forward <= reverse
                        else segment[::-1]
                    )

                    candidate = (
                        remaining[:insertion]
                        + piece
                        + remaining[insertion:]
                    )

                    if valid_tour(candidate, n):

                        tour = candidate

                        improved_any = True
                        improvement_found = True

                        break

                if improvement_found:
                    break

            if improvement_found:
                break

    return tour, improved_any

# --------------------------------------------------------------------------
# SELECTIVE LKH: Bounded, Cached, Validated & Interruptible Solver
# --------------------------------------------------------------------------
LKH_CACHE: dict[tuple, tuple[list[int], float]] = {}


def clear_lkh_cache():
    """Clears the global symmetric LKH tour cache."""
    LKH_CACHE.clear()


def canonicalize_tour(tour: list[int]) -> tuple[int, ...]:
    """Returns canonical orientation of a cyclic TSP tour for symmetric caching.

    Rotates so the minimum city ID is at index 0, and flips if neighbor[1] > neighbor[-1].
    """
    n = len(tour)
    if n < 2:
        return tuple(tour)
    min_idx = 0
    min_val = tour[0]
    for i in range(1, n):
        if tour[i] < min_val:
            min_val = tour[i]
            min_idx = i

    rotated = tour[min_idx:] + tour[:min_idx]
    if rotated[1] > rotated[-1]:
        rotated = [rotated[0]] + rotated[:0:-1]

    return tuple(rotated)


def _solve_elkai_internal(matrix: list[list[int]], runs: int, max_trials: int, timeout_sec: float) -> list[int]:
    """Internal elkai solver supporting LKH TIME_LIMIT and MAX_TRIALS parameters."""
    try:
        import elkai._elkai as _elkai
        dim = len(matrix)
        time_limit = max(1, int(math.ceil(timeout_sec)))
        params = f"RUNS = {runs}\nMAX_TRIALS = {max_trials}\nTIME_LIMIT = {time_limit}\nPROBLEM_FILE = :stdin:\n"
        problem = f"TYPE : TSP\nDIMENSION : {dim}\nEDGE_WEIGHT_TYPE : EXPLICIT\nEDGE_WEIGHT_FORMAT : FULL_MATRIX\nEDGE_WEIGHT_SECTION\n"
        for row in matrix:
            problem += " ".join(map(str, row)) + "\n"
        raw = _elkai.solve_problem(params, problem)
        return [idx - 1 for idx in raw]
    except Exception:
        return elkai.solve_int_matrix(matrix, runs=runs)


def _solve_lkh_external(tour: list[int], dist: np.ndarray, exe_path: str, runs: int, max_trials: int, timeout_sec: float) -> Optional[list[int]]:
    """Executes external LKH binary if present, with strict subprocess timeout."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        tsp_file = tmp / "problem.tsp"
        par_file = tmp / "problem.par"
        tour_file = tmp / "problem.tour"

        n = len(tour)
        with open(tsp_file, "w", encoding="utf-8") as f:
            f.write(f"NAME : problem\nTYPE : TSP\nDIMENSION : {n}\nEDGE_WEIGHT_TYPE : EXPLICIT\nEDGE_WEIGHT_FORMAT : FULL_MATRIX\nEDGE_WEIGHT_SECTION\n")
            int_dist = np.rint(dist).astype(int)
            for row in int_dist:
                f.write(" ".join(map(str, row)) + "\n")
            f.write("EOF\n")

        time_limit = max(1, int(math.ceil(timeout_sec)))
        with open(par_file, "w", encoding="utf-8") as f:
            f.write(f"PROBLEM_FILE = {tsp_file}\nOUTPUT_TOUR_FILE = {tour_file}\nRUNS = {runs}\nMAX_TRIALS = {max_trials}\nTIME_LIMIT = {time_limit}\n")

        res = subprocess.run([exe_path, str(par_file)], capture_output=True, timeout=timeout_sec, text=True)
        if res.returncode == 0 and tour_file.exists():
            lines = tour_file.read_text(encoding="utf-8").splitlines()
            reading = False
            route = []
            for line in lines:
                line = line.strip()
                if line == "TOUR_SECTION":
                    reading = True
                    continue
                if line in ("-1", "EOF"):
                    break
                if reading and line.isdigit():
                    route.append(int(line) - 1)
            if len(route) == n and valid_tour(route, n):
                return route
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
    """Bounded, cached, validated, and safely interrupted LKH execution.

    Returns: (tour, cost, status) where status is one of:
    'improved', 'no_improvement', 'cached', 'timed_out', 'error', 'fallback_improved', 'fallback'
    """
    original = tour[:]
    n = len(tour)
    orig_cost = tour_length(original, dist)

    if n < 4:
        return original, orig_cost, "too_small"

    # 1. Caching
    canon = canonicalize_tour(original) if cache_enabled else None
    cache_key = (instance_name, n, canon, runs, max_trials) if cache_enabled else None
    if cache_enabled and cache_key in LKH_CACHE:
        cached_tour, cached_cost = LKH_CACHE[cache_key]
        if cached_cost + 1e-9 < orig_cost:
            return cached_tour[:], cached_cost, "cached"
        return original, orig_cost, "cached"

    result = None
    status = "no_improvement"

    # 2. Check external LKH executable if configured
    exe = lkh_exe or os.environ.get("LKH_EXE")
    if exe and os.path.isfile(exe):
        try:
            result = _solve_lkh_external(original, dist, exe, runs, max_trials, timeout_sec)
        except Exception:
            result = None

    # 3. Use elkai if available
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

    # 4. Fallback local search if LKH failed or unavailable
    if result is None and status not in ("timed_out",):
        try:
            k_neighbors = min(15, n - 1)
            nb = build_knn_lists(dist, k_neighbors)
            cand, _ = two_opt(original, dist, nb, max_passes=2)
            cand, _ = or_opt(cand, dist, nb)
            result = cand
            status = "fallback"
        except Exception:
            result = original

    # 5. Validation
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
    tour,
    dist,
    neighbors=None,
    runs=1,
    max_trials=50,
    timeout_sec=5.0,
    instance_name="tsp",
    cache_enabled=True,
    int_matrix=None,
    lkh_exe=None
):
    """Refines route using bounded LKH. Maintains 2-tuple return for backwards compatibility."""
    orig_cost = tour_length(tour, dist)
    polished, polished_cost, _ = solve_lkh_bounded(
        tour=tour,
        dist=dist,
        runs=runs,
        max_trials=max_trials,
        timeout_sec=timeout_sec,
        instance_name=instance_name,
        cache_enabled=cache_enabled,
        int_matrix=int_matrix,
        lkh_exe=lkh_exe
    )
    improved = (polished_cost + 1e-9 < orig_cost)
    return polished, improved


def selective_lkh_detailed(
    tour,
    dist,
    neighbors=None,
    runs=1,
    max_trials=50,
    timeout_sec=5.0,
    instance_name="tsp",
    cache_enabled=True,
    int_matrix=None,
    lkh_exe=None
):
    """Detailed LKH execution returning (tour, improved_bool, status_str, duration_seconds)."""
    t0 = time.perf_counter()
    orig_cost = tour_length(tour, dist)
    polished, polished_cost, status = solve_lkh_bounded(
        tour=tour,
        dist=dist,
        runs=runs,
        max_trials=max_trials,
        timeout_sec=timeout_sec,
        instance_name=instance_name,
        cache_enabled=cache_enabled,
        int_matrix=int_matrix,
        lkh_exe=lkh_exe
    )
    dur = time.perf_counter() - t0
    improved = (polished_cost + 1e-9 < orig_cost)
    return polished, improved, status, dur

@dataclass
class Individual:
    id: int
    tour: list
    cost: float
    fitness: float
    method: str
    generation: int

def create_individual(
    idx,
    tour,
    method,
    generation,
    dist
):

    cost = tour_length(
        tour,
        dist
    )

    return Individual(
        id=idx,
        tour=tour[:],
        cost=cost,
        fitness=fitness_from_cost(cost),
        method=method,
        generation=generation
    )

def generate_initial_population(
    instance,
    optimized_seed,
    config,
    rng
):

    n = instance.n
    dist = instance.dist

    population = []

    # ========================================================
    # 17 NEAREST-NEIGHBOR TOURS
    # ========================================================

    # Evaluate every possible NN starting city,
    # then keep the best 17 available starts.

    nn_candidates = []

    for start in range(n):

        tour = nearest_neighbor_tour(
            dist,
            start
        )

        cost = tour_length(
            tour,
            dist
        )

        nn_candidates.append(
            (cost, start, tour)
        )

    nn_candidates.sort(
        key=lambda x: x[0]
    )

    selected_nn = nn_candidates[
        :min(config.nn_count, n)
    ]

    for _, start, tour in selected_nn:

        population.append(
            create_individual(
                len(population) + 1,
                tour,
                f"Nearest Neighbor (start={start})",
                0,
                dist
            )
        )

    # If n < 17, fill remaining NN slots with randomized starts
    while len(population) < config.nn_count:

        start = rng.randrange(n)

        tour = nearest_neighbor_tour(
            dist,
            start
        )

        population.append(
            create_individual(
                len(population) + 1,
                tour,
                f"Nearest Neighbor (start={start})",
                0,
                dist
            )
        )


    # ========================================================
    # 11 GREEDY EDGE TOURS
    # ========================================================

    greedy_start_index = len(population)

    # The optimized high-quality tour is intentionally used
    # as the first important greedy-derived GA seed.

    population.append(
        create_individual(
            len(population) + 1,
            optimized_seed,
            "Greedy + 2Opt + 2Opt + Selective LKH Seed",
            0,
            dist
        )
    )

    while (
        len(population)
        < greedy_start_index + config.greedy_count
    ):

        tour = greedy_edge_tour(
            dist,
            rng
        )

        # Introduce small perturbation so the 11 greedy seeds
        # do not collapse into exactly the same permutation.
        if (
            len(population)
            > greedy_start_index
            and rng.random() < 0.75
        ):

            i, j = rng.sample(
                range(n),
                2
            )

            tour[i], tour[j] = (
                tour[j],
                tour[i]
            )

        population.append(
            create_individual(
                len(population) + 1,
                tour,
                "Greedy Edge",
                0,
                dist
            )
        )


    # ========================================================
    # 42 RANDOM TOURS
    # ========================================================

    while len(population) < config.population_size:

        tour = random_tour(
            n,
            rng
        )

        population.append(
            create_individual(
                len(population) + 1,
                tour,
                "Random",
                0,
                dist
            )
        )


    assert len(population) == config.population_size

    for ind in population:

        assert valid_tour(
            ind.tour,
            n
        )

    return population

class PredictiveMemoryGraph:

    def __init__(
        self,
        n,
        decay=0.92,
        elite_fraction=0.20
    ):

        self.n = n

        self.decay = decay

        self.elite_fraction = elite_fraction

        # 1. EDGE MEMORY
        self.edge_memory = defaultdict(float)

        # 2. SUCCESSOR MEMORY
        self.successor_memory = [
            Counter()
            for _ in range(n)
        ]

        # 3. SEGMENT MEMORY
        self.segment_memory = Counter()

        # 4. PREDICTION MEMORY
        self.prediction_memory = set()

        self.trust = 1.0


    @staticmethod
    def edge_key(a, b):

        return (
            (a, b)
            if a < b
            else (b, a)
        )


    def decay_memory(self):

        # Edge memory
        for edge in list(
            self.edge_memory.keys()
        ):

            self.edge_memory[edge] *= self.decay

            if self.edge_memory[edge] < 1e-5:

                del self.edge_memory[edge]


        # Successor memory
        for row in self.successor_memory:

            for city in list(
                row.keys()
            ):

                row[city] *= self.decay

                if row[city] < 1e-5:

                    del row[city]


        # Segment memory
        for segment in list(
            self.segment_memory.keys()
        ):

            self.segment_memory[segment] *= self.decay

            if self.segment_memory[segment] < 1e-5:

                del self.segment_memory[segment]


    def update(
        self,
        population
    ):

        self.decay_memory()

        ranked = sorted(
            population,
            key=lambda x: x.cost
        )

        elite_count = max(
            1,
            int(
                len(population)
                * self.elite_fraction
            )
        )

        elites = ranked[:elite_count]

        best_cost = elites[0].cost


        for ind in elites:

            tour = ind.tour

            weight = (
                best_cost
                / max(ind.cost, 1e-12)
            )

            n = len(tour)

            # --------------------------------------------
            # EDGE + SUCCESSOR MEMORY
            # --------------------------------------------

            for i in range(n):

                a = tour[i]
                b = tour[(i + 1) % n]

                edge = self.edge_key(
                    a,
                    b
                )

                self.edge_memory[edge] += weight

                self.successor_memory[a][b] += weight

                # weaker reverse reinforcement
                self.successor_memory[b][a] += (
                    weight * 0.5
                )


            # --------------------------------------------
            # SEGMENT MEMORY
            # --------------------------------------------

            for L in (3, 4, 5):

                for i in range(n):

                    segment = tuple(
                        tour[(i + j) % n]
                        for j in range(L)
                    )

                    self.segment_memory[
                        segment
                    ] += weight


        # --------------------------------------------
        # PREDICTION MEMORY
        # --------------------------------------------

        self.rebuild_prediction_memory()


    def rebuild_prediction_memory(
        self,
        fraction=0.15
    ):

        if (
            not self.edge_memory
            or self.trust < 0.05
        ):

            self.prediction_memory = set()

            return


        ranked = sorted(
            self.edge_memory.items(),
            key=lambda x: -x[1]
        )

        effective_fraction = (
            fraction
            * self.trust
        )

        count = max(
            1,
            int(
                len(ranked)
                * effective_fraction
            )
        )

        self.prediction_memory = {
            edge
            for edge, _
            in ranked[:count]
        }


    def edge_quality(
        self,
        a,
        b
    ):

        return (
            self.trust
            * self.edge_memory.get(
                self.edge_key(a, b),
                0.0
            )
        )


    def predicted_successors(
        self,
        city,
        k=5
    ):

        return [
            c
            for c, _
            in self.successor_memory[
                city
            ].most_common(k)
        ]


    def elite_segments(
        self,
        top_k=30
    ):

        ranked = sorted(
            self.segment_memory.items(),
            key=lambda x: -x[1]
        )

        return [
            list(segment)
            for segment, _
            in ranked[:top_k]
        ]


    def increase_trust(
        self,
        amount=0.05
    ):

        self.trust = min(
            1.0,
            self.trust + amount
        )

        self.rebuild_prediction_memory()


    def reduce_trust(
        self,
        value=0.15
    ):

        self.trust = max(
            0.0,
            min(1.0, value)
        )

        self.rebuild_prediction_memory()

def tournament_selection(
    population,
    rng,
    tournament_size=4
):

    contenders = rng.sample(
        population,
        min(
            tournament_size,
            len(population)
        )
    )

    return min(
        contenders,
        key=lambda x: x.cost
    )

def ordered_crossover(
    parent1,
    parent2,
    rng
):

    n = len(parent1)

    i, j = sorted(
        rng.sample(
            range(n),
            2
        )
    )

    child = [None] * n

    child[i:j + 1] = (
        parent1[i:j + 1]
    )

    used = set(
        parent1[i:j + 1]
    )

    fill = [
        city
        for city in parent2
        if city not in used
    ]

    pointer = 0

    for idx in range(n):

        if child[idx] is None:

            child[idx] = fill[pointer]

            pointer += 1

    return child

def segment_crossover(
    parent1,
    parent2,
    rng,
    learned_segments
):

    if not learned_segments:

        return ordered_crossover(
            parent1,
            parent2,
            rng
        )


    n = len(parent1)

    usable = [
        seg
        for seg in learned_segments
        if (
            2 <= len(seg) < n
            and len(set(seg)) == len(seg)
            and all(city in parent1 for city in seg)
        )
    ]

    if not usable:

        return ordered_crossover(
            parent1,
            parent2,
            rng
        )


    segment = rng.choice(
        usable
    )

    segment_set = set(segment)

    child = [None] * n

    start = parent1.index(
        segment[0]
    )


    # If learned segment fits linearly, preserve it.
    if start + len(segment) <= n:

        for offset, city in enumerate(segment):

            child[start + offset] = city

    else:

        # Wrapped PMG segment: place at a legal random position.
        start = rng.randrange(
            0,
            n - len(segment) + 1
        )

        for offset, city in enumerate(segment):

            child[start + offset] = city


    fill = [
        city
        for city in parent2
        if city not in segment_set
    ]

    pointer = 0

    for idx in range(n):

        if child[idx] is None:

            child[idx] = fill[pointer]

            pointer += 1

    return child

def swap_mutation(
    tour,
    rng,
    fraction=0.02
):

    child = tour[:]

    n = len(child)

    swaps = max(
        1,
        int(n * fraction)
    )

    for _ in range(swaps):

        i, j = rng.sample(
            range(n),
            2
        )

        child[i], child[j] = (
            child[j],
            child[i]
        )

    return child

def weak_edge_surgery(
    tour,
    dist,
    neighbors,
    pmg,
    rng
):

    n = len(tour)

    candidate = tour[:]

    edge_info = []

    for i in range(n):

        a = candidate[i]
        b = candidate[(i + 1) % n]

        memory_quality = pmg.edge_quality(
            a,
            b
        )

        distance = dist[a, b]

        edge_info.append(
            (
                memory_quality,
                -distance,
                i
            )
        )

    # weakest learned edges first
    edge_info.sort(
        key=lambda x: (
            x[0],
            x[1]
        )
    )

    weak_positions = [
        x[2]
        for x in edge_info[
            :max(3, n // 10)
        ]
    ]

    old_cost = tour_length(
        candidate,
        dist
    )

    for i in weak_positions:

        a = candidate[i]

        predicted = pmg.predicted_successors(
            a,
            k=5
        )

        options = (
            predicted
            + neighbors[a]
        )

        for target in options:

            if target == a:
                continue

            pos = position_map(
                candidate
            )

            j = pos[target]

            if (
                j == i
                or j == (i + 1) % n
            ):
                continue

            trial = candidate[:]

            left = i + 1
            right = j

            if left <= right:

                trial[left:right + 1] = (
                    trial[left:right + 1][::-1]
                )

            else:

                continue

            if not valid_tour(
                trial,
                n
            ):
                continue

            new_cost = tour_length(
                trial,
                dist
            )

            if new_cost + 1e-9 < old_cost:

                return trial, True

    return tour[:], False

class AdaptiveOperatorSelector:

    def __init__(
        self,
        rng
    ):

        self.rng = rng

        self.names = [
            "weak_edge_surgery",
            "predictive_or_opt",
            "two_opt"
        ]

        self.credit = {
            name: 1.0
            for name in self.names
        }

        self.plays = {
            name: 0
            for name in self.names
        }


    def select(self):

        values = np.array(
            [
                self.credit[name]
                for name in self.names
            ],
            dtype=float
        )

        # Exploration floor
        probs = (
            0.10
            + 0.70
            * values / values.sum()
        )

        probs /= probs.sum()

        r = self.rng.random()

        cumulative = 0.0

        for name, p in zip(
            self.names,
            probs
        ):

            cumulative += p

            if r <= cumulative:

                self.plays[name] += 1

                return name

        name = self.names[-1]

        self.plays[name] += 1

        return name


    def update(
        self,
        operator,
        relative_gain
    ):

        reward = (
            1.0
            + 20.0
            * max(
                0.0,
                relative_gain
            )
        )

        self.credit[operator] = (
            0.75
            * self.credit[operator]
            + 0.25
            * reward
        )

def edge_set(tour):

    n = len(tour)

    return {
        tuple(
            sorted(
                (
                    tour[i],
                    tour[(i + 1) % n]
                )
            )
        )
        for i in range(n)
    }

def population_diversity(
    population,
    rng,
    sample_pairs=40
):

    if len(population) < 2:
        return 0.0

    total = 0.0
    count = 0

    for _ in range(sample_pairs):

        a, b = rng.sample(
            population,
            2
        )

        ea = edge_set(
            a.tour
        )

        eb = edge_set(
            b.tour
        )

        n = len(a.tour)

        difference = (
            len(ea.symmetric_difference(eb))
            / (2.0 * n)
        )

        total += difference
        count += 1

    return (
        total / count
        if count
        else 0.0
    )

class StagnationDetector:

    def __init__(
        self,
        window=15,
        epsilon=1e-4,
        diversity_floor=0.05
    ):

        self.window = window

        self.epsilon = epsilon

        self.diversity_floor = diversity_floor

        self.history = deque(
            maxlen=window
        )


    def observe(
        self,
        best_cost,
        diversity
    ):

        self.history.append(
            best_cost
        )

        if len(self.history) < self.window:

            return False


        old = self.history[0]

        new = self.history[-1]

        relative_improvement = (
            (old - new)
            / max(old, 1e-12)
        )

        fitness_stalled = (
            relative_improvement
            < self.epsilon
        )

        diversity_low = (
            diversity
            < self.diversity_floor
        )

        return (
            fitness_stalled
            and diversity_low
        )

def double_bridge(
    tour,
    rng
):

    n = len(tour)

    if n < 8:

        candidate = tour[:]

        rng.shuffle(candidate)

        return candidate


    p1, p2, p3 = sorted(
        rng.sample(
            range(1, n),
            3
        )
    )

    A = tour[:p1]
    B = tour[p1:p2]
    C = tour[p2:p3]
    D = tour[p3:]

    return (
        A
        + C
        + B
        + D
    )


def make_config(overrides=None, **kwargs):
    if isinstance(overrides, GAConfig):
        cfg = copy.copy(overrides)
        combined = kwargs
    else:
        cfg = GAConfig()
        combined = dict(overrides or {})
        combined.update(kwargs)
    if "stagnation_patience" in combined and "no_improvement_limit" not in combined:
        combined["no_improvement_limit"] = int(combined["stagnation_patience"])
    for k, v in combined.items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)
    cfg.no_improvement_limit = int(cfg.no_improvement_limit)
    cfg.stagnation_patience = cfg.no_improvement_limit
    cfg.population_size = max(4, int(cfg.population_size))
    cfg.nn_count = min(int(cfg.nn_count), cfg.population_size)
    cfg.greedy_count = min(int(cfg.greedy_count), max(0, cfg.population_size - cfg.nn_count))
    cfg.random_count = max(0, cfg.population_size - cfg.nn_count - cfg.greedy_count)

    # Environment variable overrides for LKH
    if "LKH_ENABLED" in os.environ:
        cfg.lkh_enabled = os.environ["LKH_ENABLED"].lower() in ("true", "1", "yes")
    if "LKH_MAX_SECONDS_PER_CALL" in os.environ:
        try: cfg.lkh_max_seconds_per_call = float(os.environ["LKH_MAX_SECONDS_PER_CALL"])
        except ValueError: pass
    if "LKH_MIN_GENERATION_GAP" in os.environ:
        try: cfg.lkh_min_generation_gap = int(os.environ["LKH_MIN_GENERATION_GAP"])
        except ValueError: pass
    if "LKH_TRIGGER_STAGNATION" in os.environ:
        try: cfg.lkh_trigger_stagnation = int(os.environ["LKH_TRIGGER_STAGNATION"])
        except ValueError: pass
    if "LKH_MAX_CALLS_PER_RUN" in os.environ:
        try: cfg.lkh_max_calls_per_run = int(os.environ["LKH_MAX_CALLS_PER_RUN"])
        except ValueError: pass
    if "LKH_MAX_TRIALS" in os.environ:
        try: cfg.lkh_max_trials = int(os.environ["LKH_MAX_TRIALS"])
        except ValueError: pass
    if "LKH_RUNS" in os.environ:
        try: cfg.lkh_runs = int(os.environ["LKH_RUNS"])
        except ValueError: pass

    return cfg


def _plain_tsplib(path):
    """Small fallback parser for common NODE_COORD_SECTION / EDGE_WEIGHT_SECTION TSPLIB files."""
    txt=Path(path).read_text(encoding='utf-8',errors='ignore')
    lines=[x.strip() for x in txt.splitlines()]
    hdr={}; sec=None; coords={}; weights=[]
    for line in lines:
        if not line: continue
        u=line.upper()
        if u.startswith('NODE_COORD_SECTION'):
            sec='coords'; continue
        if u.startswith('EDGE_WEIGHT_SECTION'):
            sec='weights'; continue
        if u.startswith('EOF'): break
        if ':' in line and sec is None:
            k,v=line.split(':',1); hdr[k.strip().upper()]=v.strip(); continue
        if sec=='coords':
            p=line.split()
            if len(p)>=3:
                try: coords[int(p[0])] = (float(p[1]),float(p[2]))
                except: pass
        elif sec=='weights':
            for p in line.split():
                try: weights.append(float(p))
                except: pass
    n=int(hdr.get('DIMENSION',len(coords)))
    nodes=list(sorted(coords)) if coords else list(range(1,n+1))
    index={node:i for i,node in enumerate(nodes)}
    dist=np.zeros((n,n),dtype=float)
    if coords:
        arr=np.array([coords[node] for node in nodes],dtype=float)
        d=arr[:,None,:]-arr[None,:,:]
        et=str(hdr.get('EDGE_WEIGHT_TYPE','EUC_2D')).upper()
        if et in {'EUC_2D','EUC_3D'}:
            dist=np.sqrt((d**2).sum(axis=2))
            if et=='EUC_2D': dist=np.floor(dist+0.5)
        else:
            dist=np.sqrt((d**2).sum(axis=2))
    elif weights:
        typ=str(hdr.get('EDGE_WEIGHT_FORMAT','FULL_MATRIX')).upper()
        a=np.array(weights,float)
        if typ=='FULL_MATRIX' and len(a)>=n*n: dist=a[:n*n].reshape(n,n)
        else:
            # triangular formats used by many TSPLIB instances
            dist=np.zeros((n,n),float); k=0
            if typ=='LOWER_DIAG_ROW':
                for i in range(n):
                    for j in range(i+1):
                        dist[i,j]=dist[j,i]=a[k]; k+=1
            elif typ=='UPPER_ROW':
                for i in range(n-1):
                    for j in range(i+1,n): dist[i,j]=dist[j,i]=a[k]; k+=1
            else:
                raise ValueError('Unsupported EDGE_WEIGHT_FORMAT in fallback parser')
    return TSPInstance(hdr.get('NAME',Path(path).stem),n,nodes,dist,np.array([coords[x] for x in nodes]) if coords else None,str(hdr.get('EDGE_WEIGHT_TYPE','UNKNOWN')))


def load_any_instance(path):
    if tsplib95 is not None:
        try: return load_tsplib_instance(path)
        except Exception: pass
    return _plain_tsplib(path)


def initial_pipeline(instance,config):
    rng=random.Random(config.seed)
    neighbors=build_knn_lists(instance.dist,config.k_neighbors)
    t0=time.perf_counter(); greedy=greedy_edge_tour(instance.dist,rng); tg=time.perf_counter()-t0
    greedy_cost=tour_length(greedy,instance.dist)
    t0=time.perf_counter(); s1,imp1=two_opt(greedy,instance.dist,neighbors,max_passes=1); t1=time.perf_counter()-t0
    s1c=tour_length(s1,instance.dist)
    t0=time.perf_counter(); s2,imp2=two_opt(s1,instance.dist,neighbors,max_passes=3); t2=time.perf_counter()-t0
    s2c=tour_length(s2,instance.dist)
    t0=time.perf_counter(); pop=generate_initial_population(instance,s2,config,rng); tp=time.perf_counter()-t0
    return {'neighbors':neighbors,'greedy':greedy,'greedy_cost':greedy_cost,'greedy_time':tg,
            'two_opt_local':s1,'two_opt_local_cost':s1c,'two_opt_local_improved':imp1,'two_opt_local_time':t1,
            'two_opt_refined':s2,'two_opt_refined_cost':s2c,'two_opt_refined_improved':imp2,'two_opt_refined_time':t2,
            'population':pop,'population_time':tp}


def snapshot_population(pop):
    return [{'id':int(x.id),'generation':int(x.generation),'method':x.method,'cost':float(x.cost),
             'fitness':float(x.fitness),'tour':list(map(int,x.tour))} for x in sorted(pop,key=lambda z:z.cost)]


def setup_state(instance,overrides=None):
    cfg=make_config(overrides); pipe=initial_pipeline(instance,cfg); pop=pipe['population']
    pmg=PredictiveMemoryGraph(instance.n,decay=cfg.memory_decay,elite_fraction=cfg.memory_elite_fraction); pmg.update(pop)
    rng=random.Random(cfg.seed); selector=AdaptiveOperatorSelector(rng)
    stagnation=StagnationDetector(window=cfg.stagnation_window,epsilon=cfg.stagnation_epsilon,diversity_floor=cfg.diversity_floor)
    best=min(pop,key=lambda x:x.cost); opt=best_known(instance.name)
    tol=float(getattr(cfg,'improvement_tolerance',1e-9))
    converged=bool(opt is not None and best.cost<=opt+tol)
    is_stopped=converged
    state='completed' if converged else 'idle'
    stop_reason='0.0% gap achieved (Known optimum found in initial population)' if converged else 'Ready'
    patience=int(getattr(cfg,'no_improvement_limit',30))

    rec={'generation':0,'best_cost':float(best.cost),'generation_best_cost':float(best.cost),
         'gap':gap_percent(best.cost,opt),'improvement':0.0,'best_id':int(best.id),
         'best_method':best.method,'route':list(map(int,best.tour)),'diversity':None,
         'operator':'initialization','memory_trust':float(pmg.trust),
         'no_improvement_count':0,'stagnation_patience':patience,
         'is_stopped':is_stopped,'state':state,'stop_reason':stop_reason,
         'execution_time':float(pipe['greedy_time']+pipe['two_opt_local_time']+pipe['two_opt_refined_time']+pipe['population_time']),
         'stage_times':{'greedy':pipe['greedy_time'],'two_opt_local':pipe['two_opt_local_time'],'two_opt_refined':pipe['two_opt_refined_time'],'initial_population':pipe['population_time']},
         'lkh_applied':False,'lkh_improvements':0,'lkh_status':'idle','lkh_duration':0.0,'lkh_calls_total':0}

    int_mat = np.rint(instance.dist).astype(int).tolist()

    return {'instance':instance,'config':cfg,'pipe':pipe,'neighbors':pipe['neighbors'],
            'population':pop,'pmg':pmg,'selector':selector,'stagnation':stagnation,'rng':rng,
            'generation':0,'global_best_tour':best.tour[:],'global_best_cost':float(best.cost),
            'best_generation':0,'history':[float(best.cost)],'records':[rec],
            'populations':{0:snapshot_population(pop)},'improvements':[rec.copy()],
            'converged':converged,'is_stopped':is_stopped,'state':state,
            'stop_reason':stop_reason,'stop_requested':False,'no_improvement':0,
            'feature_totals':Counter(),'route_history':{0:list(map(int,best.tour))},
            'int_matrix':int_mat,
            'lkh_calls_total':0,
            'last_lkh_generation':-999,
            'lkh_last_duration':0.0,
            'lkh_status':'idle',
            'diagnostics':[f"Initialized {instance.name} ({instance.n} cities). Initial best cost: {best.cost:,.2f}"]}


def run_generation(st,options=None):
    options=options or {}
    # Do not execute if already stopped
    if st.get('is_stopped',False):
        return st['records'][-1] if st.get('records') else None

    cfg,rng=st['config'],st['rng']; inst=st['instance']; dist=inst.dist; pop=st['population']; pmg=st['pmg']
    selector=st['selector']; stagnation=st['stagnation']; gen=st['generation']+1; prev=st['global_best_cost']
    timings={}; feature_gain=defaultdict(float); changed=defaultdict(int)
    t_all=time.perf_counter()

    # 1 selection/crossover/mutation/fast 2-opt
    t=time.perf_counter(); ranked=sorted(pop,key=lambda x:x.cost); next_tours=[(e.tour[:],'Elite') for e in ranked[:cfg.elite_count]]
    learned=pmg.elite_segments(top_k=30) if options.get('predictive_memory',True) else []
    while len(next_tours)<cfg.population_size:
        p1=tournament_selection(pop,rng,cfg.tournament_size); p2=tournament_selection(pop,rng,cfg.tournament_size)
        use_seg=bool(learned) and options.get('segment_crossover',True) and rng.random()<(cfg.segment_crossover_probability*max(.25,pmg.trust))
        if use_seg: child=segment_crossover(p1.tour,p2.tour,rng,learned); method='Segment Crossover'; changed['segment_crossover']+=1
        else: child=ordered_crossover(p1.tour,p2.tour,rng); method='OX Crossover'; changed['ordered_crossover']+=1
        if options.get('swap_mutation',True) and rng.random()<cfg.mutation_rate:
            child=swap_mutation(child,rng,cfg.swap_fraction); method+=' + Swap Mutation'; changed['swap_mutation']+=1
        if options.get('fast_2opt',True) and rng.random()<cfg.offspring_local_search_probability:
            before=tour_length(child,dist); child,_=two_opt(child,dist,st['neighbors'],max_passes=cfg.fast_two_opt_passes); after=tour_length(child,dist)
            if after<before-1e-9: feature_gain['fast_2opt']+=before-after
            method+=' + Fast 2-Opt'; changed['fast_2opt']+=1
        next_tours.append((child,method))
    pop=[create_individual(i,t,m,gen,dist) for i,(t,m) in enumerate(next_tours,1)]; timings['evolution_operators']=time.perf_counter()-t

    # 2 fitness ranking
    t=time.perf_counter(); ranked=sorted(pop,key=lambda x:x.cost); pre=ranked[0].cost; timings['fitness_ranking']=time.perf_counter()-t

    # 3 predictive memory update
    t=time.perf_counter();
    if options.get('predictive_memory',True): pmg.update(pop)
    timings['predictive_memory']=time.perf_counter()-t

    # 4 adaptive operator refinement
    selected=selector.select() if options.get('adaptive_operator',True) else 'two_opt'
    t=time.perf_counter(); adaptive_count=min(cfg.adaptive_elite_count,len(pop)); idxs=sorted(range(len(pop)),key=lambda i:pop[i].cost)[:adaptive_count]
    for idx in idxs:
        cur=pop[idx]; cand=cur.tour[:]; ok=False
        if selected=='weak_edge_surgery' and options.get('weak_edge_surgery',True): cand,ok=weak_edge_surgery(cand,dist,st['neighbors'],pmg,rng)
        elif selected=='predictive_or_opt' and options.get('predictive_or_opt',True): cand,ok=or_opt(cand,dist,st['neighbors'],priority_edges=pmg.prediction_memory)
        else: cand,ok=two_opt(cand,dist,st['neighbors'],max_passes=2) if options.get('adaptive_two_opt',True) else (cand,False)
        if ok:
            c=tour_length(cand,dist)
            if c<cur.cost-1e-9:
                feature_gain[selected]+=cur.cost-c; changed[selected]+=1
                pop[idx]=create_individual(cur.id,cand,cur.method+' + '+selected,gen,dist)
    post=min(x.cost for x in pop); gain=max(0.0,(pre-post)/max(pre,1e-12))
    if options.get('adaptive_operator',True): selector.update(selected,gain)
    timings['adaptive_refinement']=time.perf_counter()-t

    # 5 diversity and stagnation
    t=time.perf_counter(); div=population_diversity(pop,rng,sample_pairs=40) if options.get('diversity_analysis',True) else None; timings['diversity']=time.perf_counter()-t
    t=time.perf_counter(); cur_best=min(pop,key=lambda x:x.cost); stagnant=stagnation.observe(cur_best.cost,div if div is not None else 1.0); timings['stagnation']=time.perf_counter()-t
    bridge=False
    if stagnant and options.get('double_bridge_escape',True):
        pmg.reduce_trust(.15); idxs=sorted(range(len(pop)),key=lambda i:pop[i].cost,reverse=True)[:max(2,cfg.population_size//6)]
        for idx in idxs:
            esc=double_bridge(st['global_best_tour'],rng); esc,_=two_opt(esc,dist,st['neighbors'],max_passes=1)
            pop[idx]=create_individual(pop[idx].id,esc,'Double-Bridge Escape + Fast 2-Opt',gen,dist); changed['double_bridge_escape']+=1; bridge=True
    timings['diversification']=time.perf_counter()-t

    # 6 Adaptive Selective LKH Optimization
    lkh_applied = False
    lkh_imp = 0
    lkh_status = st.get('lkh_status', 'idle')
    lkh_dur = 0.0

    lkh_opt_enabled = options.get('selective_lkh', True) and getattr(cfg, 'lkh_enabled', True)
    is_already_stopped = st.get('is_stopped', False) or st.get('converged', False)

    if lkh_opt_enabled and not is_already_stopped:
        calls_total = st.get('lkh_calls_total', 0)
        last_lkh_gen = st.get('last_lkh_generation', -999)
        max_calls = getattr(cfg, 'lkh_max_calls_per_run', 8)
        min_gap = getattr(cfg, 'lkh_min_generation_gap', 10)
        stag_threshold = getattr(cfg, 'lkh_trigger_stagnation', 10)
        cadence = getattr(cfg, 'lkh_cadence', 15)

        gap_satisfied = (gen - last_lkh_gen) >= min_gap
        budget_available = calls_total < max_calls

        stagnation_triggered = st.get('no_improvement', 0) >= stag_threshold
        cadence_triggered = (cadence > 0 and gen % cadence == 0)

        should_trigger = (stagnation_triggered or cadence_triggered) and gap_satisfied and budget_available

        if not budget_available and (stagnation_triggered or cadence_triggered):
            lkh_status = 'skipped_budget_exhausted'
        elif not gap_satisfied and (stagnation_triggered or cadence_triggered):
            lkh_status = 'skipped_min_gap'
        elif should_trigger:
            t_lkh = time.perf_counter()
            lkh_applied = True
            num_inds = max(1, getattr(cfg, 'lkh_individuals', 1))
            cand_idxs = sorted(range(len(pop)), key=lambda i: pop[i].cost)[:min(num_inds, len(pop))]

            seen_canons = set()
            unique_idxs = []
            for idx in cand_idxs:
                c_key = canonicalize_tour(pop[idx].tour)
                if c_key not in seen_canons:
                    seen_canons.add(c_key)
                    unique_idxs.append(idx)

            int_mat = st.get('int_matrix')
            if int_mat is None:
                int_mat = np.rint(dist).astype(int).tolist()
                st['int_matrix'] = int_mat

            per_call_statuses = []
            for idx in unique_idxs:
                cur = pop[idx]
                polished, ok, status_str, call_dur = selective_lkh_detailed(
                    cur.tour,
                    dist,
                    st['neighbors'],
                    runs=getattr(cfg, 'lkh_runs', 1),
                    max_trials=getattr(cfg, 'lkh_max_trials', 50),
                    timeout_sec=getattr(cfg, 'lkh_max_seconds_per_call', 5.0),
                    instance_name=inst.name,
                    cache_enabled=getattr(cfg, 'lkh_cache_enabled', True),
                    int_matrix=int_mat
                )
                st['lkh_calls_total'] = st.get('lkh_calls_total', 0) + 1
                per_call_statuses.append(status_str)
                lkh_dur += call_dur

                if ok:
                    c = tour_length(polished, dist)
                    if c < cur.cost - 1e-9:
                        feature_gain['selective_lkh'] += (cur.cost - c)
                        lkh_imp += 1
                        changed['selective_lkh'] += 1
                        pop[idx] = create_individual(cur.id, polished, cur.method + ' + Selective LKH', gen, dist)

            lkh_elapsed = time.perf_counter() - t_lkh
            timings['selective_lkh'] = lkh_elapsed
            st['last_lkh_generation'] = gen
            st['lkh_last_duration'] = lkh_elapsed
            lkh_status = per_call_statuses[0] if per_call_statuses else 'completed'
            st['lkh_status'] = lkh_status
        else:
            timings['selective_lkh'] = 0.0
            lkh_status = 'idle'
    else:
        timings['selective_lkh'] = 0.0
        if not lkh_opt_enabled:
            lkh_status = 'disabled'
        elif is_already_stopped:
            lkh_status = 'skipped_run_stopped'

    st['lkh_status'] = lkh_status

    # Global best comparison with strict numerical tolerance
    tol=float(getattr(cfg,'improvement_tolerance',1e-9))
    gb=min(pop,key=lambda x:x.cost); new_global=False; improvement=0.0
    if gb.cost < st['global_best_cost'] - tol:
        improvement=st['global_best_cost']-gb.cost
        st['global_best_cost']=float(gb.cost)
        st['global_best_tour']=gb.tour[:]
        st['best_generation']=gen
        st['no_improvement']=0
        new_global=True
        if 'diagnostics' in st:
            st['diagnostics'].append(f"Gen {gen}: Discovered new global-best {gb.cost:,.2f} (-{improvement:,.2f})")
    else:
        st['no_improvement']+=1

    opt=best_known(inst.name); gap=gap_percent(st['global_best_cost'],opt)
    st['generation']=gen; st['population']=pop; st['history'].append(st['global_best_cost']); st['route_history'][gen]=list(map(int,st['global_best_tour']))

    # Immediate stopping criterion evaluation
    patience=int(getattr(cfg,'no_improvement_limit',30))
    if opt is not None and st['global_best_cost']<=opt+tol:
        st['converged']=True
        st['is_stopped']=True
        st['state']='completed'
        st['stop_reason']='0.0% gap achieved (Known optimum found)'
        if 'diagnostics' in st:
            st['diagnostics'].append(f"Gen {gen}: 0.0% optimality gap achieved. Experiment completed.")
    elif st['no_improvement']>=patience:
        st['is_stopped']=True
        st['state']='stopped_stagnation'
        st['stop_reason']=f'No improvement for {patience} consecutive generations'
        if 'diagnostics' in st:
            st['diagnostics'].append(f"Gen {gen}: Stagnation limit reached ({patience} non-improving generations). Terminated.")
    elif gen>=cfg.max_generations:
        st['is_stopped']=True
        st['state']='stopped_max_generations'
        st['stop_reason']=f'Maximum generation limit reached ({cfg.max_generations})'
        if 'diagnostics' in st:
            st['diagnostics'].append(f"Gen {gen}: Maximum generation limit reached.")
    elif st.get('stop_requested',False):
        st['is_stopped']=True
        st['state']='stopped_by_user'
        st['stop_reason']='Stopped by user request'
        if 'diagnostics' in st:
            st['diagnostics'].append(f"Gen {gen}: Stopped by user request.")
    else:
        st['state']='running'

    rec={'generation':gen,'best_cost':float(st['global_best_cost']),'generation_best_cost':float(gb.cost),
         'gap':gap,'improvement':float(improvement),'best_id':int(gb.id),'best_method':gb.method,
         'route':list(map(int,st['global_best_tour'])),
         'diversity':None if div is None else float(div),'operator':selected,
         'memory_trust':float(pmg.trust),'no_improvement_count':int(st['no_improvement']),
         'stagnation_patience':patience,'is_stopped':bool(st.get('is_stopped',False)),
         'state':st.get('state','running'),'stop_reason':st.get('stop_reason',''),
         'new_global_best':new_global,'double_bridge_applied':bridge,
         'lkh_applied':lkh_applied,'lkh_improvements':lkh_imp,'lkh_status':lkh_status,
         'lkh_duration':float(timings.get('selective_lkh',0.0)),
         'lkh_calls_total':int(st.get('lkh_calls_total',0)),
         'feature_gain':dict(feature_gain),
         'changed':dict(changed),'execution_time':float(time.perf_counter()-t_all),
         'stage_times':{k:float(v) for k,v in timings.items()}}

    st['records'].append(rec); st['populations'][gen]=snapshot_population(pop); st['route_history'][gen]=list(map(int,st['global_best_tour']))
    if new_global: st['improvements'].append(rec.copy())
    for k,v in feature_gain.items(): st['feature_totals'][k]+=v
    if not stagnant and cur_best.cost<prev-1e-9: pmg.increase_trust(.05)
    return rec


