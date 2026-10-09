"""
Adaptive Probe Genetic Algorithm Core with Predictive Memory,
Adaptive Operator Selection, and Bounded LKH Integration.
"""
from __future__ import annotations

import copy
import random
import time
from collections import Counter, defaultdict, deque
from typing import Any, Optional

import numpy as np

from backend.config import GAConfig, make_config
from backend.optimizer.heuristics import (
    double_bridge,
    greedy_edge_tour,
    nearest_neighbor_tour,
    or_opt,
    random_tour,
    two_opt,
)
from backend.optimizer.lkh import canonicalize_tour, selective_lkh_detailed
from backend.optimizer.models import (
    Individual,
    TSPInstance,
    best_known,
    build_knn_lists,
    fitness_from_cost,
    gap_percent,
    population_diversity,
    position_map,
    tour_length,
    valid_tour,
)


class PredictiveMemoryGraph:
    """Predictive Memory Graph (PMG) for edge frequencies and predictive elite segments."""

    def __init__(self, n: int, decay: float = 0.96, elite_fraction: float = 0.20):
        self.n = n
        self.decay = decay
        self.elite_fraction = elite_fraction
        self.edge_memory: dict[tuple[int, int], float] = defaultdict(float)
        self.successor_memory: list[Counter] = [Counter() for _ in range(n)]
        self.segment_memory: Counter = Counter()
        self.prediction_memory: set[tuple[int, int]] = set()
        self.trust: float = 0.50

    @staticmethod
    def edge_key(u: int, v: int) -> tuple[int, int]:
        return (u, v) if u < v else (v, u)

    def update(self, population: list[Individual]):
        for key in list(self.edge_memory.keys()):
            self.edge_memory[key] *= self.decay
            if self.edge_memory[key] < 1e-4:
                del self.edge_memory[key]

        ranked = sorted(population, key=lambda x: x.cost)
        elite_count = max(2, int(len(ranked) * self.elite_fraction))
        elites = ranked[:elite_count]

        for rank, individual in enumerate(elites):
            weight = (elite_count - rank) / elite_count
            tour = individual.tour
            n = len(tour)
            for i in range(n):
                a, b = tour[i], tour[(i + 1) % n]
                self.edge_memory[self.edge_key(a, b)] += weight
                self.successor_memory[a][b] += weight
                self.successor_memory[b][a] += weight * 0.5

            for L in (3, 4, 5):
                for i in range(n):
                    seg = tuple(tour[(i + j) % n] for j in range(L))
                    self.segment_memory[seg] += weight

        self.rebuild_prediction_memory()

    def rebuild_prediction_memory(self, fraction: float = 0.15):
        if not self.edge_memory or self.trust < 0.05:
            self.prediction_memory = set()
            return
        ranked = sorted(self.edge_memory.items(), key=lambda x: -x[1])
        count = max(1, int(len(ranked) * fraction * self.trust))
        self.prediction_memory = {edge for edge, _ in ranked[:count]}

    def edge_quality(self, a: int, b: int) -> float:
        return self.trust * self.edge_memory.get(self.edge_key(a, b), 0.0)

    def predicted_successors(self, city: int, k: int = 5) -> list[int]:
        return [c for c, _ in self.successor_memory[city].most_common(k)]

    def elite_segments(self, top_k: int = 30) -> list[list[int]]:
        ranked = sorted(self.segment_memory.items(), key=lambda x: -x[1])
        return [list(segment) for segment, _ in ranked[:top_k]]

    def increase_trust(self, amount: float = 0.05):
        self.trust = min(1.0, self.trust + amount)
        self.rebuild_prediction_memory()

    def reduce_trust(self, value: float = 0.15):
        self.trust = max(0.0, min(1.0, self.trust - value))
        self.rebuild_prediction_memory()


class AdaptiveOperatorSelector:
    """Multi-armed bandit selector with adaptive credit assignment."""

    def __init__(self, rng: random.Random):
        self.rng = rng
        self.names = ["weak_edge_surgery", "predictive_or_opt", "two_opt"]
        self.credit = {name: 1.0 for name in self.names}
        self.plays = {name: 0 for name in self.names}

    def select(self) -> str:
        values = np.array([self.credit[name] for name in self.names], dtype=float)
        probs = 0.10 + 0.70 * values / values.sum()
        probs /= probs.sum()
        r = self.rng.random()
        cumulative = 0.0
        for name, p in zip(self.names, probs):
            cumulative += p
            if r <= cumulative:
                self.plays[name] += 1
                return name
        name = self.names[-1]
        self.plays[name] += 1
        return name

    def update(self, operator: str, relative_gain: float):
        reward = 1.0 + 20.0 * max(0.0, relative_gain)
        self.credit[operator] = 0.75 * self.credit[operator] + 0.25 * reward


class StagnationDetector:
    """Sliding-window stagnation detector evaluating fitness stalls and diversity floors."""

    def __init__(self, window: int = 15, epsilon: float = 1e-4, diversity_floor: float = 0.05):
        self.window = window
        self.epsilon = epsilon
        self.diversity_floor = diversity_floor
        self.history = deque(maxlen=window)

    def observe(self, best_cost: float, diversity: float) -> bool:
        self.history.append(best_cost)
        if len(self.history) < self.window:
            return False
        old = self.history[0]
        new = self.history[-1]
        relative_improvement = (old - new) / max(old, 1e-12)
        return (relative_improvement < self.epsilon) and (diversity < self.diversity_floor)


def tournament_selection(population: list[Individual], rng: random.Random, tournament_size: int = 4) -> Individual:
    contenders = rng.sample(population, min(tournament_size, len(population)))
    return min(contenders, key=lambda x: x.cost)


def ordered_crossover(parent1: list[int], parent2: list[int], rng: random.Random) -> list[int]:
    n = len(parent1)
    i, j = sorted(rng.sample(range(n), 2))
    child = [None] * n
    child[i : j + 1] = parent1[i : j + 1]
    used = set(parent1[i : j + 1])
    fill = [city for city in parent2 if city not in used]
    pointer = 0
    for idx in range(n):
        if child[idx] is None:
            child[idx] = fill[pointer]
            pointer += 1
    return child


def segment_crossover(parent1: list[int], parent2: list[int], rng: random.Random, learned_segments: list[list[int]]) -> list[int]:
    if not learned_segments:
        return ordered_crossover(parent1, parent2, rng)
    n = len(parent1)
    usable = [seg for seg in learned_segments if 2 <= len(seg) < n and len(set(seg)) == len(seg) and all(c in parent1 for c in seg)]
    if not usable:
        return ordered_crossover(parent1, parent2, rng)
    segment = rng.choice(usable)
    segment_set = set(segment)
    child = [None] * n
    start = parent1.index(segment[0])
    if start + len(segment) <= n:
        for offset, city in enumerate(segment):
            child[start + offset] = city
    else:
        start = rng.randrange(0, n - len(segment) + 1)
        for offset, city in enumerate(segment):
            child[start + offset] = city
    fill = [city for city in parent2 if city not in segment_set]
    pointer = 0
    for idx in range(n):
        if child[idx] is None:
            child[idx] = fill[pointer]
            pointer += 1
    return child


def swap_mutation(tour: list[int], rng: random.Random, fraction: float = 0.02) -> list[int]:
    child = tour[:]
    n = len(child)
    swaps = max(1, int(n * fraction))
    for _ in range(swaps):
        i, j = rng.sample(range(n), 2)
        child[i], child[j] = child[j], child[i]
    return child


def weak_edge_surgery(tour: list[int], dist: np.ndarray, neighbors: list[list[int]], pmg: PredictiveMemoryGraph, rng: random.Random) -> tuple[list[int], bool]:
    n = len(tour)
    candidate = tour[:]
    edge_info = []
    for i in range(n):
        a, b = candidate[i], candidate[(i + 1) % n]
        edge_info.append((pmg.edge_quality(a, b), -dist[a, b], i))
    edge_info.sort(key=lambda x: (x[0], x[1]))
    weak_positions = [x[2] for x in edge_info[: max(3, n // 10)]]
    old_cost = tour_length(candidate, dist)
    for i in weak_positions:
        a = candidate[i]
        options = pmg.predicted_successors(a, k=5) + neighbors[a]
        for target in options:
            if target == a:
                continue
            pos = position_map(candidate)
            j = pos[target]
            if j == i or j == (i + 1) % n:
                continue
            trial = candidate[:]
            left, right = i + 1, j
            if left <= right:
                trial[left : right + 1] = trial[left : right + 1][::-1]
            else:
                continue
            if not valid_tour(trial, n):
                continue
            new_cost = tour_length(trial, dist)
            if new_cost + 1e-9 < old_cost:
                return trial, True
    return tour[:], False


def create_individual(idx: int, tour: list[int], method: str, generation: int, dist: np.ndarray) -> Individual:
    cost = tour_length(tour, dist)
    return Individual(
        id=idx,
        tour=tour[:],
        cost=cost,
        fitness=fitness_from_cost(cost),
        method=method,
        generation=generation,
    )


def generate_initial_population(instance: TSPInstance, optimized_seed: list[int], config: GAConfig, rng: random.Random) -> list[Individual]:
    n = instance.n
    dist = instance.dist
    population = []

    # 1. Nearest Neighbor tours
    nn_candidates = []
    for start in range(n):
        t = nearest_neighbor_tour(dist, start)
        c = tour_length(t, dist)
        nn_candidates.append((c, start, t))
    nn_candidates.sort(key=lambda x: x[0])
    selected_nn = nn_candidates[: min(config.nn_count, n)]
    for _, start, tour in selected_nn:
        population.append(create_individual(len(population) + 1, tour, f"Nearest Neighbor (start={start})", 0, dist))
    while len(population) < config.nn_count:
        start = rng.randrange(n)
        tour = nearest_neighbor_tour(dist, start)
        population.append(create_individual(len(population) + 1, tour, f"Nearest Neighbor (start={start})", 0, dist))

    # 2. Greedy Edge tours
    greedy_start_index = len(population)
    population.append(create_individual(len(population) + 1, optimized_seed, "Greedy + 2Opt + 2Opt + Seed", 0, dist))
    while len(population) < greedy_start_index + config.greedy_count:
        t = greedy_edge_tour(dist, rng)
        if len(population) > greedy_start_index and rng.random() < 0.75:
            i, j = rng.sample(range(n), 2)
            t[i], t[j] = t[j], t[i]
        population.append(create_individual(len(population) + 1, t, "Greedy Edge", 0, dist))

    # 3. Random tours
    while len(population) < config.population_size:
        t = random_tour(n, rng)
        population.append(create_individual(len(population) + 1, t, "Random", 0, dist))

    return population


def initial_pipeline(instance: TSPInstance, config: GAConfig) -> dict[str, Any]:
    rng = random.Random(config.seed)
    neighbors = build_knn_lists(instance.dist, config.k_neighbors)
    t0 = time.perf_counter()
    greedy = greedy_edge_tour(instance.dist, rng)
    tg = time.perf_counter() - t0
    greedy_cost = tour_length(greedy, instance.dist)

    t0 = time.perf_counter()
    s1, imp1 = two_opt(greedy, instance.dist, neighbors, max_passes=1)
    t1 = time.perf_counter() - t0
    s1c = tour_length(s1, instance.dist)

    t0 = time.perf_counter()
    s2, imp2 = two_opt(s1, instance.dist, neighbors, max_passes=3)
    t2 = time.perf_counter() - t0
    s2c = tour_length(s2, instance.dist)

    t0 = time.perf_counter()
    pop = generate_initial_population(instance, s2, config, rng)
    tp = time.perf_counter() - t0

    return {
        "neighbors": neighbors,
        "greedy": greedy,
        "greedy_cost": greedy_cost,
        "greedy_time": tg,
        "two_opt_local": s1,
        "two_opt_local_cost": s1c,
        "two_opt_local_improved": imp1,
        "two_opt_local_time": t1,
        "two_opt_refined": s2,
        "two_opt_refined_cost": s2c,
        "two_opt_refined_improved": imp2,
        "two_opt_refined_time": t2,
        "population": pop,
        "population_time": tp,
    }


def snapshot_population(pop: list[Individual]) -> list[dict[str, Any]]:
    return [
        {
            "id": int(x.id),
            "generation": int(x.generation),
            "method": x.method,
            "cost": float(x.cost),
            "fitness": float(x.fitness),
            "tour": list(map(int, x.tour)),
        }
        for x in sorted(pop, key=lambda z: z.cost)
    ]


def setup_state(instance: TSPInstance, overrides: Optional[dict[str, Any] | GAConfig] = None) -> dict[str, Any]:
    cfg = make_config(overrides)
    pipe = initial_pipeline(instance, cfg)
    pop = pipe["population"]
    pmg = PredictiveMemoryGraph(instance.n, decay=cfg.memory_decay, elite_fraction=cfg.memory_elite_fraction)
    pmg.update(pop)
    rng = random.Random(cfg.seed)
    selector = AdaptiveOperatorSelector(rng)
    stagnation = StagnationDetector(window=cfg.stagnation_window, epsilon=cfg.stagnation_epsilon, diversity_floor=cfg.diversity_floor)
    best = min(pop, key=lambda x: x.cost)
    opt = best_known(instance.name)
    tol = float(getattr(cfg, "improvement_tolerance", 1e-9))
    converged = bool(opt is not None and best.cost <= opt + tol)
    is_stopped = converged
    state = "completed" if converged else "idle"
    stop_reason = "0.0% gap achieved (Known optimum found in initial population)" if converged else "Ready"
    patience = int(getattr(cfg, "no_improvement_limit", 30))

    rec = {
        "generation": 0,
        "best_cost": float(best.cost),
        "generation_best_cost": float(best.cost),
        "gap": gap_percent(best.cost, opt),
        "improvement": 0.0,
        "best_id": int(best.id),
        "best_method": best.method,
        "route": list(map(int, best.tour)),
        "diversity": None,
        "operator": "initialization",
        "memory_trust": float(pmg.trust),
        "no_improvement_count": 0,
        "stagnation_patience": patience,
        "is_stopped": is_stopped,
        "state": state,
        "stop_reason": stop_reason,
        "execution_time": float(pipe["greedy_time"] + pipe["two_opt_local_time"] + pipe["two_opt_refined_time"] + pipe["population_time"]),
        "stage_times": {
            "greedy": pipe["greedy_time"],
            "two_opt_local": pipe["two_opt_local_time"],
            "two_opt_refined": pipe["two_opt_refined_time"],
            "initial_population": pipe["population_time"],
        },
        "lkh_applied": False,
        "lkh_improvements": 0,
        "lkh_status": "idle",
        "lkh_duration": 0.0,
        "lkh_calls_total": 0,
    }

    int_mat = np.rint(instance.dist).astype(int).tolist()

    return {
        "instance": instance,
        "config": cfg,
        "pipe": pipe,
        "neighbors": pipe["neighbors"],
        "population": pop,
        "pmg": pmg,
        "selector": selector,
        "stagnation": stagnation,
        "rng": rng,
        "generation": 0,
        "global_best_tour": best.tour[:],
        "global_best_cost": float(best.cost),
        "best_generation": 0,
        "history": [float(best.cost)],
        "records": [rec],
        "populations": {0: snapshot_population(pop)},
        "improvements": [rec.copy()],
        "converged": converged,
        "is_stopped": is_stopped,
        "state": state,
        "stop_reason": stop_reason,
        "stop_requested": False,
        "no_improvement": 0,
        "feature_totals": Counter(),
        "route_history": {0: list(map(int, best.tour))},
        "int_matrix": int_mat,
        "lkh_calls_total": 0,
        "last_lkh_generation": -999,
        "lkh_last_duration": 0.0,
        "lkh_status": "idle",
        "diagnostics": [f"Initialized {instance.name} ({instance.n} cities). Initial best cost: {best.cost:,.2f}"],
    }


def run_generation(st: dict[str, Any], options: Optional[dict[str, Any]] = None) -> Optional[dict[str, Any]]:
    options = options or {}
    if st.get("is_stopped", False):
        return st["records"][-1] if st.get("records") else None

    cfg, rng = st["config"], st["rng"]
    inst = st["instance"]
    dist = inst.dist
    pop = st["population"]
    pmg = st["pmg"]
    selector = st["selector"]
    stagnation = st["stagnation"]
    gen = st["generation"] + 1
    prev = st["global_best_cost"]
    timings = {}
    feature_gain = defaultdict(float)
    changed = defaultdict(int)
    t_all = time.perf_counter()

    # 1. Selection / Crossover / Mutation / Fast 2-Opt
    t = time.perf_counter()
    ranked = sorted(pop, key=lambda x: x.cost)
    next_tours = [(e.tour[:], "Elite") for e in ranked[: cfg.elite_count]]
    learned = pmg.elite_segments(top_k=30) if options.get("predictive_memory", True) else []

    while len(next_tours) < cfg.population_size:
        p1 = tournament_selection(pop, rng, cfg.tournament_size)
        p2 = tournament_selection(pop, rng, cfg.tournament_size)
        use_seg = bool(learned) and options.get("segment_crossover", True) and rng.random() < (cfg.segment_crossover_probability * max(0.25, pmg.trust))
        if use_seg:
            child = segment_crossover(p1.tour, p2.tour, rng, learned)
            method = "Segment Crossover"
            changed["segment_crossover"] += 1
        else:
            child = ordered_crossover(p1.tour, p2.tour, rng)
            method = "OX Crossover"
            changed["ordered_crossover"] += 1
        if options.get("swap_mutation", True) and rng.random() < cfg.mutation_rate:
            child = swap_mutation(child, rng, cfg.swap_fraction)
            method += " + Swap Mutation"
            changed["swap_mutation"] += 1
        if options.get("fast_2opt", True) and rng.random() < cfg.offspring_local_search_probability:
            before = tour_length(child, dist)
            child, _ = two_opt(child, dist, st["neighbors"], max_passes=cfg.fast_two_opt_passes)
            after = tour_length(child, dist)
            if after < before - 1e-9:
                feature_gain["fast_2opt"] += before - after
            method += " + Fast 2-Opt"
            changed["fast_2opt"] += 1
        next_tours.append((child, method))

    pop = [create_individual(i, t, m, gen, dist) for i, (t, m) in enumerate(next_tours, 1)]
    timings["evolution_operators"] = time.perf_counter() - t

    # 2. Fitness ranking
    t = time.perf_counter()
    ranked = sorted(pop, key=lambda x: x.cost)
    pre = ranked[0].cost
    timings["fitness_ranking"] = time.perf_counter() - t

    # 3. Predictive memory update
    t = time.perf_counter()
    if options.get("predictive_memory", True):
        pmg.update(pop)
    timings["predictive_memory"] = time.perf_counter() - t

    # 4. Adaptive operator refinement
    selected = selector.select() if options.get("adaptive_operator", True) else "two_opt"
    t = time.perf_counter()
    adaptive_count = min(cfg.adaptive_elite_count, len(pop))
    idxs = sorted(range(len(pop)), key=lambda i: pop[i].cost)[:adaptive_count]
    for idx in idxs:
        cur = pop[idx]
        cand = cur.tour[:]
        ok = False
        if selected == "weak_edge_surgery" and options.get("weak_edge_surgery", True):
            cand, ok = weak_edge_surgery(cand, dist, st["neighbors"], pmg, rng)
        elif selected == "predictive_or_opt" and options.get("predictive_or_opt", True):
            cand, ok = or_opt(cand, dist, st["neighbors"], priority_edges=pmg.prediction_memory)
        else:
            cand, ok = two_opt(cand, dist, st["neighbors"], max_passes=2) if options.get("adaptive_two_opt", True) else (cand, False)
        if ok:
            c = tour_length(cand, dist)
            if c < cur.cost - 1e-9:
                feature_gain[selected] += cur.cost - c
                changed[selected] += 1
                pop[idx] = create_individual(cur.id, cand, cur.method + " + " + selected, gen, dist)

    post = min(x.cost for x in pop)
    gain = max(0.0, (pre - post) / max(pre, 1e-12))
    if options.get("adaptive_operator", True):
        selector.update(selected, gain)
    timings["adaptive_refinement"] = time.perf_counter() - t

    # 5. Diversity and stagnation
    t = time.perf_counter()
    div = population_diversity(pop, rng, sample_pairs=40) if options.get("diversity_analysis", True) else None
    timings["diversity"] = time.perf_counter() - t
    t = time.perf_counter()
    cur_best = min(pop, key=lambda x: x.cost)
    stagnant = stagnation.observe(cur_best.cost, div if div is not None else 1.0)
    timings["stagnation"] = time.perf_counter() - t
    bridge = False
    if stagnant and options.get("double_bridge_escape", True):
        pmg.reduce_trust(0.15)
        idxs = sorted(range(len(pop)), key=lambda i: pop[i].cost, reverse=True)[: max(2, cfg.population_size // 6)]
        for idx in idxs:
            esc = double_bridge(st["global_best_tour"], rng)
            esc, _ = two_opt(esc, dist, st["neighbors"], max_passes=1)
            pop[idx] = create_individual(pop[idx].id, esc, "Double-Bridge Escape + Fast 2-Opt", gen, dist)
            changed["double_bridge_escape"] += 1
            bridge = True
    timings["diversification"] = time.perf_counter() - t

    # 6. Adaptive Selective LKH Optimization
    lkh_applied = False
    lkh_imp = 0
    lkh_status = st.get("lkh_status", "idle")
    lkh_dur = 0.0

    lkh_opt_enabled = options.get("selective_lkh", True) and getattr(cfg, "lkh_enabled", True)
    is_already_stopped = st.get("is_stopped", False) or st.get("converged", False)

    if lkh_opt_enabled and not is_already_stopped:
        calls_total = st.get("lkh_calls_total", 0)
        last_lkh_gen = st.get("last_lkh_generation", -999)
        max_calls = getattr(cfg, "lkh_max_calls_per_run", 8)
        min_gap = getattr(cfg, "lkh_min_generation_gap", 10)
        stag_threshold = getattr(cfg, "lkh_trigger_stagnation", 10)
        cadence = getattr(cfg, "lkh_cadence", 15)

        gap_satisfied = (gen - last_lkh_gen) >= min_gap
        budget_available = calls_total < max_calls

        stagnation_triggered = st.get("no_improvement", 0) >= stag_threshold
        cadence_triggered = cadence > 0 and gen % cadence == 0

        should_trigger = (stagnation_triggered or cadence_triggered) and gap_satisfied and budget_available

        if not budget_available and (stagnation_triggered or cadence_triggered):
            lkh_status = "skipped_budget_exhausted"
        elif not gap_satisfied and (stagnation_triggered or cadence_triggered):
            lkh_status = "skipped_min_gap"
        elif should_trigger:
            t_lkh = time.perf_counter()
            lkh_applied = True
            num_inds = max(1, getattr(cfg, "lkh_individuals", 1))
            cand_idxs = sorted(range(len(pop)), key=lambda i: pop[i].cost)[: min(num_inds, len(pop))]

            seen_canons = set()
            unique_idxs = []
            for idx in cand_idxs:
                c_key = canonicalize_tour(pop[idx].tour)
                if c_key not in seen_canons:
                    seen_canons.add(c_key)
                    unique_idxs.append(idx)

            int_mat = st.get("int_matrix")
            if int_mat is None:
                int_mat = np.rint(dist).astype(int).tolist()
                st["int_matrix"] = int_mat

            per_call_statuses = []
            for idx in unique_idxs:
                cur = pop[idx]
                polished, ok, status_str, call_dur = selective_lkh_detailed(
                    cur.tour,
                    dist,
                    st["neighbors"],
                    runs=getattr(cfg, "lkh_runs", 1),
                    max_trials=getattr(cfg, "lkh_max_trials", 50),
                    timeout_sec=getattr(cfg, "lkh_max_seconds_per_call", 5.0),
                    instance_name=inst.name,
                    cache_enabled=getattr(cfg, "lkh_cache_enabled", True),
                    int_matrix=int_mat,
                )
                st["lkh_calls_total"] = st.get("lkh_calls_total", 0) + 1
                per_call_statuses.append(status_str)
                lkh_dur += call_dur

                if ok:
                    c = tour_length(polished, dist)
                    if c < cur.cost - 1e-9:
                        feature_gain["selective_lkh"] += cur.cost - c
                        lkh_imp += 1
                        changed["selective_lkh"] += 1
                        pop[idx] = create_individual(cur.id, polished, cur.method + " + Selective LKH", gen, dist)

            lkh_elapsed = time.perf_counter() - t_lkh
            timings["selective_lkh"] = lkh_elapsed
            st["last_lkh_generation"] = gen
            st["lkh_last_duration"] = lkh_elapsed
            lkh_status = per_call_statuses[0] if per_call_statuses else "completed"
            st["lkh_status"] = lkh_status
        else:
            timings["selective_lkh"] = 0.0
            lkh_status = "idle"
    else:
        timings["selective_lkh"] = 0.0
        if not lkh_opt_enabled:
            lkh_status = "disabled"
        elif is_already_stopped:
            lkh_status = "skipped_run_stopped"

    st["lkh_status"] = lkh_status

    # Global best comparison with strict numerical tolerance
    tol = float(getattr(cfg, "improvement_tolerance", 1e-9))
    gb = min(pop, key=lambda x: x.cost)
    new_global = False
    improvement = 0.0
    if gb.cost < st["global_best_cost"] - tol:
        improvement = st["global_best_cost"] - gb.cost
        st["global_best_cost"] = float(gb.cost)
        st["global_best_tour"] = gb.tour[:]
        st["best_generation"] = gen
        st["no_improvement"] = 0
        new_global = True
        if "diagnostics" in st:
            st["diagnostics"].append(f"Gen {gen}: Discovered new global-best {gb.cost:,.2f} (-{improvement:,.2f})")
    else:
        st["no_improvement"] += 1

    opt = best_known(inst.name)
    gap = gap_percent(st["global_best_cost"], opt)
    st["generation"] = gen
    st["population"] = pop
    st["history"].append(st["global_best_cost"])
    st["route_history"][gen] = list(map(int, st["global_best_tour"]))

    # Immediate stopping criterion evaluation
    patience = int(getattr(cfg, "no_improvement_limit", 30))
    if opt is not None and st["global_best_cost"] <= opt + tol:
        st["converged"] = True
        st["is_stopped"] = True
        st["state"] = "completed"
        st["stop_reason"] = "0.0% gap achieved (Known optimum found)"
        if "diagnostics" in st:
            st["diagnostics"].append(f"Gen {gen}: 0.0% optimality gap achieved. Experiment completed.")
    elif st["no_improvement"] >= patience:
        st["is_stopped"] = True
        st["state"] = "stopped_stagnation"
        st["stop_reason"] = f"No improvement for {patience} consecutive generations"
        if "diagnostics" in st:
            st["diagnostics"].append(f"Gen {gen}: Stagnation limit reached ({patience} non-improving generations). Terminated.")
    elif gen >= cfg.max_generations:
        st["is_stopped"] = True
        st["state"] = "stopped_max_generations"
        st["stop_reason"] = f"Maximum generation limit reached ({cfg.max_generations})"
        if "diagnostics" in st:
            st["diagnostics"].append(f"Gen {gen}: Maximum generation limit reached.")
    elif st.get("stop_requested", False):
        st["is_stopped"] = True
        st["state"] = "stopped_by_user"
        st["stop_reason"] = "Stopped by user request"
        if "diagnostics" in st:
            st["diagnostics"].append(f"Gen {gen}: Stopped by user request.")
    else:
        st["state"] = "running"

    rec = {
        "generation": gen,
        "best_cost": float(st["global_best_cost"]),
        "generation_best_cost": float(gb.cost),
        "gap": gap,
        "improvement": float(improvement),
        "best_id": int(gb.id),
        "best_method": gb.method,
        "route": list(map(int, st["global_best_tour"])),
        "diversity": None if div is None else float(div),
        "operator": selected,
        "memory_trust": float(pmg.trust),
        "no_improvement_count": int(st["no_improvement"]),
        "stagnation_patience": patience,
        "is_stopped": bool(st.get("is_stopped", False)),
        "state": st.get("state", "running"),
        "stop_reason": st.get("stop_reason", ""),
        "new_global_best": new_global,
        "double_bridge_applied": bridge,
        "lkh_applied": lkh_applied,
        "lkh_improvements": lkh_imp,
        "lkh_status": lkh_status,
        "lkh_duration": float(timings.get("selective_lkh", 0.0)),
        "lkh_calls_total": int(st.get("lkh_calls_total", 0)),
        "feature_gain": dict(feature_gain),
        "changed": dict(changed),
        "execution_time": float(time.perf_counter() - t_all),
        "stage_times": {k: float(v) for k,v in timings.items()},
    }

    st["records"].append(rec)
    st["populations"][gen] = snapshot_population(pop)
    st["route_history"][gen] = list(map(int, st["global_best_tour"]))
    if new_global:
        st["improvements"].append(rec.copy())
    for k, v in feature_gain.items():
        st["feature_totals"][k] += v
    if not stagnant and cur_best.cost < prev - 1e-9:
        pmg.increase_trust(0.05)
    return rec
