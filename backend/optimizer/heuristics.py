"""
Constructive and Local Search Heuristics for the Traveling Salesperson Problem.
"""
from __future__ import annotations

import random
from typing import Optional

import numpy as np

from backend.optimizer.models import tour_length, valid_tour, position_map


def nearest_neighbor_tour(dist: np.ndarray, start: int = 0) -> list[int]:
    n = dist.shape[0]
    visited = [False] * n
    tour = [start]
    visited[start] = True
    curr = start
    for _ in range(n - 1):
        best_d = np.inf
        nxt = -1
        for j in range(n):
            if not visited[j] and dist[curr, j] < best_d:
                best_d = dist[curr, j]
                nxt = j
        tour.append(nxt)
        visited[nxt] = True
        curr = nxt
    return tour


def greedy_edge_tour(dist: np.ndarray, rng: random.Random) -> list[int]:
    n = dist.shape[0]
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            edges.append((dist[i, j], i, j))
    rng.shuffle(edges)
    edges.sort(key=lambda x: x[0])

    parent = list(range(n))
    degree = [0] * n
    adj: list[list[int]] = [[] for _ in range(n)]

    def find(i):
        path = []
        while parent[i] != i:
            path.append(i)
            i = parent[i]
        for node in path:
            parent[node] = i
        return i

    def union(i, j):
        root_i = find(i)
        root_j = find(j)
        if root_i != root_j:
            parent[root_i] = root_j
            return True
        return False

    edge_count = 0
    for _, u, v in edges:
        if degree[u] < 2 and degree[v] < 2:
            if edge_count < n - 1:
                if find(u) != find(v):
                    union(u, v)
                    degree[u] += 1
                    degree[v] += 1
                    adj[u].append(v)
                    adj[v].append(u)
                    edge_count += 1
            else:
                degree[u] += 1
                degree[v] += 1
                adj[u].append(v)
                adj[v].append(u)
                edge_count += 1
                break

    # Reconstruct tour from adjacency lists
    start = 0
    tour = [start]
    prev = -1
    curr = start
    while len(tour) < n:
        nbrs = adj[curr]
        nxt = nbrs[0] if (len(nbrs) > 0 and nbrs[0] != prev) else (nbrs[1] if len(nbrs) > 1 else -1)
        if nxt == -1:
            break
        tour.append(nxt)
        prev, curr = curr, nxt

    if len(tour) < n:
        missing = [x for x in range(n) if x not in tour]
        rng.shuffle(missing)
        tour.extend(missing)

    return tour


def random_tour(n: int, rng: random.Random) -> list[int]:
    t = list(range(n))
    rng.shuffle(t)
    return t


def two_opt(tour: list[int], dist: np.ndarray, neighbors: list[list[int]], max_passes: int = 1) -> tuple[list[int], bool]:
    candidate = tour[:]
    n = len(candidate)
    improved_any = False
    for _ in range(max_passes):
        improved = False
        pos = position_map(candidate)
        for i in range(n):
            u = candidate[i]
            u_next = candidate[(i + 1) % n]
            d_u_unext = dist[u, u_next]
            for v in neighbors[u]:
                if v == u or v == u_next:
                    continue
                j = pos[v]
                v_next = candidate[(j + 1) % n]
                if v_next == u or v_next == u_next:
                    continue
                left, right = (i, j) if i < j else (j, i)
                if left == 0 and right == n - 1:
                    continue
                old_len = dist[candidate[left], candidate[(left + 1) % n]] + dist[candidate[right], candidate[(right + 1) % n]]
                new_len = dist[candidate[left], candidate[right]] + dist[candidate[(left + 1) % n], candidate[(right + 1) % n]]
                if new_len + 1e-9 < old_len:
                    candidate[left + 1 : right + 1] = candidate[left + 1 : right + 1][::-1]
                    pos = position_map(candidate)
                    improved = True
                    improved_any = True
                    break
            if improved:
                break
        if not improved:
            break
    return candidate, improved_any


def or_opt(tour: list[int], dist: np.ndarray, neighbors: list[list[int]], priority_edges: Optional[set] = None) -> tuple[list[int], bool]:
    candidate = tour[:]
    n = len(candidate)
    improved = False
    old_cost = tour_length(candidate, dist)
    for seg_len in (3, 2, 1):
        for i in range(n):
            if i + seg_len > n:
                continue
            seg = candidate[i : i + seg_len]
            rem = candidate[:i] + candidate[i + seg_len :]
            a = seg[0]
            targets = []
            if priority_edges:
                for x in neighbors[a]:
                    if (a, x) in priority_edges or (x, a) in priority_edges:
                        targets.append(x)
            targets += [x for x in neighbors[a] if x not in targets]
            for t in targets:
                if t not in rem:
                    continue
                idx = rem.index(t)
                for pos in (idx, idx + 1):
                    trial = rem[:pos] + seg + rem[pos:]
                    if not valid_tour(trial, n):
                        continue
                    new_cost = tour_length(trial, dist)
                    if new_cost + 1e-9 < old_cost:
                        return trial, True
    return candidate, False


def double_bridge(tour: list[int], rng: random.Random) -> list[int]:
    n = len(tour)
    if n < 8:
        candidate = tour[:]
        rng.shuffle(candidate)
        return candidate
    p1, p2, p3 = sorted(rng.sample(range(1, n), 3))
    return tour[:p1] + tour[p2:p3] + tour[p1:p2] + tour[p3:]
