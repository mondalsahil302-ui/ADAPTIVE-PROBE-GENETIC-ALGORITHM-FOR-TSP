"""
Ablation Definitions and Execution Runner for APGGS Study.
Preserves the exact ablation logic and parameters from the evaluation study.
"""
from __future__ import annotations

import contextlib
import io
import random
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import apggs_source


@dataclass
class AblationConfig:
    use_probe: bool = True
    use_pmg: bool = True
    use_double_bridge: bool = True
    use_selective_lkh: bool = True
    use_adaptive_control: bool = True
    basic_ga: bool = False


# Exact required comparison structure
REQUIRED_VARIANTS: List[str] = [
    "Full APGGS",
    "Without PMG",
    "Without Double Bridge",
    "Without Adaptive Control",
    "Basic GA",
]

ABLATION_CONFIGS: Dict[str, AblationConfig] = {
    "Full APGGS": AblationConfig(
        use_probe=True,
        use_pmg=True,
        use_double_bridge=True,
        use_selective_lkh=True,
        use_adaptive_control=True,
        basic_ga=False,
    ),
    "Without PMG": AblationConfig(
        use_pmg=False,
        use_selective_lkh=False,
    ),
    "Without Double Bridge": AblationConfig(
        use_double_bridge=False,
        use_selective_lkh=False,
    ),
    "Without Adaptive Control": AblationConfig(
        use_adaptive_control=False,
        use_selective_lkh=False,
    ),
    "Basic GA": AblationConfig(
        use_probe=False,
        use_pmg=False,
        use_double_bridge=False,
        use_selective_lkh=False,
        use_adaptive_control=False,
        basic_ga=True,
    ),
}


def variant_label_from_config(config: AblationConfig) -> str:
    """Returns the canonical variant label from an AblationConfig."""
    if config.basic_ga:
        return "Basic GA"
    if not config.use_pmg:
        return "Without PMG"
    if not config.use_double_bridge:
        return "Without Double Bridge"
    if not config.use_adaptive_control:
        return "Without Adaptive Control"
    return "Full APGGS"


def build_standard_population(
    instance: Any,
    seed: int,
) -> Tuple[List[Any], Dict[int, List[int]]]:
    """
    Constructs the shared initial population and neighbor lists using the
    exact evaluation methodology so that all 5 variants start with
    the identical population.
    """
    rng = random.Random(seed)
    config = apggs_source.GAConfig(seed=seed)
    neighbors = apggs_source.build_knn_lists(instance.dist, config.k_neighbors)
    seed_tour = apggs_source.nearest_neighbor_tour(instance.dist, start=0)
    optimized_seed, _ = apggs_source.two_opt(
        seed_tour,
        instance.dist,
        neighbors,
        max_passes=2,
    )
    population = apggs_source.generate_initial_population(
        instance,
        optimized_seed,
        config,
        rng,
    )
    return population, neighbors


def run_variant(
    instance: Any,
    initial_population: List[Any],
    neighbors: Dict[int, List[int]],
    config: AblationConfig,
    seed: int,
    max_generations: int,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Executes a single ablation variant up to max_generations.
    Preserves exact ablation mechanics via dynamic replacement and restores
    original components upon completion.
    """
    cfg = apggs_source.GAConfig(seed=seed)
    cfg.max_generations = int(max_generations)

    # Disable selective LKH for all ablation variants silently
    if not config.use_selective_lkh:
        cfg.lkh_cadence = int(max_generations) + 1

    original = {
        "AdaptiveOperatorSelector": apggs_source.AdaptiveOperatorSelector,
        "PredictiveMemoryGraph": apggs_source.PredictiveMemoryGraph,
        "stagnation_observe": apggs_source.StagnationDetector.observe,
    }

    try:
        # 1. Without Adaptive Control: fixed 2-opt selector
        if not config.use_adaptive_control:
            class FixedSelector:
                def __init__(self, rng):
                    self.plays = {
                        "weak_edge_surgery": 0,
                        "predictive_or_opt": 0,
                        "two_opt": 0,
                    }

                def select(self):
                    return "two_opt"

                def update(self, operator, relative_gain):
                    pass

            apggs_source.AdaptiveOperatorSelector = FixedSelector

        # 2. Without Double Bridge: stagnation detector never triggers
        if not config.use_double_bridge:
            def never_stagnant(self, best_cost, diversity):
                return False

            apggs_source.StagnationDetector.observe = never_stagnant

        # 3. Without PMG / Without Probe: PMG memory and probe disabled
        if not config.use_pmg or not config.use_probe:
            base_pmg = original["PredictiveMemoryGraph"]
            use_memory = config.use_pmg
            use_probe = config.use_probe

            class AblatedPredictiveMemoryGraph(base_pmg):
                def update(self, population):
                    if use_memory:
                        return super().update(population)

                def edge_quality(self, a, b):
                    return super().edge_quality(a, b) if use_memory else 0.0

                def predicted_successors(self, city, k=5):
                    return (
                        super().predicted_successors(city, k)
                        if use_memory and use_probe
                        else []
                    )

                def elite_segments(self, top_k=30):
                    return super().elite_segments(top_k) if use_memory else []

            apggs_source.PredictiveMemoryGraph = AblatedPredictiveMemoryGraph

        # 4. Basic GA: standard parameter configuration
        if config.basic_ga:
            cfg.mutation_rate = 0.15
            cfg.offspring_local_search_probability = 0.15
            cfg.adaptive_elite_count = 0
            cfg.memory_elite_fraction = 0.0

        start = time.perf_counter()

        # Suppress internal noisy prints if verbose is False
        if not verbose:
            with contextlib.redirect_stdout(io.StringIO()):
                result = apggs_source.run_adaptive_ga(
                    instance,
                    initial_population,
                    neighbors,
                    cfg,
                )
        else:
            result = apggs_source.run_adaptive_ga(
                instance,
                initial_population,
                neighbors,
                cfg,
            )

        runtime = time.perf_counter() - start

        result["runtime"] = float(runtime)
        result["variant"] = variant_label_from_config(config)
        result["seed"] = int(seed)
        result["max_generations"] = int(max_generations)

        assert result["generations"] <= int(max_generations), (
            f"Variant {result['variant']} executed {result['generations']} "
            f"generations, exceeding cutoff {max_generations}."
        )

        return result

    finally:
        # Restore original classes and methods
        apggs_source.AdaptiveOperatorSelector = original["AdaptiveOperatorSelector"]
        apggs_source.PredictiveMemoryGraph = original["PredictiveMemoryGraph"]
        apggs_source.StagnationDetector.observe = original["stagnation_observe"]

