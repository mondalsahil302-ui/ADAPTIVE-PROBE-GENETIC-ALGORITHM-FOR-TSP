"""
ablation.py

The five APGGS ablation configurations and the variant-running harness.
Adapted from APGGS_EVALUATION_ABLATION_STUDY.ipynb: the AblationConfig
dataclass, the five ABLATION_CONFIGS definitions, and run_variant() are
carried over unchanged in behavior. The only change from the notebook
version is that this module calls directly into apggs_source instead
of dynamically re-parsing a notebook file at runtime (that indirection
is no longer needed now that apggs_source.py is a normal module).

Ablation definitions (unchanged from the evaluation study):

    Full APGGS                : everything on, selective LKH enabled.
    Without PMG                : use_pmg=False, selective LKH disabled.
    Without Double Bridge      : use_double_bridge=False, selective LKH disabled.
    Without Adaptive Control   : use_adaptive_control=False, selective LKH disabled.
    Basic GA                   : everything off, basic_ga=True, selective LKH disabled.

Full APGGS keeps selective LKH enabled; all four ablation variants keep
it disabled by pushing lkh_cadence past the run's cutoff generation (no
"LKH disabled" style status messages are ever printed).
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass

import apggs_source as source


@dataclass
class AblationConfig:
    use_probe: bool = True
    use_pmg: bool = True
    use_double_bridge: bool = True
    use_selective_lkh: bool = True
    use_adaptive_control: bool = True
    basic_ga: bool = False


ABLATION_CONFIGS = {
    "Full APGGS": AblationConfig(),
    "Without PMG": AblationConfig(use_pmg=False, use_selective_lkh=False),
    "Without Double Bridge": AblationConfig(use_double_bridge=False, use_selective_lkh=False),
    "Without Adaptive Control": AblationConfig(use_adaptive_control=False, use_selective_lkh=False),
    "Basic GA": AblationConfig(
        use_probe=False,
        use_pmg=False,
        use_double_bridge=False,
        use_selective_lkh=False,
        use_adaptive_control=False,
        basic_ga=True,
    ),
}

# Fixed processing order for each dataset's five-configuration run.
VARIANT_ORDER = [
    "Full APGGS",
    "Without PMG",
    "Without Double Bridge",
    "Without Adaptive Control",
    "Basic GA",
]


def variant_name(config: AblationConfig) -> str:
    if config.basic_ga:
        return "Basic GA"
    if not config.use_pmg:
        return "Without PMG"
    if not config.use_double_bridge:
        return "Without Double Bridge"
    if not config.use_adaptive_control:
        return "Without Adaptive Control"
    return "Full APGGS"


def standard_population(instance, seed: int):
    """Build the same standard 70-individual initial population used by
    the source notebook, for a given TSP instance and seed. Fair
    comparison across variants relies on this being identical for every
    ablation of a given dataset/seed pair."""
    rng = random.Random(seed)
    config = source.GAConfig(seed=seed)
    neighbors = source.build_knn_lists(instance.dist, config.k_neighbors)
    seed_tour = source.nearest_neighbor_tour(instance.dist, start=0)
    optimized_seed, _ = source.two_opt(
        seed_tour,
        instance.dist,
        neighbors,
        max_passes=2,
    )
    population = source.generate_initial_population(
        instance,
        optimized_seed,
        config,
        rng,
    )
    return population, neighbors


def run_variant(instance, config: AblationConfig, seed: int, cutoff_generation: int):
    """Run a single ablation variant through exactly cutoff_generation
    generations (never beyond it -- see CUTOFF RULE in the project
    spec). Returns the run_adaptive_ga() result dict plus 'runtime',
    'variant' and 'seed' fields."""
    cfg = source.GAConfig(seed=seed)
    cfg.max_generations = int(cutoff_generation)

    if not config.use_selective_lkh:
        # Push the periodic-LKH cadence past the run so it never fires,
        # without emitting any "LKH disabled" style message.
        cfg.lkh_cadence = int(cutoff_generation) + 1

    population, neighbors = standard_population(instance, seed)

    original_selector = source.AdaptiveOperatorSelector
    original_pmg = source.PredictiveMemoryGraph
    original_stagnation_observe = source.StagnationDetector.observe

    try:
        if not config.use_adaptive_control:
            class FixedSelector:
                def __init__(self, rng):
                    self.plays = {"weak_edge_surgery": 0, "predictive_or_opt": 0, "two_opt": 0}

                def select(self):
                    return "two_opt"

                def update(self, operator, relative_gain):
                    pass

            source.AdaptiveOperatorSelector = FixedSelector

        if not config.use_double_bridge:
            def never_stagnant(self, best_cost, diversity):
                return False

            source.StagnationDetector.observe = never_stagnant

        if not config.use_pmg or not config.use_probe:
            base_pmg = original_pmg
            use_memory = config.use_pmg
            use_probe = config.use_probe

            class AblatedPredictiveMemoryGraph(base_pmg):
                def update(self, population):
                    if use_memory:
                        return super().update(population)

                def edge_quality(self, a, b):
                    return super().edge_quality(a, b) if use_memory else 0.0

                def predicted_successors(self, city, k=5):
                    return super().predicted_successors(city, k) if use_memory and use_probe else []

                def elite_segments(self, top_k=30):
                    return super().elite_segments(top_k) if use_memory else []

            source.PredictiveMemoryGraph = AblatedPredictiveMemoryGraph

        if config.basic_ga:
            cfg.mutation_rate = 0.15
            cfg.offspring_local_search_probability = 0.15
            cfg.adaptive_elite_count = 0
            cfg.memory_elite_fraction = 0.0

        start = time.time()
        result = source.run_adaptive_ga(instance, population, neighbors, cfg)
        result["runtime"] = time.time() - start
        result["variant"] = variant_name(config)
        result["seed"] = seed
        result["cutoff_generation"] = int(cutoff_generation)

        if result["generations"] > cutoff_generation:
            raise RuntimeError(
                f"Variant {variant_name(config)} executed "
                f"{result['generations']} generations, exceeding its "
                f"evaluation cutoff of {cutoff_generation}."
            )

        return result
    finally:
        source.AdaptiveOperatorSelector = original_selector
        source.PredictiveMemoryGraph = original_pmg
        source.StagnationDetector.observe = original_stagnation_observe
