# ADAPTIVE-PROBE-GENETIC-ALGORITHM-FOR-TSP
Hybrid Traveling Salesman Problem (TSP) optimization framework combining Genetic Algorithms, Predictive Memory Graph (PMG), Fast 2-Opt local search, adaptive operator selection, stagnation handling, and periodic Selective LKH refinement.
# APGGS — Adaptive Probe-Guided Genetic Search for TSP

A hybrid TSP optimization framework for TSPLIB benchmark instances.

* Upload and parse `.tsp` / TSPLIB files
* Generate complete distance matrix
* Build K-Nearest Neighbor (KNN) lists
* Construct high-quality Greedy + 2-Opt seed tours
* Create a 70-individual initial population
* 17 Nearest Neighbor tours
* 11 Greedy Edge Selection tours
* 42 Random valid TSP tours
* Implement Predictive Memory Graph (PMG)
* Learn edge, successor, segment, and prediction memories
* Use Tournament Selection for parent selection
* Support OX and Segment Crossover operators
* Apply Swap Mutation while preserving valid permutations
* Perform Fast 2-Opt local evolution on offspring
* Use adaptive operators: Weak Edge Surgery, Predictive Or-Opt, and 2-Opt
* Analyze population diversity and detect stagnation
* Apply Double-Bridge escape perturbation when necessary
* Run Selective LKH refinement every 15 generations
* Stop automatically at 0.0% gap or after 30 generations without improvement
* Provide convergence graphs, final optimized route visualization, and detailed evolutionary outputs

