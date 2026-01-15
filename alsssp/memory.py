"""
Shared Memory Architecture for ALSSSP
=====================================

This module implements the persistent memory layer that enables ALSSSP to
learn from and adapt to query patterns over time. It maintains:

1. Graph Data (CSR format)
   - Cache-optimized edge storage with sorted neighbors
   - BFS-based vertex reordering for locality

2. Query Cache (Multi-level)
   - Level 1: LRU cache for exact (source, target) pairs
   - Level 2: Precomputed SSSP trees for "hot" sources
   - Level 3: Landmark distances for heuristic guidance

3. Learning Components
   - Query pattern tracker (identifies frequently accessed vertices)
   - Algorithm performance model (predicts best algorithm per query type)
   - Heuristic refinement (improves A* guidance over time)

The key insight is that shortest path queries in real applications are not
uniformly random - they exhibit temporal and spatial locality that can be
exploited for better performance.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Set
from collections import defaultdict
import heapq

from .learning import LRUCache, QueryPatternLearner, AlgorithmSelector, LearnedHeuristic

# Sentinel value for unreachable vertices
INF = float('inf')


class SharedGraphMemory:
    """
    Central memory manager for ALSSSP.

    This class orchestrates all persistent state:
    - Graph representation in cache-friendly format
    - Multi-level query cache
    - Learning components for adaptation

    The design goal is to make repeated queries significantly faster by:
    1. Caching exact results (instant lookup)
    2. Precomputing SSSP trees for popular sources
    3. Improving algorithm selection based on observed performance
    4. Refining heuristics based on actual distances

    Memory usage is controlled via the cache_size parameter. Larger caches
    improve hit rates but consume more RAM.
    """

    def __init__(self, n: int, edges: List[Tuple[int, int, float]],
                 cache_size: int = 100000, hot_threshold: int = 10):
        """
        Initialize shared memory with a graph.

        Args:
            n: Number of vertices in the graph
            edges: List of (source, target, weight) tuples
            cache_size: Maximum entries in the query cache
            hot_threshold: Queries before a source gets precomputed SSSP tree

        The initialization performs several preprocessing steps that take
        O(m + n log n) time but enable faster queries afterward.
        """
        self.n = n
        self.m = len(edges)
        self.original_edges = edges

        # Analyze graph structure to guide algorithm selection
        self._analyze_graph(edges)

        # Compute cache-friendly vertex ordering via BFS
        # This ensures topologically nearby vertices are stored together
        self.node_order = self._compute_cache_friendly_order(edges)
        self.reverse_order = np.argsort(self.node_order)

        # Build the main graph representation
        self._build_csr(edges)

        # Initialize cache hierarchy
        self.query_cache = LRUCache(maxsize=cache_size)
        self.hot_sources: Set[int] = set()
        self.hot_source_trees: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}

        # Initialize learning components
        self.query_learner = QueryPatternLearner(hot_threshold=hot_threshold)
        self.algo_selector = AlgorithmSelector()
        self.heuristic = LearnedHeuristic(n, num_landmarks=16)

        # Performance tracking
        self.query_count = 0
        self.cache_hits = 0
        self.hot_hits = 0

        # Precompute landmark distances for heuristics
        self._initialize_landmarks()

    def _analyze_graph(self, edges: List[Tuple[int, int, float]]):
        """
        Analyze graph properties to inform algorithm selection.

        We extract features that help predict which algorithm will perform
        best for queries on this graph:
        - Weight distribution (unit, small integer, continuous)
        - Graph density (sparse vs dense)
        - Average degree (affects search space growth)
        """
        self.weights = [w for _, _, w in edges]

        if self.weights:
            self.max_weight = max(self.weights)
            self.min_weight = min(self.weights)
            self.avg_weight = sum(self.weights) / len(self.weights)
        else:
            self.max_weight = 0
            self.min_weight = 0
            self.avg_weight = 0

        # Detect special weight structures that enable specialized algorithms
        self.has_unit_weights = len(set(int(w) for w in self.weights)) == 1
        self.has_small_weights = self.max_weight < self.n
        self.has_integer_weights = all(w == int(w) for w in self.weights)

        # Compute density metrics
        self.density = self.m / max(self.n * (self.n - 1), 1)
        self.avg_degree = 2 * self.m / max(self.n, 1)

    def _compute_cache_friendly_order(self, edges: List[Tuple[int, int, float]]) -> np.ndarray:
        """
        Compute vertex ordering for better cache locality.

        The idea: if vertices that are close in the graph are also close in
        memory, then shortest path searches will access memory more sequentially,
        improving cache hit rates.

        We use BFS ordering from vertex 0, which places vertices at similar
        distances from 0 together. This is simple but effective.

        For even better results, graph partitioning tools like METIS could be
        used, but BFS provides a good balance of quality vs. preprocessing time.
        """
        if self.n == 0:
            return np.array([], dtype=np.int32)

        # Build undirected adjacency for ordering purposes
        adj = defaultdict(list)
        for u, v, _ in edges:
            adj[u].append(v)
            adj[v].append(u)

        # BFS traversal from vertex 0
        order = []
        visited = set()
        queue = [0]
        visited.add(0)

        while queue:
            u = queue.pop(0)
            order.append(u)

            for v in adj[u]:
                if v not in visited:
                    visited.add(v)
                    queue.append(v)

        # Handle disconnected components
        for v in range(self.n):
            if v not in visited:
                order.append(v)

        return np.array(order, dtype=np.int32)

    def _build_csr(self, edges: List[Tuple[int, int, float]]):
        """
        Build Compressed Sparse Row representation.

        CSR stores the graph as three arrays:
        - offsets: For each vertex v, offsets[v] is where its edges start
        - neighbors: Destination vertices, packed contiguously
        - edge_weights: Corresponding weights

        Benefits:
        - Sequential memory access when iterating edges
        - Minimal memory overhead (no pointers)
        - Excellent cache behavior on modern CPUs
        """
        # Count outgoing edges per vertex
        degree = np.zeros(self.n, dtype=np.int32)
        for u, v, w in edges:
            degree[u] += 1

        # Compute prefix sums for offset array
        self.offsets = np.zeros(self.n + 1, dtype=np.int32)
        self.offsets[1:] = np.cumsum(degree)

        # Allocate packed edge arrays
        self.neighbors = np.zeros(self.m, dtype=np.int32)
        self.edge_weights = np.zeros(self.m, dtype=np.float64)

        # Fill edge arrays
        current = np.zeros(self.n, dtype=np.int32)
        for u, v, w in edges:
            idx = self.offsets[u] + current[u]
            self.neighbors[idx] = v
            self.edge_weights[idx] = w
            current[u] += 1

        # Sort neighbors within each adjacency list for cache locality
        for u in range(self.n):
            start, end = self.offsets[u], self.offsets[u + 1]
            if end > start:
                order = np.argsort(self.neighbors[start:end])
                self.neighbors[start:end] = self.neighbors[start:end][order]
                self.edge_weights[start:end] = self.edge_weights[start:end][order]

        # Also maintain adjacency list format for algorithms that need it
        self.adj = defaultdict(list)
        for u, v, w in edges:
            self.adj[u].append((v, w))

    def _initialize_landmarks(self):
        """
        Initialize landmarks for heuristic computation.

        Landmarks are a small set of vertices with precomputed distances
        to/from all other vertices. Using the triangle inequality, these
        distances provide admissible lower bounds for A* search:

            |dist(u, L) - dist(v, L)| <= dist(u, v)

        We select landmarks spread across the graph and precompute their
        SSSP trees once during initialization.
        """
        self.heuristic.select_landmarks(self.adj)

        # Precompute distances from each landmark
        def dijkstra_from(source):
            return self._run_dijkstra(source)

        self.heuristic.precompute_distances(dijkstra_from)

    def _run_dijkstra(self, source: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run standard Dijkstra for internal use (landmark precomputation).

        This is a simple, standalone implementation used during initialization.
        For actual queries, we use the optimized algorithm classes.
        """
        dist = np.full(self.n, INF, dtype=np.float64)
        pred = np.full(self.n, -1, dtype=np.int32)
        dist[source] = 0.0

        pq = [(0.0, source)]

        while pq:
            d, u = heapq.heappop(pq)
            if d > dist[u]:
                continue

            for idx in range(self.offsets[u], self.offsets[u + 1]):
                v = self.neighbors[idx]
                w = self.edge_weights[idx]
                if d + w < dist[v]:
                    dist[v] = d + w
                    pred[v] = u
                    heapq.heappush(pq, (dist[v], v))

        return dist, pred

    def get_cached_distance(self, source: int, target: int) -> Optional[float]:
        """
        Attempt to retrieve a cached distance.

        Checks caches in order of decreasing specificity:
        1. Exact query cache (source, target) -> distance
        2. Hot source tree (if source has precomputed SSSP)

        Returns None if the distance is not cached.
        """
        self.query_count += 1

        # Level 1: Check exact query cache
        cached = self.query_cache.get((source, target))
        if cached is not None:
            self.cache_hits += 1
            return cached

        # Level 2: Check hot source trees
        if source in self.hot_source_trees:
            dist, _ = self.hot_source_trees[source]
            if target < len(dist):
                self.hot_hits += 1
                return dist[target]

        return None

    def cache_result(self, source: int, target: int, distance: float):
        """
        Store a computed distance in the cache.

        The LRU cache automatically evicts old entries when full.
        """
        self.query_cache.put((source, target), distance)

    def update_hot_sources(self, source: int):
        """
        Check if a source should be promoted to "hot" status.

        When a source vertex is queried frequently, it becomes worthwhile
        to precompute its entire SSSP tree. This makes all future queries
        from this source instant (O(1) lookup instead of O(m + n log n)).

        The hot_threshold parameter controls this tradeoff.
        """
        if self.query_learner.should_precompute(source):
            if source not in self.hot_sources:
                self.hot_sources.add(source)
                # Invest in full SSSP computation now to benefit later
                self.hot_source_trees[source] = self._run_dijkstra(source)

    def get_features(self, source: int, target: int = -1) -> Dict:
        """
        Extract features for algorithm selection.

        The returned dictionary contains properties that help predict
        which algorithm will be fastest for this query:
        - Graph-level: size, density, weight distribution
        - Query-level: source/target degrees, estimated distance
        """
        features = {
            'n': self.n,
            'm': self.m,
            'avg_degree': self.avg_degree,
            'density': self.density,
            'max_weight': self.max_weight,
            'has_unit_weights': self.has_unit_weights,
            'has_small_weights': self.has_small_weights,
            'has_integer_weights': self.has_integer_weights,
            'query_type': 'point_to_point' if target >= 0 else 'sssp',
            'source_degree': self.offsets[source + 1] - self.offsets[source] if source < self.n else 0,
        }

        if target >= 0:
            features['target_degree'] = self.offsets[target + 1] - self.offsets[target] if target < self.n else 0
            features['estimated_distance'] = self.heuristic.h(source, target)

        return features

    def get_stats(self) -> Dict:
        """
        Get cache and performance statistics.

        Useful for monitoring and debugging. Returns metrics like:
        - Total queries processed
        - Cache hit rate (fraction answered from cache)
        - Number of hot sources
        """
        return {
            'query_count': self.query_count,
            'cache_hits': self.cache_hits,
            'hot_hits': self.hot_hits,
            'cache_size': len(self.query_cache),
            'hot_sources': len(self.hot_sources),
            'hit_rate': (self.cache_hits + self.hot_hits) / max(self.query_count, 1)
        }
