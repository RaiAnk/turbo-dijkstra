"""
Online Learning Components for ALSSSP
=====================================

This module implements the learning mechanisms that enable ALSSSP to improve
its performance over time. Unlike offline machine learning that requires
training data upfront, these components learn incrementally from each query.

Components:

LRUCache
    Simple least-recently-used cache for storing query results.

QueryPatternLearner
    Tracks query patterns to identify vertices that should have precomputed
    SSSP trees ("hot" sources) and pairs that should be directly cached.

AlgorithmSelector
    Learns which algorithm performs best for different query types based on
    features like graph size, density, and weight distribution.

LearnedHeuristic
    Refines A* search heuristics based on observed shortest path distances,
    making the heuristics tighter over time.

The learning is designed to be:
- Lightweight: minimal overhead per query
- Robust: graceful degradation if predictions are wrong
- Adaptive: continuous improvement without manual tuning
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Set
from collections import defaultdict, Counter
import heapq
import time

# Sentinel for unreachable vertices
INF = float('inf')


class LRUCache:
    """
    Least-Recently-Used cache for query results.

    This is a simple implementation that tracks access times and evicts
    the oldest entries when capacity is exceeded. For production use,
    consider Python's functools.lru_cache or a dedicated library.

    The cache maps (source, target) tuples to shortest path distances.
    """

    def __init__(self, maxsize: int = 100000):
        """
        Initialize cache with maximum capacity.

        Args:
            maxsize: Maximum number of entries to store
        """
        self.maxsize = maxsize
        self.cache: Dict[Tuple[int, int], float] = {}
        self.access_order: Dict[Tuple[int, int], float] = {}
        self.time = 0  # Logical timestamp for LRU ordering

    def get(self, key: Tuple[int, int]) -> Optional[float]:
        """
        Retrieve a cached value, updating access time.

        Returns None if key is not in cache.
        """
        if key in self.cache:
            self.time += 1
            self.access_order[key] = self.time
            return self.cache[key]
        return None

    def put(self, key: Tuple[int, int], value: float):
        """
        Store a value in the cache.

        If the cache is full, evicts least-recently-used entries.
        """
        self.time += 1
        self.cache[key] = value
        self.access_order[key] = self.time

        # Trigger eviction if over capacity
        if len(self.cache) > self.maxsize:
            self._evict()

    def _evict(self):
        """
        Remove oldest entries to bring cache under capacity.

        We evict 20% of entries at once to avoid frequent eviction overhead.
        """
        if len(self.cache) <= self.maxsize * 0.9:
            return

        # Sort by access time (oldest first)
        items = sorted(self.access_order.items(), key=lambda x: x[1])
        to_remove = len(self.cache) - int(self.maxsize * 0.8)

        for key, _ in items[:to_remove]:
            del self.cache[key]
            del self.access_order[key]

    def __contains__(self, key: Tuple[int, int]) -> bool:
        """Check if key is in cache (without updating access time)."""
        return key in self.cache

    def __len__(self) -> int:
        """Return current number of cached entries."""
        return len(self.cache)


class QueryPatternLearner:
    """
    Learns query patterns to optimize caching and precomputation.

    Real workloads often exhibit structure:
    - Some sources are queried much more often than others
    - Some source-target pairs repeat frequently
    - Recent queries may predict near-future queries

    This class tracks these patterns and provides recommendations for
    what to cache or precompute.
    """

    def __init__(self, hot_threshold: int = 10):
        """
        Initialize pattern learner.

        Args:
            hot_threshold: Number of queries before a source is considered "hot"
        """
        # Historical record of queries (bounded to avoid memory growth)
        self.query_history: List[Tuple[int, int, float, float]] = []
        self.max_history = 1000

        # Frequency counters
        self.source_frequency: Counter = Counter()
        self.target_frequency: Counter = Counter()
        self.pair_frequency: Counter = Counter()

        # Hot source threshold
        self.hot_threshold = hot_threshold

        # Recent queries for temporal pattern detection
        self.recent_sources: List[int] = []
        self.recent_targets: List[int] = []

    def record_query(self, source: int, target: int, distance: float, time_taken: float):
        """
        Record a completed query for pattern learning.

        This should be called after every query to keep the learner updated.
        """
        # Update frequency counters
        self.source_frequency[source] += 1
        self.target_frequency[target] += 1

        if target >= 0:
            self.pair_frequency[(source, target)] += 1

        # Maintain bounded history
        if len(self.query_history) < self.max_history:
            self.query_history.append((source, target, distance, time_taken))
        else:
            # Rolling window: drop oldest 10%
            self.query_history = self.query_history[100:] + [(source, target, distance, time_taken)]

        # Track recent queries for temporal patterns
        self.recent_sources.append(source)
        self.recent_targets.append(target)
        if len(self.recent_sources) > 100:
            self.recent_sources = self.recent_sources[-100:]
            self.recent_targets = self.recent_targets[-100:]

    def get_hot_sources(self) -> Set[int]:
        """
        Get source vertices that warrant precomputed SSSP trees.

        A source is "hot" if it's been queried at least hot_threshold times.
        The full SSSP tree from a hot source enables O(1) distance lookups
        for any target.
        """
        return {s for s, count in self.source_frequency.items()
                if count >= self.hot_threshold}

    def get_hot_targets(self) -> Set[int]:
        """
        Get frequently queried target vertices.

        These could benefit from reverse SSSP tree precomputation in
        future versions.
        """
        return {t for t, count in self.target_frequency.items()
                if count >= self.hot_threshold}

    def get_frequent_pairs(self, k: int = 100) -> List[Tuple[int, int]]:
        """
        Get the k most frequently queried source-target pairs.

        These are prime candidates for direct caching.
        """
        return [pair for pair, _ in self.pair_frequency.most_common(k)]

    def predict_next_queries(self, k: int = 10) -> List[Tuple[int, int]]:
        """
        Predict likely upcoming queries based on recent patterns.

        This could be used for prefetching in future versions.
        """
        recent_pairs = []
        for i, s in enumerate(self.recent_sources):
            t = self.recent_targets[i]
            if t >= 0:
                recent_pairs.append((s, t))

        pair_counts = Counter(recent_pairs)
        return [pair for pair, _ in pair_counts.most_common(k)]

    def should_precompute(self, node: int) -> bool:
        """
        Determine if a node's SSSP tree should be precomputed.

        Returns True if the node has been used as a source at least
        hot_threshold times.
        """
        return self.source_frequency[node] >= self.hot_threshold

    def get_frequent_nodes(self, top_k: int = 10) -> List[int]:
        """
        Get nodes that appear most often in queries (as source or target).
        """
        all_nodes = Counter()
        all_nodes.update(self.source_frequency)
        all_nodes.update(self.target_frequency)
        return [node for node, _ in all_nodes.most_common(top_k)]


class AlgorithmSelector:
    """
    Learns which algorithm performs best for different query types.

    The key insight is that different algorithms have different performance
    characteristics depending on:
    - Graph size and density
    - Weight distribution (unit, integer, continuous)
    - Query type (SSSP vs point-to-point)
    - Source/target properties

    This class maintains performance estimates per algorithm per feature
    bucket and uses them to select the predicted fastest algorithm.
    """

    # Available algorithms in the portfolio
    ALGORITHMS = [
        'dijkstra',
        'delta_stepping',
        'dial',
        'bidirectional_bfs',
        'bidirectional_astar',
        'cached_lookup'
    ]

    def __init__(self, learning_rate: float = 0.1):
        """
        Initialize algorithm selector.

        Args:
            learning_rate: How quickly to adapt to new observations (0-1)
        """
        self.learning_rate = learning_rate

        # Performance estimates: algorithm -> feature_hash -> (avg_time, count)
        # We discretize the feature space to enable generalization
        self.performance: Dict[str, Dict[int, Tuple[float, int]]] = {
            algo: {} for algo in self.ALGORITHMS
        }

        # Global averages as fallback for unseen feature combinations
        self.global_avg: Dict[str, Tuple[float, int]] = {
            algo: (0.001, 1) for algo in self.ALGORITHMS
        }

        self.total_queries = 0

    def _hash_features(self, features: Dict) -> int:
        """
        Hash features into discrete buckets.

        This enables generalization: queries with similar features will
        share performance estimates even if not identical.
        """
        n = features.get('n', 0)
        m = features.get('m', 0)
        density = m / max(n, 1)
        query_type = features.get('query_type', 'sssp')
        has_small_weights = features.get('has_small_weights', False)

        # Discretize into buckets
        n_bucket = min(int(np.log2(n + 1)), 20)
        density_bucket = min(int(density), 20)
        type_bucket = 1 if query_type == 'point_to_point' else 0
        weight_bucket = 1 if has_small_weights else 0

        # Combine into single hash value
        return n_bucket * 1000 + density_bucket * 50 + type_bucket * 2 + weight_bucket

    def record_performance(self, algorithm: str, features: Dict, time_taken: float):
        """
        Record observed performance for a completed query.

        Uses exponential moving average to adapt to changing conditions
        while not forgetting historical performance entirely.
        """
        if algorithm not in self.ALGORITHMS:
            return

        feature_hash = self._hash_features(features)

        # Update bucket-specific estimate
        if feature_hash in self.performance[algorithm]:
            old_avg, count = self.performance[algorithm][feature_hash]
            new_avg = old_avg * (1 - self.learning_rate) + time_taken * self.learning_rate
            self.performance[algorithm][feature_hash] = (new_avg, count + 1)
        else:
            self.performance[algorithm][feature_hash] = (time_taken, 1)

        # Update global estimate
        old_avg, count = self.global_avg[algorithm]
        new_avg = old_avg * (1 - self.learning_rate) + time_taken * self.learning_rate
        self.global_avg[algorithm] = (new_avg, count + 1)

        self.total_queries += 1

    def predict_best_algorithm(self, features: Dict) -> str:
        """
        Predict which algorithm will be fastest for these features.

        Combines learned performance estimates with heuristic rules.
        """
        feature_hash = self._hash_features(features)

        # Get performance predictions for each algorithm
        predictions = {}
        for algo in self.ALGORITHMS:
            if feature_hash in self.performance[algo]:
                avg, count = self.performance[algo][feature_hash]
                predictions[algo] = avg
            else:
                # Fall back to global average with uncertainty penalty
                avg, count = self.global_avg[algo]
                predictions[algo] = avg * 1.5

        # Apply domain knowledge (strong heuristics that should override learning)
        query_type = features.get('query_type', 'sssp')
        has_small_weights = features.get('has_small_weights', False)
        n = features.get('n', 0)

        # Bidirectional is almost always best for point-to-point
        if query_type == 'point_to_point':
            predictions['bidirectional_astar'] *= 0.8
            predictions['bidirectional_bfs'] *= 0.7 if features.get('has_unit_weights') else 1.5

        # Dial's algorithm excels with small integer weights
        if has_small_weights and features.get('max_weight', INF) < n:
            predictions['dial'] *= 0.7

        # Dijkstra has low overhead for small graphs
        if n < 1000:
            predictions['dijkstra'] *= 0.9

        return min(predictions, key=predictions.get)

    def has_enough_data(self) -> bool:
        """
        Check if we have sufficient data for reliable predictions.

        We require at least 50 queries to start trusting the learned model.
        """
        return self.total_queries >= 50


class LearnedHeuristic:
    """
    Learns improved heuristics for A* search.

    A* search uses a heuristic h(u, target) that estimates the distance
    from u to the target. Better heuristics lead to faster search by
    guiding exploration toward the target.

    We use landmarks (precomputed distances from selected vertices) as the
    base heuristic, then refine it using observations from actual queries.

    The landmark heuristic uses the triangle inequality:
        |dist(u, L) - dist(target, L)| <= dist(u, target)

    Taking the max over all landmarks gives an admissible lower bound.
    """

    def __init__(self, n: int, num_landmarks: int = 16):
        """
        Initialize learned heuristic.

        Args:
            n: Number of vertices in graph
            num_landmarks: Number of landmark vertices to use
        """
        self.n = n
        # Scale landmarks with graph size, but not too many
        self.num_landmarks = min(num_landmarks, max(1, n // 100))

        # Landmark data
        self.landmarks: List[int] = []
        self.landmark_distances: Dict[int, np.ndarray] = {}

        # Learned corrections to improve heuristic tightness
        # Indexed by (source_bucket, target_bucket) for generalization
        self.correction_sum: Dict[Tuple[int, int], float] = defaultdict(float)
        self.correction_count: Dict[Tuple[int, int], int] = defaultdict(int)

    def select_landmarks(self, adj: Dict[int, List[Tuple[int, float]]]) -> List[int]:
        """
        Select diverse landmark vertices.

        Ideally we'd use farthest-first traversal, but for simplicity we
        just spread landmarks evenly across vertex IDs. This works well
        when vertices are BFS-ordered.
        """
        if self.n == 0:
            return []

        landmarks = [0]  # Always include vertex 0
        landmark_set = {0}

        # Spread remaining landmarks across the graph
        step = max(1, self.n // self.num_landmarks)
        for i in range(step, self.n, step):
            if len(landmarks) >= self.num_landmarks:
                break
            if i not in landmark_set:
                landmarks.append(i)
                landmark_set.add(i)

        self.landmarks = landmarks
        return landmarks

    def precompute_distances(self, dijkstra_func):
        """
        Precompute distances from all landmarks.

        Args:
            dijkstra_func: Function that runs Dijkstra from a source
        """
        for landmark in self.landmarks:
            dist, _ = dijkstra_func(landmark)
            self.landmark_distances[landmark] = dist

    def h(self, u: int, target: int) -> float:
        """
        Compute heuristic estimate from u to target.

        Uses triangle inequality with landmarks, plus learned corrections.
        """
        if not self.landmark_distances:
            return 0.0

        # Base heuristic: max over landmarks using triangle inequality
        h_val = 0.0
        for landmark, distances in self.landmark_distances.items():
            if u < len(distances) and target < len(distances):
                diff = abs(distances[u] - distances[target])
                h_val = max(h_val, diff)

        # Apply learned correction (if we have enough data)
        bucket_u = u // 100
        bucket_t = target // 100
        key = (bucket_u, bucket_t)

        if self.correction_count[key] > 5:
            avg_correction = self.correction_sum[key] / self.correction_count[key]
            # Only add positive corrections to stay admissible
            if avg_correction > 0:
                # Be conservative to maintain admissibility
                h_val += min(avg_correction * 0.5, h_val * 0.2)

        return h_val

    def update_from_query(self, source: int, target: int, actual_distance: float):
        """
        Update heuristic based on observed distance.

        After each query, we compare our heuristic estimate to the actual
        distance and use the error to improve future estimates.
        """
        predicted = self.h(source, target)
        error = actual_distance - predicted  # Should be >= 0 for admissibility

        # Update correction for this region of the graph
        bucket_s = source // 100
        bucket_t = target // 100
        key = (bucket_s, bucket_t)

        self.correction_sum[key] += error
        self.correction_count[key] += 1

    def get_landmark_distances(self) -> Dict[int, np.ndarray]:
        """
        Get precomputed landmark distances.

        Used by BidirectionalAStar for heuristic computation.
        """
        return self.landmark_distances
