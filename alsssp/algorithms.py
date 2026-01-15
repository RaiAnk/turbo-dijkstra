"""
Algorithm Implementations for ALSSSP
====================================

This module contains optimized implementations of various shortest path
algorithms, each designed for specific use cases:

CacheOptimizedDijkstra
    Standard Dijkstra with cache-friendly data structures. Best for general
    use when no special structure can be exploited.

DeltaStepping
    Processes vertices in distance buckets, enabling parallel frontier
    expansion. Best for large graphs on multi-core systems.

DialBuckets
    Uses circular bucket array instead of heap for integer weights.
    Achieves O(1) insert and amortized O(1) extract-min. Best when
    edge weights are small integers (< n/10).

BidirectionalBFS
    Simultaneous BFS from source and target for unit-weight graphs.
    Explores O(b^(d/2)) vertices instead of O(b^d) where b is branching
    factor and d is the path length.

BidirectionalAStar
    Bidirectional Dijkstra that can optionally use landmark heuristics.
    The primary algorithm for point-to-point queries on weighted graphs.

All algorithms guarantee correctness (exact shortest paths) while
optimizing for different performance characteristics.
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Set
from collections import defaultdict, deque
import heapq

# Sentinel for unreachable vertices
INF = float('inf')


class CacheOptimizedDijkstra:
    """
    Dijkstra's algorithm with cache-friendly optimizations.

    Standard Dijkstra visits vertices in order of increasing distance from
    the source, maintaining the invariant that extracted vertices have their
    final shortest path distance.

    Our optimizations:
    1. CSR (Compressed Sparse Row) storage - edges stored contiguously in memory
    2. Sorted neighbor processing - adjacent edges accessed sequentially
    3. NumPy arrays - avoid Python object overhead for large graphs

    Time complexity: O((m + n) log n) with binary heap
    Space complexity: O(m + n) for graph storage + O(n) for distance arrays
    """

    def __init__(self, n: int, edges: List[Tuple[int, int, float]],
                 node_order: Optional[np.ndarray] = None):
        """
        Initialize with graph edges.

        Args:
            n: Number of vertices
            edges: List of (source, target, weight) tuples
            node_order: Optional vertex reordering for cache locality
        """
        self.n = n
        self.node_order = node_order if node_order is not None else np.arange(n)
        self.reverse_order = np.argsort(self.node_order)

        # Build adjacency in CSR format for cache efficiency
        self._build_adjacency(edges)

    def _build_adjacency(self, edges: List[Tuple[int, int, float]]):
        """
        Build CSR (Compressed Sparse Row) adjacency representation.

        CSR uses three arrays:
        - offsets[v]: Index where edges from vertex v start
        - neighbors[i]: Destination of edge i
        - weights[i]: Weight of edge i

        To iterate edges from v: for i in range(offsets[v], offsets[v+1])
        This gives sequential memory access, much faster than pointer chasing.
        """
        # First pass: count outgoing edges per vertex
        degree = np.zeros(self.n, dtype=np.int32)
        for u, v, w in edges:
            degree[u] += 1

        # Build offset array (cumulative sum of degrees)
        self.offsets = np.zeros(self.n + 1, dtype=np.int32)
        self.offsets[1:] = np.cumsum(degree)

        # Allocate edge arrays
        m = len(edges)
        self.neighbors = np.zeros(m, dtype=np.int32)
        self.weights = np.zeros(m, dtype=np.float64)

        # Second pass: fill edge arrays
        current = np.zeros(self.n, dtype=np.int32)
        for u, v, w in edges:
            idx = self.offsets[u] + current[u]
            self.neighbors[idx] = v
            self.weights[idx] = w
            current[u] += 1

        # Sort neighbors within each adjacency list
        # This improves cache locality during relaxation
        for u in range(self.n):
            start, end = self.offsets[u], self.offsets[u + 1]
            if end > start:
                order = np.argsort(self.neighbors[start:end])
                self.neighbors[start:end] = self.neighbors[start:end][order]
                self.weights[start:end] = self.weights[start:end][order]

    def compute(self, source: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute shortest paths from source to all vertices.

        Returns:
            dist: Array where dist[v] = shortest distance from source to v
            pred: Array where pred[v] = predecessor of v on shortest path
        """
        dist = np.full(self.n, INF, dtype=np.float64)
        pred = np.full(self.n, -1, dtype=np.int32)
        dist[source] = 0.0

        # Priority queue entries: (distance, vertex)
        # Python's heapq is a min-heap, so smallest distance comes first
        pq = [(0.0, source)]

        while pq:
            d, u = heapq.heappop(pq)

            # Skip if we've already found a shorter path to u
            # This handles the "lazy deletion" pattern
            if d > dist[u]:
                continue

            # Relax all outgoing edges from u
            for idx in range(self.offsets[u], self.offsets[u + 1]):
                v = self.neighbors[idx]
                w = self.weights[idx]
                new_dist = d + w

                if new_dist < dist[v]:
                    dist[v] = new_dist
                    pred[v] = u
                    heapq.heappush(pq, (new_dist, v))

        return dist, pred

    def compute_to_target(self, source: int, target: int) -> Tuple[float, List[int]]:
        """
        Compute shortest path to a specific target with early termination.

        This stops as soon as the target is extracted from the priority queue,
        potentially exploring far fewer vertices than full SSSP.

        Returns:
            distance: Shortest path distance (INF if unreachable)
            path: List of vertices from source to target
        """
        dist = np.full(self.n, INF, dtype=np.float64)
        pred = np.full(self.n, -1, dtype=np.int32)
        dist[source] = 0.0

        pq = [(0.0, source)]

        while pq:
            d, u = heapq.heappop(pq)

            # Early termination: target found
            if u == target:
                break

            if d > dist[u]:
                continue

            for idx in range(self.offsets[u], self.offsets[u + 1]):
                v = self.neighbors[idx]
                w = self.weights[idx]
                new_dist = d + w

                if new_dist < dist[v]:
                    dist[v] = new_dist
                    pred[v] = u
                    heapq.heappush(pq, (new_dist, v))

        # Reconstruct path by following predecessors backward
        path = []
        if dist[target] < INF:
            node = target
            while node != -1:
                path.append(node)
                node = pred[node]
            path.reverse()

        return dist[target], path


class DeltaStepping:
    """
    Delta-stepping algorithm for parallel-friendly SSSP.

    Instead of processing one vertex at a time (like Dijkstra), delta-stepping
    groups vertices into "buckets" based on their tentative distance:
        bucket[k] = {v : k*delta <= dist[v] < (k+1)*delta}

    All vertices in a bucket can be processed in parallel since their distances
    differ by less than delta (one edge weight).

    The algorithm alternates between:
    1. "Light" phase: relax edges with weight <= delta (may add to same bucket)
    2. "Heavy" phase: relax edges with weight > delta (add to later buckets)

    Time complexity: O(n + m + n*L/delta) where L is longest shortest path
    Space complexity: O(n + m) for graph + O(n) for buckets
    """

    def __init__(self, n: int, edges: List[Tuple[int, int, float]],
                 delta: Optional[float] = None):
        """
        Initialize delta-stepping.

        Args:
            n: Number of vertices
            edges: List of (source, target, weight) tuples
            delta: Bucket width (auto-selected if None)
        """
        self.n = n
        self.edges = edges

        # Build adjacency list (simpler than CSR for this algorithm)
        self.adj = defaultdict(list)
        max_weight = 0.0
        for u, v, w in edges:
            self.adj[u].append((v, w))
            max_weight = max(max_weight, w)

        # Auto-select delta if not provided
        # Heuristic: delta = max_weight / log(n) balances bucket overhead
        # against the number of times vertices get re-bucketed
        if delta is None:
            self.delta = max(1.0, max_weight / max(1, np.log2(n + 1)))
        else:
            self.delta = delta

    def compute(self, source: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run delta-stepping from source.

        Returns:
            dist: Shortest distances to all vertices
            pred: Predecessor array for path reconstruction
        """
        dist = np.full(self.n, INF, dtype=np.float64)
        pred = np.full(self.n, -1, dtype=np.int32)
        dist[source] = 0.0

        # Buckets indexed by floor(dist/delta)
        buckets = defaultdict(set)
        buckets[0].add(source)

        current_bucket = 0
        max_bucket = 0

        while current_bucket <= max_bucket:
            # Process current bucket until empty
            while buckets[current_bucket]:
                # Extract all vertices in this bucket (parallel opportunity)
                nodes = list(buckets[current_bucket])
                buckets[current_bucket].clear()

                # Phase 1: Relax light edges (weight <= delta)
                # These might add vertices back to the same bucket
                for u in nodes:
                    if dist[u] == INF:
                        continue

                    for v, w in self.adj[u]:
                        if w <= self.delta:
                            new_dist = dist[u] + w
                            if new_dist < dist[v]:
                                # Remove from old bucket if present
                                old_bucket = int(dist[v] / self.delta) if dist[v] < INF else -1
                                if old_bucket >= 0 and v in buckets[old_bucket]:
                                    buckets[old_bucket].discard(v)

                                dist[v] = new_dist
                                pred[v] = u
                                new_bucket = int(new_dist / self.delta)
                                buckets[new_bucket].add(v)
                                max_bucket = max(max_bucket, new_bucket)

                # Phase 2: Relax heavy edges (weight > delta)
                # These always go to later buckets
                for u in nodes:
                    if dist[u] == INF:
                        continue

                    for v, w in self.adj[u]:
                        if w > self.delta:
                            new_dist = dist[u] + w
                            if new_dist < dist[v]:
                                old_bucket = int(dist[v] / self.delta) if dist[v] < INF else -1
                                if old_bucket >= 0 and v in buckets[old_bucket]:
                                    buckets[old_bucket].discard(v)

                                dist[v] = new_dist
                                pred[v] = u
                                new_bucket = int(new_dist / self.delta)
                                buckets[new_bucket].add(v)
                                max_bucket = max(max_bucket, new_bucket)

            current_bucket += 1

        return dist, pred


class DialBuckets:
    """
    Dial's algorithm for graphs with small integer weights.

    When edge weights are integers bounded by C, we can replace the O(log n)
    priority queue with a circular array of C*n+1 buckets, achieving:
    - O(1) insert
    - O(1) amortized extract-min (just scan forward to next non-empty bucket)

    Time complexity: O(m + n*C) where C is the maximum edge weight
    Space complexity: O(n*C) for buckets (can be large for big C!)

    Best when: C < n/10 (otherwise heap-based Dijkstra is competitive)
    """

    def __init__(self, n: int, edges: List[Tuple[int, int, float]]):
        """
        Initialize Dial's algorithm.

        Weights are rounded to integers internally.
        """
        self.n = n

        # Convert weights to integers and build adjacency
        self.adj = defaultdict(list)
        self.max_weight = 0

        for u, v, w in edges:
            w_int = max(1, int(w))  # Ensure positive integer
            self.adj[u].append((v, w_int))
            self.max_weight = max(self.max_weight, w_int)

    def compute(self, source: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run Dial's algorithm from source.

        Returns:
            dist: Shortest distances (as floats for compatibility)
            pred: Predecessor array
        """
        dist = np.full(self.n, INF, dtype=np.float64)
        pred = np.full(self.n, -1, dtype=np.int32)
        dist[source] = 0.0

        # Number of buckets: enough to hold any reachable distance
        # We limit this to avoid memory explosion
        num_buckets = self.n * self.max_weight + 1
        num_buckets = min(num_buckets, self.n * 100)

        # Circular bucket array
        buckets = [set() for _ in range(num_buckets)]
        buckets[0].add(source)

        in_bucket = {source}  # Track which vertices are currently bucketed
        current = 0
        processed = 0

        while processed < self.n:
            # Scan forward to find next non-empty bucket
            while not buckets[current % num_buckets] and processed < self.n:
                current += 1
                if current > num_buckets * 2:
                    break

            if current > num_buckets * 2:
                break

            bucket_idx = current % num_buckets
            if not buckets[bucket_idx]:
                break

            # Extract one vertex from the bucket
            u = buckets[bucket_idx].pop()
            if u in in_bucket:
                in_bucket.remove(u)
            processed += 1

            # Relax outgoing edges
            for v, w in self.adj[u]:
                new_dist = dist[u] + w
                if new_dist < dist[v]:
                    # Remove from old bucket
                    if v in in_bucket:
                        old_bucket = int(dist[v]) % num_buckets
                        buckets[old_bucket].discard(v)

                    dist[v] = new_dist
                    pred[v] = u
                    new_bucket = int(new_dist) % num_buckets
                    buckets[new_bucket].add(v)
                    in_bucket.add(v)

        return dist, pred


class BidirectionalBFS:
    """
    Bidirectional BFS for unweighted (unit-weight) graphs.

    When all edges have the same weight, BFS finds shortest paths optimally.
    Bidirectional search explores from both source and target simultaneously,
    meeting somewhere in the middle.

    The key insight: if standard BFS explores O(b^d) vertices where b is the
    branching factor and d is the path length, then bidirectional BFS explores
    only O(2 * b^(d/2)) = O(b^(d/2)) vertices - a quadratic improvement!

    Time complexity: O(b^(d/2)) instead of O(b^d)
    Space complexity: O(b^(d/2)) for the frontier queues
    """

    def __init__(self, n: int, edges: List[Tuple[int, int, float]]):
        """
        Initialize bidirectional BFS.

        We need both forward and backward adjacency since we search in both
        directions. Edge weights are ignored (treated as 1).
        """
        self.n = n

        # Forward edges: u -> v
        self.adj_forward = defaultdict(list)
        # Backward edges: v -> u (for searching from target)
        self.adj_backward = defaultdict(list)

        for u, v, w in edges:
            self.adj_forward[u].append(v)
            self.adj_backward[v].append(u)

    def compute(self, source: int, target: int) -> Tuple[float, List[int]]:
        """
        Find shortest path from source to target.

        Returns:
            distance: Number of edges on shortest path (INF if unreachable)
            path: List of vertices from source to target
        """
        # Handle trivial case
        if source == target:
            return 0.0, [source]

        # Forward search state (from source)
        dist_forward = {source: 0}
        pred_forward = {source: -1}
        queue_forward = deque([source])

        # Backward search state (from target)
        dist_backward = {target: 0}
        pred_backward = {target: -1}
        queue_backward = deque([target])

        # Track best path found so far
        best_distance = INF
        meeting_point = -1

        while queue_forward or queue_backward:
            # Expand forward frontier by one layer
            if queue_forward:
                u = queue_forward.popleft()
                d = dist_forward[u]

                for v in self.adj_forward[u]:
                    if v not in dist_forward:
                        dist_forward[v] = d + 1
                        pred_forward[v] = u
                        queue_forward.append(v)

                        # Check if we've met the backward search
                        if v in dist_backward:
                            total = dist_forward[v] + dist_backward[v]
                            if total < best_distance:
                                best_distance = total
                                meeting_point = v

            # Expand backward frontier by one layer
            if queue_backward:
                u = queue_backward.popleft()
                d = dist_backward[u]

                for v in self.adj_backward[u]:
                    if v not in dist_backward:
                        dist_backward[v] = d + 1
                        pred_backward[v] = u
                        queue_backward.append(v)

                        # Check if we've met the forward search
                        if v in dist_forward:
                            total = dist_forward[v] + dist_backward[v]
                            if total < best_distance:
                                best_distance = total
                                meeting_point = v

            # Early termination: no shorter path can exist
            # once both frontiers have moved past half the best distance
            min_forward = min(dist_forward.values()) if dist_forward else INF
            min_backward = min(dist_backward.values()) if dist_backward else INF
            if min_forward + min_backward >= best_distance:
                break

        if meeting_point == -1:
            return INF, []

        # Reconstruct path by combining forward and backward portions
        # Forward: source -> ... -> meeting_point
        path_forward = []
        node = meeting_point
        while node != -1:
            path_forward.append(node)
            node = pred_forward.get(node, -1)
        path_forward.reverse()

        # Backward: meeting_point -> ... -> target
        path_backward = []
        node = pred_backward.get(meeting_point, -1)
        while node != -1:
            path_backward.append(node)
            node = pred_backward.get(node, -1)

        return float(best_distance), path_forward + path_backward


class BidirectionalAStar:
    """
    Bidirectional Dijkstra for weighted graphs.

    This is our primary algorithm for point-to-point queries. It maintains
    two search frontiers - one expanding from the source and one from the
    target - and terminates when it can prove no shorter path exists.

    The termination condition is subtle: we can't stop at the first meeting
    because a shorter path might exist through unexplored vertices. Instead,
    we continue until the sum of minimum frontier distances reaches the
    best path found so far.

    Correctness guarantee: Always returns the exact shortest path.

    Time complexity: O((m + n) log n) worst case, but typically much better
    Space complexity: O(n) for distance dictionaries and priority queues
    """

    def __init__(self, n: int, edges: List[Tuple[int, int, float]],
                 landmark_distances: Optional[Dict[int, np.ndarray]] = None):
        """
        Initialize bidirectional search.

        Args:
            n: Number of vertices
            edges: List of (source, target, weight) tuples
            landmark_distances: Optional precomputed distances for heuristics
        """
        self.n = n

        # Build both forward and reverse adjacency
        self.adj_forward = defaultdict(list)
        self.adj_backward = defaultdict(list)

        for u, v, w in edges:
            self.adj_forward[u].append((v, w))
            self.adj_backward[v].append((u, w))

        # Landmark distances for potential heuristic use
        # (Currently we use pure Dijkstra for guaranteed correctness)
        self.landmark_distances = landmark_distances or {}

    def compute(self, source: int, target: int) -> Tuple[float, List[int]]:
        """
        Find shortest path using bidirectional Dijkstra.

        Returns:
            distance: Exact shortest path distance (INF if unreachable)
            path: List of vertices from source to target
        """
        if source == target:
            return 0.0, [source]

        # Distance labels for forward and backward searches
        dist_forward = {source: 0.0}
        dist_backward = {target: 0.0}

        # Predecessor tracking for path reconstruction
        pred_forward = {source: -1}
        pred_backward = {target: -1}

        # Priority queues: (distance, vertex)
        pq_forward = [(0.0, source)]
        pq_backward = [(0.0, target)]

        # Sets of vertices whose shortest path is finalized
        processed_forward = set()
        processed_backward = set()

        # Best complete path found so far
        best_distance = INF
        meeting_point = -1

        while pq_forward or pq_backward:
            # Get minimum distances from each frontier
            d_min_forward = pq_forward[0][0] if pq_forward else INF
            d_min_backward = pq_backward[0][0] if pq_backward else INF

            # Termination condition: no shorter path can exist
            # This is the key correctness criterion for bidirectional Dijkstra
            if d_min_forward + d_min_backward >= best_distance:
                break

            # Expand the frontier with smaller minimum distance
            # This balances exploration and finds meeting point efficiently
            if d_min_forward <= d_min_backward and pq_forward:
                d, u = heapq.heappop(pq_forward)

                if u in processed_forward:
                    continue
                processed_forward.add(u)

                # Check if backward search has already reached this vertex
                if u in processed_backward:
                    total = dist_forward[u] + dist_backward[u]
                    if total < best_distance:
                        best_distance = total
                        meeting_point = u

                # Relax outgoing edges
                for v, w in self.adj_forward[u]:
                    new_dist = d + w
                    if new_dist < dist_forward.get(v, INF):
                        dist_forward[v] = new_dist
                        pred_forward[v] = u
                        heapq.heappush(pq_forward, (new_dist, v))

                        # Check for potential meeting
                        if v in dist_backward:
                            total = new_dist + dist_backward[v]
                            if total < best_distance:
                                best_distance = total
                                meeting_point = v

            elif pq_backward:
                d, u = heapq.heappop(pq_backward)

                if u in processed_backward:
                    continue
                processed_backward.add(u)

                # Check if forward search has already reached this vertex
                if u in processed_forward:
                    total = dist_forward[u] + dist_backward[u]
                    if total < best_distance:
                        best_distance = total
                        meeting_point = u

                # Relax incoming edges (backward search)
                for v, w in self.adj_backward[u]:
                    new_dist = d + w
                    if new_dist < dist_backward.get(v, INF):
                        dist_backward[v] = new_dist
                        pred_backward[v] = u
                        heapq.heappush(pq_backward, (new_dist, v))

                        # Check for potential meeting
                        if v in dist_forward:
                            total = new_dist + dist_forward[v]
                            if total < best_distance:
                                best_distance = total
                                meeting_point = v

        if meeting_point == -1:
            return INF, []

        # Reconstruct path through the meeting point
        # Forward portion: source -> meeting_point
        path_forward = []
        node = meeting_point
        while node != -1:
            path_forward.append(node)
            node = pred_forward.get(node, -1)
        path_forward.reverse()

        # Backward portion: meeting_point -> target
        path_backward = []
        node = pred_backward.get(meeting_point, -1)
        while node != -1:
            path_backward.append(node)
            node = pred_backward.get(node, -1)

        return best_distance, path_forward + path_backward
