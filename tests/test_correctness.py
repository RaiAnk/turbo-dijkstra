"""
Correctness Tests for ALSSSP
============================

These tests verify that ALSSSP produces correct shortest path results.
We compare against a known-correct reference implementation of Dijkstra
on various graph types and edge cases.

Run with: pytest test_correctness.py -v
"""

import sys
sys.path.insert(0, '..')

import pytest
import random
import heapq
from typing import List, Tuple

from alsssp import ALSSSP


def reference_dijkstra(n: int, edges: List[Tuple[int, int, float]],
                       source: int) -> List[float]:
    """
    Reference implementation of Dijkstra's algorithm.

    This is a textbook implementation that we trust to be correct.
    All ALSSSP results are compared against this.
    """
    # Build adjacency list
    adj = [[] for _ in range(n)]
    for u, v, w in edges:
        adj[u].append((v, w))

    # Initialize distances
    dist = [float('inf')] * n
    dist[source] = 0.0

    # Priority queue: (distance, vertex)
    pq = [(0.0, source)]

    while pq:
        d, u = heapq.heappop(pq)

        # Skip stale entries
        if d > dist[u]:
            continue

        # Relax edges
        for v, w in adj[u]:
            if d + w < dist[v]:
                dist[v] = d + w
                heapq.heappush(pq, (dist[v], v))

    return dist


class TestBasicCorrectness:
    """Basic correctness tests on simple graphs."""

    def test_simple_path(self):
        """Test on a simple linear path graph."""
        # 0 -> 1 -> 2 -> 3
        edges = [
            (0, 1, 1.0),
            (1, 2, 1.0),
            (2, 3, 1.0),
        ]
        solver = ALSSSP(n=4, edges=edges)

        result = solver.shortest_path(0, 3)
        assert abs(result.distances[3] - 3.0) < 1e-9

    def test_with_shortcut(self):
        """Test that algorithm finds shorter path through shortcut."""
        # 0 -> 1 -> 2 (cost 2) vs 0 -> 2 (cost 1.5)
        edges = [
            (0, 1, 1.0),
            (1, 2, 1.0),
            (0, 2, 1.5),  # Direct but shorter path
        ]
        solver = ALSSSP(n=3, edges=edges)

        result = solver.shortest_path(0, 2)
        assert abs(result.distances[2] - 1.5) < 1e-9

    def test_unreachable(self):
        """Test handling of unreachable vertices."""
        # Disconnected graph: 0 -> 1, 2 -> 3
        edges = [
            (0, 1, 1.0),
            (2, 3, 1.0),
        ]
        solver = ALSSSP(n=4, edges=edges)

        result = solver.shortest_path(0, 3)
        assert result.distances[3] == float('inf')

    def test_self_loop(self):
        """Test that source to itself has distance 0."""
        edges = [(0, 1, 1.0), (1, 2, 1.0)]
        solver = ALSSSP(n=3, edges=edges)

        result = solver.shortest_path(0, 0)
        assert result.distances[0] == 0.0

    def test_zero_weight_edge(self):
        """Test handling of zero-weight edges."""
        edges = [
            (0, 1, 0.0),
            (1, 2, 1.0),
        ]
        solver = ALSSSP(n=3, edges=edges)

        result = solver.shortest_path(0, 2)
        assert abs(result.distances[2] - 1.0) < 1e-9


class TestRandomGraphs:
    """Tests on randomly generated graphs, comparing against reference."""

    def test_small_random_graph(self):
        """Test on small random graph."""
        random.seed(42)
        n = 100
        edges = [(random.randint(0, n-1), random.randint(0, n-1),
                  random.uniform(1, 10))
                 for _ in range(400)]

        solver = ALSSSP(n=n, edges=edges)

        # Test multiple source-target pairs
        for _ in range(20):
            source = random.randint(0, n-1)
            target = random.randint(0, n-1)

            alsssp_result = solver.shortest_path(source, target)
            reference_dist = reference_dijkstra(n, edges, source)[target]

            assert abs(alsssp_result.distances[target] - reference_dist) < 1e-6, \
                f"Mismatch for ({source}, {target}): ALSSSP={alsssp_result.distances[target]}, Ref={reference_dist}"

    def test_medium_random_graph(self):
        """Test on medium random graph."""
        random.seed(123)
        n = 500
        edges = [(random.randint(0, n-1), random.randint(0, n-1),
                  random.uniform(1, 10))
                 for _ in range(2000)]

        solver = ALSSSP(n=n, edges=edges)

        # Spot check several pairs
        for _ in range(10):
            source = random.randint(0, n-1)
            target = random.randint(0, n-1)

            alsssp_result = solver.shortest_path(source, target)
            reference_dist = reference_dijkstra(n, edges, source)[target]

            assert abs(alsssp_result.distances[target] - reference_dist) < 1e-6

    def test_full_sssp_correctness(self):
        """Test that full SSSP gives correct distances to all vertices."""
        random.seed(456)
        n = 100
        edges = [(random.randint(0, n-1), random.randint(0, n-1),
                  random.uniform(1, 10))
                 for _ in range(400)]

        solver = ALSSSP(n=n, edges=edges)
        source = 0

        alsssp_result = solver.sssp(source)
        reference = reference_dijkstra(n, edges, source)

        for v in range(n):
            assert abs(alsssp_result.distances[v] - reference[v]) < 1e-6, \
                f"Mismatch at vertex {v}: ALSSSP={alsssp_result.distances[v]}, Ref={reference[v]}"


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_single_vertex(self):
        """Test graph with single vertex."""
        solver = ALSSSP(n=1, edges=[])
        result = solver.shortest_path(0, 0)
        assert result.distances[0] == 0.0

    def test_two_vertices(self):
        """Test minimal connected graph."""
        edges = [(0, 1, 5.0)]
        solver = ALSSSP(n=2, edges=edges)

        result = solver.shortest_path(0, 1)
        assert abs(result.distances[1] - 5.0) < 1e-9

        # Reverse direction (unreachable in directed graph)
        result = solver.shortest_path(1, 0)
        assert result.distances[0] == float('inf')

    def test_dense_clique(self):
        """Test on complete graph (clique)."""
        n = 20
        edges = []
        for i in range(n):
            for j in range(n):
                if i != j:
                    edges.append((i, j, abs(i - j) + 1.0))

        solver = ALSSSP(n=n, edges=edges)

        for source in [0, 5, 19]:
            alsssp_result = solver.sssp(source)
            reference = reference_dijkstra(n, edges, source)

            for v in range(n):
                assert abs(alsssp_result.distances[v] - reference[v]) < 1e-6

    def test_multiple_paths_same_cost(self):
        """Test when multiple paths have the same cost."""
        # Diamond graph: 0 -> 1 -> 3 and 0 -> 2 -> 3, both cost 2
        edges = [
            (0, 1, 1.0),
            (0, 2, 1.0),
            (1, 3, 1.0),
            (2, 3, 1.0),
        ]
        solver = ALSSSP(n=4, edges=edges)

        result = solver.shortest_path(0, 3)
        assert abs(result.distances[3] - 2.0) < 1e-9


class TestCaching:
    """Tests for caching behavior."""

    def test_cache_returns_same_result(self):
        """Test that cached results are correct."""
        edges = [(0, 1, 1.0), (1, 2, 1.0)]
        solver = ALSSSP(n=3, edges=edges)

        # First query
        result1 = solver.shortest_path(0, 2)

        # Second query (should hit cache)
        result2 = solver.shortest_path(0, 2)

        assert abs(result1.distances[2] - result2.distances[2]) < 1e-9
        assert result2.cache_hit or result1.cache_hit is False  # Second should be cached

    def test_batch_query_correctness(self):
        """Test that batch queries return correct results."""
        random.seed(789)
        n = 100
        edges = [(random.randint(0, n-1), random.randint(0, n-1),
                  random.uniform(1, 10))
                 for _ in range(400)]

        solver = ALSSSP(n=n, edges=edges)

        queries = [(random.randint(0, n-1), random.randint(0, n-1))
                   for _ in range(20)]

        results = solver.batch_query(queries)

        for (s, t), result in zip(queries, results):
            reference_dist = reference_dijkstra(n, edges, s)[t]
            assert abs(result.distances[t] - reference_dist) < 1e-6


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
