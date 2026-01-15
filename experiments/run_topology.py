"""
Topology Comparison Experiment
==============================

This script compares ALSSSP performance across different graph topologies.
Different real-world networks have different structures:
- Road networks: sparse, planar, low degree
- Social networks: small-world, high clustering
- Web graphs: power-law degree distribution
- Grid networks: regular structure

We test on synthetic graphs that mimic these structures to understand
where ALSSSP provides the most benefit.

Usage:
    python run_topology.py
"""

import sys
sys.path.insert(0, '..')

import time
import random
import heapq
import math
from typing import List, Tuple, Dict

# Optional plotting
try:
    import matplotlib.pyplot as plt
    import numpy as np
    HAS_PLOTTING = True
except ImportError:
    HAS_PLOTTING = False

from alsssp import ALSSSP


def generate_grid_graph(rows: int, cols: int) -> Tuple[int, List[Tuple[int, int, float]]]:
    """
    Generate a 2D grid graph.

    Grid graphs are commonly used to model road networks and spatial data.
    Each vertex is connected to its 4-neighbors (up, down, left, right).
    """
    n = rows * cols
    edges = []

    for r in range(rows):
        for c in range(cols):
            u = r * cols + c

            # Connect to right neighbor
            if c + 1 < cols:
                v = r * cols + (c + 1)
                w = random.uniform(1.0, 3.0)
                edges.append((u, v, w))
                edges.append((v, u, w))

            # Connect to bottom neighbor
            if r + 1 < rows:
                v = (r + 1) * cols + c
                w = random.uniform(1.0, 3.0)
                edges.append((u, v, w))
                edges.append((v, u, w))

    return n, edges


def generate_random_graph(n: int, avg_degree: float = 4.0) -> List[Tuple[int, int, float]]:
    """
    Generate a random (Erdos-Renyi style) graph.

    Random graphs have uniform degree distribution and short average
    path lengths. They serve as a baseline topology.
    """
    edges = []
    m = int(n * avg_degree)

    for _ in range(m):
        u = random.randint(0, n - 1)
        v = random.randint(0, n - 1)
        if u != v:
            w = random.uniform(1.0, 10.0)
            edges.append((u, v, w))

    return edges


def generate_power_law_graph(n: int, avg_degree: float = 4.0) -> List[Tuple[int, int, float]]:
    """
    Generate a graph with power-law degree distribution.

    Power-law graphs model social networks and web graphs where a few
    nodes have very high degree (hubs) while most have low degree.

    We use preferential attachment: new edges are more likely to connect
    to high-degree nodes.
    """
    # Start with a small clique
    edges = []
    degrees = [0] * n

    # Initial clique of 5 vertices
    for i in range(min(5, n)):
        for j in range(i + 1, min(5, n)):
            w = random.uniform(1.0, 10.0)
            edges.append((i, j, w))
            edges.append((j, i, w))
            degrees[i] += 1
            degrees[j] += 1

    # Add remaining edges with preferential attachment
    m = int(n * avg_degree) - len(edges)
    total_degree = sum(degrees) + n  # +n to avoid zero probability

    for _ in range(m):
        # Source uniformly random
        u = random.randint(0, n - 1)

        # Target with probability proportional to degree
        r = random.uniform(0, total_degree)
        cumsum = 0
        v = 0
        for i in range(n):
            cumsum += degrees[i] + 1
            if cumsum >= r:
                v = i
                break

        if u != v:
            w = random.uniform(1.0, 10.0)
            edges.append((u, v, w))
            degrees[u] += 1
            degrees[v] += 1
            total_degree += 2

    return edges


def generate_small_world_graph(n: int, k: int = 4, p: float = 0.1) -> List[Tuple[int, int, float]]:
    """
    Generate a Watts-Strogatz small-world graph.

    Small-world graphs have high clustering (like regular lattices) but
    short path lengths (like random graphs). They model social networks.

    Start with a ring lattice where each node connects to k nearest
    neighbors, then rewire each edge with probability p.
    """
    edges = []
    edge_set = set()

    # Create ring lattice
    for i in range(n):
        for j in range(1, k // 2 + 1):
            target = (i + j) % n
            w = random.uniform(1.0, 10.0)

            # Rewire with probability p
            if random.random() < p:
                # Pick new random target
                new_target = random.randint(0, n - 1)
                if new_target != i and (i, new_target) not in edge_set:
                    target = new_target

            if (i, target) not in edge_set:
                edges.append((i, target, w))
                edges.append((target, i, w))
                edge_set.add((i, target))
                edge_set.add((target, i))

    return edges


def standard_dijkstra(n: int, adj: List[List[Tuple[int, float]]],
                      source: int, target: int) -> float:
    """Standard Dijkstra for comparison."""
    dist = [float('inf')] * n
    dist[source] = 0.0
    pq = [(0.0, source)]

    while pq:
        d, u = heapq.heappop(pq)
        if u == target:
            break
        if d > dist[u]:
            continue
        for v, w in adj[u]:
            if d + w < dist[v]:
                dist[v] = d + w
                heapq.heappush(pq, (dist[v], v))

    return dist[target]


def benchmark_topology(name: str, n: int, edges: List[Tuple[int, int, float]],
                       num_queries: int = 50, warmup: int = 10) -> Dict:
    """Benchmark a specific topology."""
    print(f"\n  {name}:")
    print(f"    Vertices: {n:,}, Edges: {len(edges):,}")

    # Build adjacency list
    adj = [[] for _ in range(n)]
    for u, v, w in edges:
        adj[u].append((v, w))

    # Create ALSSSP solver
    solver = ALSSSP(n=n, edges=edges)

    # Generate queries
    random.seed(123)
    queries = [(random.randint(0, n-1), random.randint(0, n-1))
               for _ in range(num_queries + warmup)]

    # Warmup
    for s, t in queries[:warmup]:
        solver.shortest_path(s, t)

    # Benchmark Dijkstra
    start = time.perf_counter()
    for s, t in queries[warmup:]:
        standard_dijkstra(n, adj, s, t)
    dijkstra_time = (time.perf_counter() - start) * 1000 / num_queries

    # Benchmark ALSSSP
    start = time.perf_counter()
    for s, t in queries[warmup:]:
        solver.shortest_path(s, t)
    alsssp_time = (time.perf_counter() - start) * 1000 / num_queries

    speedup = dijkstra_time / alsssp_time if alsssp_time > 0 else 0

    print(f"    Dijkstra: {dijkstra_time:.3f} ms, ALSSSP: {alsssp_time:.3f} ms")
    print(f"    Speedup: {speedup:.1f}x")

    return {
        'topology': name,
        'n': n,
        'm': len(edges),
        'dijkstra_ms': dijkstra_time,
        'alsssp_ms': alsssp_time,
        'speedup': speedup
    }


def main():
    print("=" * 70)
    print("ALSSSP Topology Comparison Experiment")
    print("=" * 70)

    random.seed(42)
    n = 5000  # Base size for comparison

    results = []

    # Grid graph (sqrt(n) x sqrt(n))
    grid_size = int(math.sqrt(n))
    n_grid, edges_grid = generate_grid_graph(grid_size, grid_size)
    results.append(benchmark_topology("Grid (Road-like)", n_grid, edges_grid))

    # Random graph
    edges_random = generate_random_graph(n, avg_degree=4.0)
    results.append(benchmark_topology("Random (Erdos-Renyi)", n, edges_random))

    # Power-law graph
    edges_power = generate_power_law_graph(n, avg_degree=4.0)
    results.append(benchmark_topology("Power-law (Web-like)", n, edges_power))

    # Small-world graph
    edges_small = generate_small_world_graph(n, k=4, p=0.1)
    results.append(benchmark_topology("Small-world (Social)", n, edges_small))

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\n{'Topology':<25} | {'Dijkstra':>10} | {'ALSSSP':>10} | {'Speedup':>8}")
    print("-" * 60)
    for r in results:
        print(f"{r['topology']:<25} | {r['dijkstra_ms']:>8.3f}ms | {r['alsssp_ms']:>8.3f}ms | {r['speedup']:>7.1f}x")

    # Observations
    print("\n" + "=" * 70)
    print("Observations:")
    print("-" * 70)

    best = max(results, key=lambda x: x['speedup'])
    worst = min(results, key=lambda x: x['speedup'])

    print(f"  Best speedup:  {best['topology']} ({best['speedup']:.1f}x)")
    print(f"  Worst speedup: {worst['topology']} ({worst['speedup']:.1f}x)")
    print("\n  Notes:")
    print("  - Grid graphs benefit most from bidirectional search")
    print("  - Power-law graphs have shorter paths (less room for improvement)")
    print("  - Small-world graphs show moderate improvement")
    print("=" * 70)


if __name__ == "__main__":
    main()
