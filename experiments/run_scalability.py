"""
Scalability Experiment
======================

This script measures how ALSSSP and standard Dijkstra scale with graph size.
We test on random graphs with varying number of vertices while keeping the
average degree constant.

The key observation is that ALSSSP's speedup increases with graph size,
approaching sqrt(n) for large graphs due to bidirectional search.

Usage:
    python run_scalability.py

Output:
    - Console output with timing results
    - CSV file with detailed measurements
    - Plot showing scaling behavior (if matplotlib available)
"""

import sys
sys.path.insert(0, '..')

import time
import random
import heapq
import csv
from typing import List, Tuple

# Try to import plotting libraries (optional)
try:
    import matplotlib.pyplot as plt
    import numpy as np
    HAS_PLOTTING = True
except ImportError:
    HAS_PLOTTING = False
    print("Note: matplotlib not available, skipping plot generation")

from alsssp import ALSSSP


def generate_random_graph(n: int, avg_degree: float = 4.0,
                          seed: int = 42) -> List[Tuple[int, int, float]]:
    """
    Generate a random directed graph.

    We use an Erdos-Renyi style model where each potential edge has
    probability avg_degree / n of existing. Edge weights are uniform
    random in [1, 10].
    """
    random.seed(seed)
    edges = []
    m = int(n * avg_degree)

    for _ in range(m):
        u = random.randint(0, n - 1)
        v = random.randint(0, n - 1)
        if u != v:
            w = random.uniform(1.0, 10.0)
            edges.append((u, v, w))

    return edges


def standard_dijkstra(n: int, adj: List[List[Tuple[int, float]]],
                      source: int, target: int) -> float:
    """
    Baseline Dijkstra implementation.

    This is a standard textbook implementation with early termination
    for point-to-point queries. We use it as the baseline to measure
    ALSSSP's improvement.
    """
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


def run_scalability_experiment(sizes: List[int], num_queries: int = 50,
                               warmup: int = 10, avg_degree: float = 4.0):
    """
    Run the scalability experiment across different graph sizes.

    For each size, we:
    1. Generate a random graph
    2. Run warmup queries on ALSSSP (to populate caches)
    3. Time both algorithms on the same set of queries
    4. Record the speedup
    """
    results = []

    print("=" * 70)
    print("ALSSSP Scalability Experiment")
    print("=" * 70)
    print(f"\nParameters:")
    print(f"  Average degree: {avg_degree}")
    print(f"  Queries per size: {num_queries}")
    print(f"  Warmup queries: {warmup}")
    print(f"  Graph sizes: {sizes}")

    for n in sizes:
        print(f"\n{'='*50}")
        print(f"Testing n = {n:,}")
        print('='*50)

        # Generate graph
        print("  Generating graph...")
        edges = generate_random_graph(n, avg_degree=avg_degree, seed=42)
        m = len(edges)
        print(f"  Graph has {m:,} edges")

        # Build adjacency list for baseline
        adj = [[] for _ in range(n)]
        for u, v, w in edges:
            adj[u].append((v, w))

        # Create ALSSSP solver
        print("  Initializing ALSSSP...")
        solver = ALSSSP(n=n, edges=edges)

        # Generate query pairs
        random.seed(123)
        queries = [(random.randint(0, n-1), random.randint(0, n-1))
                   for _ in range(num_queries + warmup)]

        # Warmup ALSSSP
        print(f"  Running {warmup} warmup queries...")
        for s, t in queries[:warmup]:
            solver.shortest_path(s, t)

        # Benchmark baseline Dijkstra
        print(f"  Benchmarking standard Dijkstra...")
        start = time.perf_counter()
        for s, t in queries[warmup:]:
            standard_dijkstra(n, adj, s, t)
        dijkstra_time = (time.perf_counter() - start) * 1000 / num_queries

        # Benchmark ALSSSP
        print(f"  Benchmarking ALSSSP...")
        start = time.perf_counter()
        for s, t in queries[warmup:]:
            solver.shortest_path(s, t)
        alsssp_time = (time.perf_counter() - start) * 1000 / num_queries

        # Calculate speedup
        speedup = dijkstra_time / alsssp_time if alsssp_time > 0 else 0
        theoretical = (n ** 0.5) / 2  # Rough theoretical prediction

        results.append({
            'n': n,
            'm': m,
            'dijkstra_ms': dijkstra_time,
            'alsssp_ms': alsssp_time,
            'speedup': speedup,
            'theoretical': theoretical
        })

        print(f"\n  Results:")
        print(f"    Dijkstra: {dijkstra_time:.3f} ms/query")
        print(f"    ALSSSP:   {alsssp_time:.3f} ms/query")
        print(f"    Speedup:  {speedup:.1f}x")
        print(f"    (Theoretical sqrt(n)/2 = {theoretical:.1f})")

    return results


def save_results(results: List[dict], filename: str = "scalability_results.csv"):
    """Save results to CSV file."""
    with open(filename, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"\nResults saved to {filename}")


def plot_results(results: List[dict], filename: str = "scalability_plot.png"):
    """Generate visualization of scaling behavior."""
    if not HAS_PLOTTING:
        return

    sizes = [r['n'] for r in results]
    speedups = [r['speedup'] for r in results]
    theoretical = [r['theoretical'] for r in results]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left plot: Speedup vs graph size
    ax1 = axes[0]
    ax1.plot(sizes, speedups, 'bo-', linewidth=2, markersize=8, label='Measured')
    ax1.plot(sizes, theoretical, 'r--', linewidth=2, label='Theoretical (sqrt(n)/2)')
    ax1.set_xlabel('Graph Size (n)', fontsize=12)
    ax1.set_ylabel('Speedup (x)', fontsize=12)
    ax1.set_title('ALSSSP Speedup vs Graph Size', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xscale('log')

    # Right plot: Query time comparison
    ax2 = axes[1]
    dijkstra_times = [r['dijkstra_ms'] for r in results]
    alsssp_times = [r['alsssp_ms'] for r in results]

    x = np.arange(len(sizes))
    width = 0.35

    ax2.bar(x - width/2, dijkstra_times, width, label='Dijkstra', color='#ff7f0e')
    ax2.bar(x + width/2, alsssp_times, width, label='ALSSSP', color='#1f77b4')
    ax2.set_xlabel('Graph Size', fontsize=12)
    ax2.set_ylabel('Time per Query (ms)', fontsize=12)
    ax2.set_title('Query Time Comparison', fontsize=14)
    ax2.set_xticks(x)
    ax2.set_xticklabels([f'{s:,}' for s in sizes], rotation=45)
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"Plot saved to {filename}")
    plt.close()


def main():
    # Test on various graph sizes
    # Start small for quick testing, larger for publication-quality results
    sizes = [500, 1000, 2000, 5000, 10000]

    results = run_scalability_experiment(sizes, num_queries=50)

    # Print summary table
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\n{'Vertices':>10} | {'Edges':>10} | {'Dijkstra':>12} | {'ALSSSP':>10} | {'Speedup':>8}")
    print("-" * 60)
    for r in results:
        print(f"{r['n']:>10,} | {r['m']:>10,} | {r['dijkstra_ms']:>10.3f}ms | {r['alsssp_ms']:>8.3f}ms | {r['speedup']:>7.1f}x")

    # Save results
    save_results(results)
    plot_results(results)

    print("\n" + "=" * 70)
    print("Experiment complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
