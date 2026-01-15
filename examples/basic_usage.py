"""
Basic Usage Example for ALSSSP
==============================

This script demonstrates the core functionality of ALSSSP with simple examples.
Run this first to verify your installation is working correctly.

Usage:
    python basic_usage.py
"""

import sys
sys.path.insert(0, '..')

from alsssp import ALSSSP


def main():
    print("=" * 60)
    print("ALSSSP Basic Usage Example")
    print("=" * 60)

    # Create a simple graph
    # This is a directed weighted graph with 6 vertices
    #
    #     1 --- 2
    #    /|     |\
    #   1 2     1 3
    #  /  |     |  \
    # 0   3 --- 4 --- 5
    #      \   /
    #       \ /
    #        5 (weight)

    edges = [
        (0, 1, 1.0),   # 0 -> 1 with weight 1
        (1, 2, 1.0),   # 1 -> 2 with weight 1
        (1, 3, 2.0),   # 1 -> 3 with weight 2
        (2, 4, 1.0),   # 2 -> 4 with weight 1
        (2, 5, 3.0),   # 2 -> 5 with weight 3
        (3, 4, 5.0),   # 3 -> 4 with weight 5
        (4, 5, 1.0),   # 4 -> 5 with weight 1
    ]

    n_vertices = 6

    print(f"\nGraph: {n_vertices} vertices, {len(edges)} edges")
    print("Edges:", edges)

    # Create ALSSSP solver
    print("\nInitializing ALSSSP solver...")
    solver = ALSSSP(n=n_vertices, edges=edges)

    # Example 1: Point-to-point query
    print("\n" + "-" * 40)
    print("Example 1: Point-to-Point Query")
    print("-" * 40)

    source, target = 0, 5
    result = solver.shortest_path(source, target)

    print(f"Shortest path from {source} to {target}:")
    print(f"  Distance: {result.distances[target]}")
    print(f"  Algorithm used: {result.algorithm_used}")
    print(f"  Time taken: {result.time_taken*1000:.3f} ms")

    # Example 2: Full SSSP query
    print("\n" + "-" * 40)
    print("Example 2: Full SSSP Query")
    print("-" * 40)

    source = 0
    result = solver.sssp(source)

    print(f"All distances from vertex {source}:")
    for v in range(n_vertices):
        dist = result.distances[v]
        print(f"  -> {v}: {dist if dist < float('inf') else 'unreachable'}")

    # Example 3: Batch queries (demonstrates caching)
    print("\n" + "-" * 40)
    print("Example 3: Batch Queries (Caching Demo)")
    print("-" * 40)

    queries = [
        (0, 5),
        (0, 3),
        (0, 5),  # Repeated - should hit cache
        (1, 4),
        (0, 5),  # Repeated - should hit cache
    ]

    print("Running batch of queries...")
    results = solver.batch_query(queries)

    for (s, t), r in zip(queries, results):
        cache_status = "CACHE HIT" if r.cache_hit else "computed"
        print(f"  ({s} -> {t}): distance={r.distances[t]:.1f}, {cache_status}")

    # Show statistics
    print("\n" + "-" * 40)
    print("Performance Statistics")
    print("-" * 40)

    stats = solver.get_stats()
    print(f"  Total queries: {stats['query_count']}")
    print(f"  Cache hits: {stats['cache_hits']}")
    print(f"  Hot source hits: {stats['hot_hits']}")
    print(f"  Hit rate: {stats['hit_rate']*100:.1f}%")

    print("\n" + "=" * 60)
    print("Example completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
