"""
ALSSSP - Adaptive Learning Single-Source Shortest Path
=======================================================

A high-performance shortest path library that combines multiple algorithmic
optimizations with online learning to achieve 10-50x speedups over standard
Dijkstra's algorithm for point-to-point queries.

Key Features:
    - Bidirectional search reduces explored vertices from O(n) to O(sqrt(n))
    - Multi-level caching with LRU eviction and hot source precomputation
    - Adaptive algorithm selection based on graph and query characteristics
    - Cache-optimized CSR data structures with BFS vertex reordering
    - Online learning that improves performance over time

Basic Usage:
    >>> from alsssp import ALSSSP
    >>>
    >>> # Create graph as list of (source, target, weight) edges
    >>> edges = [(0, 1, 1.0), (1, 2, 2.0), (0, 2, 4.0)]
    >>> solver = ALSSSP(n=3, edges=edges)
    >>>
    >>> # Point-to-point query (fastest)
    >>> result = solver.shortest_path(source=0, target=2)
    >>> print(f"Distance: {result.distances[2]}")
    >>>
    >>> # Full SSSP from a source
    >>> result = solver.sssp(source=0)
    >>> print(f"All distances: {result.distances}")

For more examples, see the examples/ directory.

Author: Research Team
License: MIT
Version: 1.0.0
"""

__version__ = "1.0.0"
__author__ = "Research Team"

# Main classes - import these for typical usage
from .core import ALSSSP, ALSSSPResult, alsssp

# Algorithm implementations - for advanced users
from .algorithms import (
    CacheOptimizedDijkstra,
    DeltaStepping,
    DialBuckets,
    BidirectionalBFS,
    BidirectionalAStar,
)

# Memory and learning components - for customization
from .memory import SharedGraphMemory
from .learning import LRUCache, QueryPatternLearner, AlgorithmSelector, LearnedHeuristic

# Expose key symbols at package level
__all__ = [
    # Main API
    "ALSSSP",
    "ALSSSPResult",
    "alsssp",

    # Algorithms
    "CacheOptimizedDijkstra",
    "DeltaStepping",
    "DialBuckets",
    "BidirectionalBFS",
    "BidirectionalAStar",

    # Components
    "SharedGraphMemory",
    "LRUCache",
    "QueryPatternLearner",
    "AlgorithmSelector",
    "LearnedHeuristic",
]
