# ALSSSP - Adaptive Learning Single-Source Shortest Path

> Also known as **Turbo-Dijkstra** — the brand name for this 50x faster shortest path algorithm.

A high-performance shortest path library achieving **10-50x speedup** over standard Dijkstra for point-to-point queries through bidirectional search and adaptive learning.

[![Watch the Video](https://img.shields.io/badge/YouTube-Watch%20Explanation-red?style=for-the-badge&logo=youtube)](https://youtu.be/HHgXiHJqJDU)
[![GitHub](https://img.shields.io/badge/GitHub-Repository-black?style=for-the-badge&logo=github)](https://github.com/RaiAnk/turbo-dijkstra)

## Video Explanation

**New to this project?** Watch the full explanation of how ALSSSP works and why it's 50x faster:

[![ALSSSP Explained](https://img.youtube.com/vi/HHgXiHJqJDU/maxresdefault.jpg)](https://youtu.be/HHgXiHJqJDU)

[Watch on YouTube: I Made Dijkstra's Algorithm 50x Faster — Here's How](https://youtu.be/HHgXiHJqJDU)

## Overview

ALSSSP (Adaptive Learning Single-Source Shortest Path) is a Python library that implements state-of-the-art shortest path algorithms with online learning capabilities. The library automatically adapts to query patterns, caches frequently accessed results, and selects the optimal algorithm for each query type.

### Key Features

- **Bidirectional Search**: Point-to-point queries explore O(sqrt(n)) vertices instead of O(n)
- **Adaptive Algorithm Selection**: Learns which algorithm performs best for your graph and query patterns
- **Multi-Level Caching**: LRU cache for exact queries, precomputed SSSP trees for hot sources
- **Cache-Optimized Storage**: CSR format with BFS vertex reordering for better locality
- **Landmark Heuristics**: A* search with learned lower bounds for faster convergence

### Algorithm Portfolio

| Algorithm | Best For | Time Complexity |
|-----------|----------|-----------------|
| Dijkstra (CSR) | General SSSP | O(m + n log n) |
| Delta-Stepping | Sparse graphs | O(m + n log n) |
| Dial's Buckets | Small integer weights | O(m + nC) |
| Bidirectional BFS | Unit weight graphs | O(m) |
| Bidirectional A* | Point-to-point queries | O(sqrt(n) log n) |

## Installation

### From Source

```bash
git clone https://github.com/RaiAnk/turbo-dijkstra.git
cd turbo-dijkstra
pip install -e .
```

### Using pip

```bash
pip install alsssp
```

### Requirements

- Python >= 3.8
- NumPy >= 1.20.0

For running experiments:
```bash
pip install alsssp[experiments]
```

## Quick Start

```python
from alsssp import ALSSSP

# Define your graph as (source, target, weight) tuples
edges = [
    (0, 1, 1.0),
    (1, 2, 2.0),
    (2, 3, 1.5),
    (0, 3, 5.0),
]

# Create solver (6 vertices)
solver = ALSSSP(n=4, edges=edges)

# Point-to-point query
result = solver.shortest_path(source=0, target=3)
print(f"Distance: {result.distances[3]}")  # Output: 4.5

# Full SSSP from source
result = solver.sssp(source=0)
print(f"All distances: {result.distances}")
```

## Usage Examples

### Basic Point-to-Point Query

```python
from alsssp import ALSSSP

# Create a random graph
import random
n = 10000
edges = [(random.randint(0, n-1), random.randint(0, n-1), random.uniform(1, 10))
         for _ in range(n * 4)]

solver = ALSSSP(n=n, edges=edges)

# Query shortest path
result = solver.shortest_path(0, 5000)
print(f"Distance: {result.distances[5000]}")
print(f"Algorithm used: {result.algorithm_used}")
print(f"Time: {result.time_taken*1000:.2f} ms")
```

### Batch Queries with Caching

```python
# Multiple queries benefit from caching
queries = [(0, 100), (0, 200), (0, 100), (50, 150), (0, 100)]
results = solver.batch_query(queries)

for (s, t), r in zip(queries, results):
    status = "CACHED" if r.cache_hit else "computed"
    print(f"({s} -> {t}): {r.distances[t]:.2f} [{status}]")

# Check cache statistics
stats = solver.get_stats()
print(f"Cache hit rate: {stats['hit_rate']*100:.1f}%")
```

### Path Reconstruction

```python
result = solver.shortest_path(0, 100)

# Get the actual path
path = []
current = 100
while current != -1:
    path.append(current)
    current = result.predecessors[current]
path.reverse()

print(f"Path: {' -> '.join(map(str, path))}")
```

## Performance

ALSSSP achieves significant speedups over standard Dijkstra, especially for point-to-point queries:

| Graph Size (n) | Dijkstra (ms) | ALSSSP (ms) | Speedup |
|----------------|---------------|-------------|---------|
| 1,000 | 0.8 | 0.12 | 6.7x |
| 5,000 | 4.2 | 0.31 | 13.5x |
| 10,000 | 9.1 | 0.42 | 21.7x |
| 50,000 | 52.3 | 1.18 | 44.3x |
| 100,000 | 118.7 | 2.41 | 49.3x |

*Benchmarks on random graphs with average degree 4, point-to-point queries*

The speedup comes from:
1. **Bidirectional search**: Explores O(sqrt(n)) vertices instead of O(n)
2. **Caching**: Repeated queries are instant (O(1) lookup)
3. **Hot sources**: Frequently queried sources get precomputed SSSP trees

## API Reference

### ALSSSP Class

```python
class ALSSSP:
    def __init__(self, n: int, edges: List[Tuple[int, int, float]],
                 cache_size: int = 100000, hot_threshold: int = 10):
        """
        Initialize ALSSSP solver.

        Args:
            n: Number of vertices
            edges: List of (source, target, weight) tuples
            cache_size: Maximum cache entries (default: 100000)
            hot_threshold: Queries before source gets precomputed tree
        """

    def shortest_path(self, source: int, target: int) -> ALSSSPResult:
        """Find shortest path between two vertices."""

    def sssp(self, source: int) -> ALSSSPResult:
        """Compute single-source shortest paths to all vertices."""

    def batch_query(self, queries: List[Tuple[int, int]]) -> List[ALSSSPResult]:
        """Process multiple queries efficiently."""

    def get_stats(self) -> Dict:
        """Get cache and performance statistics."""
```

### ALSSSPResult Class

```python
@dataclass
class ALSSSPResult:
    distances: np.ndarray      # Distance to each vertex
    predecessors: np.ndarray   # Predecessor for path reconstruction
    algorithm_used: str        # Which algorithm was selected
    time_taken: float          # Query execution time
    cache_hit: bool           # Whether result came from cache
```

## Project Structure

```
alsssp/
    __init__.py          # Package exports
    core.py              # Main ALSSSP orchestrator
    algorithms.py        # Algorithm implementations
    memory.py            # Shared memory and caching
    learning.py          # Online learning components

examples/
    basic_usage.py       # Simple usage demonstration
    benchmark_comparison.py  # Performance benchmarks

paper/
    ALSSSP_paper.tex     # Research paper
    figures/             # Experiment visualizations

tests/
    test_correctness.py  # Correctness tests
    test_performance.py  # Performance tests
```

## How It Works

### 1. Graph Preprocessing

During initialization, ALSSSP:
- Converts edges to CSR (Compressed Sparse Row) format for cache efficiency
- Reorders vertices using BFS for better memory locality
- Analyzes weight distribution to guide algorithm selection
- Precomputes landmark distances for A* heuristics

### 2. Query Processing

Each query goes through four phases:

1. **Cache Check**: Look up result in LRU cache or hot source trees
2. **Algorithm Selection**: Choose best algorithm based on learned performance
3. **Execution**: Run selected algorithm with early termination for point-to-point
4. **Post-Processing**: Cache result, update learning components

### 3. Adaptive Learning

Over time, ALSSSP:
- Identifies frequently queried sources ("hot" sources) and precomputes their SSSP trees
- Learns which algorithm performs best for different query types
- Refines A* heuristics based on observed shortest path distances

## Research Paper

For technical details and experimental evaluation of ALSSSP, see the included research paper:

- `paper/ALSSSP_paper.tex` - Full paper with proofs and experiments
- `paper/figures/` - Experimental result visualizations

Key findings:
- 10-50x speedup for point-to-point queries on large graphs
- Speedup scales with sqrt(n) as predicted by theory
- Caching provides additional 2-5x improvement for repeated queries

The paper provides rigorous correctness proofs and complexity analysis for the ALSSSP framework.

## Running Experiments

To reproduce the benchmark results:

```bash
# Install experiment dependencies
pip install alsssp[experiments]

# Run basic benchmark
python examples/benchmark_comparison.py

# Run full experiment suite
python experiments/run_all_experiments.py
```

## Contributing

Contributions to ALSSSP are welcome! Please feel free to submit issues and pull requests.

Areas for improvement:
- GPU acceleration for large-scale graphs
- Distributed processing for massive graphs
- Additional algorithm portfolio members
- Better landmark selection strategies
- Integration with popular graph libraries (NetworkX, igraph)

## License

MIT License - see LICENSE file for details.

## Citation

If you use ALSSSP (Turbo-Dijkstra) in your research, please cite:

```bibtex
@article{rai2026alsssp,
  title={ALSSSP: An Adaptive Learning Framework for Single-Source Shortest Path Computation with Provable Guarantees},
  author={Rai, Ankush},
  journal={ACM SIGMOD International Conference on Management of Data},
  year={2026},
  url={https://github.com/RaiAnk/turbo-dijkstra}
}
```

## Acknowledgments

ALSSSP builds on decades of research in shortest path algorithms, including:
- Dijkstra's original algorithm (1959)
- Bidirectional search techniques
- Delta-stepping for parallel SSSP
- Landmark-based heuristics (ALT algorithm)

## Author

**Dr. Ankush Rai**
Bhilai Institute of Technology, Durg, India

---

*ALSSSP (Turbo-Dijkstra) — Making shortest paths 50x faster.*
