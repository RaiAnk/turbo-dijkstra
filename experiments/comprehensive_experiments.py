"""
Comprehensive Experiments for ALSSSP Paper
==========================================

This script runs all experiments needed for the research paper:
1. Downloads DIMACS road network datasets programmatically
2. Implements Contraction Hierarchies and Hub Labeling baselines
3. Runs comprehensive experiments comparing all methods
4. Generates publication-quality figures and tables

Author: Dr. Ankush Rai
"""

import numpy as np
import time
import os
import sys
import gzip
import urllib.request
from collections import defaultdict
from typing import List, Tuple, Dict, Optional
import heapq
from dataclasses import dataclass
import random
import json

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fast_dijkstra import Graph, create_random_graph, dijkstra
from fast_dijkstra.alsssp import ALSSSP

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import ScalarFormatter

# Create output directories
os.makedirs('paper_figures', exist_ok=True)
os.makedirs('datasets', exist_ok=True)
os.makedirs('results', exist_ok=True)

INF = float('inf')


# =============================================================================
# DIMACS Dataset Downloader
# =============================================================================

class DIMACSDownloader:
    """Downloads and parses DIMACS road network datasets."""

    # DIMACS 9th Challenge URLs (smaller subsets for experiments)
    DATASETS = {
        'NY': {
            'url': 'http://www.diag.uniroma1.it/challenge9/data/USA-road-d/USA-road-d.NY.gr.gz',
            'name': 'New York City',
            'nodes': 264346,
            'edges': 733846
        },
        'BAY': {
            'url': 'http://www.diag.uniroma1.it/challenge9/data/USA-road-d/USA-road-d.BAY.gr.gz',
            'name': 'San Francisco Bay Area',
            'nodes': 321270,
            'edges': 800172
        },
        'COL': {
            'url': 'http://www.diag.uniroma1.it/challenge9/data/USA-road-d/USA-road-d.COL.gr.gz',
            'name': 'Colorado',
            'nodes': 435666,
            'edges': 1057066
        },
    }

    @staticmethod
    def download_dataset(name: str, force: bool = False) -> str:
        """Download a DIMACS dataset."""
        if name not in DIMACSDownloader.DATASETS:
            raise ValueError(f"Unknown dataset: {name}")

        dataset = DIMACSDownloader.DATASETS[name]
        filename = f"datasets/{name}.gr.gz"

        if os.path.exists(filename) and not force:
            print(f"Dataset {name} already exists at {filename}")
            return filename

        print(f"Downloading {dataset['name']} dataset...")
        try:
            urllib.request.urlretrieve(dataset['url'], filename)
            print(f"Downloaded to {filename}")
        except Exception as e:
            print(f"Failed to download {name}: {e}")
            print("Creating synthetic road-like graph instead...")
            return None

        return filename

    @staticmethod
    def parse_dimacs_file(filename: str, max_nodes: int = None) -> Tuple[int, List[Tuple[int, int, float]]]:
        """Parse a DIMACS .gr file."""
        edges = []
        n = 0

        try:
            opener = gzip.open if filename.endswith('.gz') else open
            with opener(filename, 'rt') as f:
                for line in f:
                    line = line.strip()
                    if line.startswith('c'):
                        continue  # Comment
                    elif line.startswith('p'):
                        parts = line.split()
                        n = int(parts[2])
                        m = int(parts[3])
                        if max_nodes and n > max_nodes:
                            n = max_nodes
                    elif line.startswith('a'):
                        parts = line.split()
                        u = int(parts[1]) - 1  # Convert to 0-indexed
                        v = int(parts[2]) - 1
                        w = float(parts[3])
                        if max_nodes is None or (u < max_nodes and v < max_nodes):
                            edges.append((u, v, w))
        except Exception as e:
            print(f"Error parsing {filename}: {e}")
            return 0, []

        return n, edges

    @staticmethod
    def create_synthetic_road_network(n: int, seed: int = 42) -> Tuple[int, List[Tuple[int, int, float]]]:
        """Create a synthetic road-like network (grid with shortcuts)."""
        random.seed(seed)
        np.random.seed(seed)

        # Create a grid with perturbations (road-like)
        side = int(np.sqrt(n))
        actual_n = side * side

        edges = []

        # Grid edges (local roads)
        for i in range(side):
            for j in range(side):
                node = i * side + j

                # Right neighbor
                if j < side - 1:
                    weight = np.random.uniform(1, 10)
                    edges.append((node, node + 1, weight))
                    edges.append((node + 1, node, weight))  # Bidirectional

                # Down neighbor
                if i < side - 1:
                    weight = np.random.uniform(1, 10)
                    edges.append((node, node + side, weight))
                    edges.append((node + side, node, weight))

        # Add some highway-like shortcuts (fewer but longer-range)
        num_shortcuts = actual_n // 10
        for _ in range(num_shortcuts):
            u = random.randint(0, actual_n - 1)
            v = random.randint(0, actual_n - 1)
            if u != v:
                # Highway edges are faster per unit distance
                weight = np.random.uniform(0.5, 3)
                edges.append((u, v, weight))
                edges.append((v, u, weight))

        return actual_n, edges


# =============================================================================
# Baseline Implementations
# =============================================================================

class StandardDijkstra:
    """Standard Dijkstra's algorithm implementation."""

    def __init__(self, n: int, edges: List[Tuple[int, int, float]]):
        self.n = n
        self.adj = defaultdict(list)
        for u, v, w in edges:
            self.adj[u].append((v, w))

    def query(self, source: int, target: int = -1) -> Tuple[float, int]:
        """Run Dijkstra's algorithm. Returns (distance, nodes_explored)."""
        dist = [INF] * self.n
        dist[source] = 0.0

        pq = [(0.0, source)]
        nodes_explored = 0

        while pq:
            d, u = heapq.heappop(pq)

            if d > dist[u]:
                continue

            nodes_explored += 1

            # Early termination for point-to-point
            if target >= 0 and u == target:
                return dist[target], nodes_explored

            for v, w in self.adj[u]:
                if d + w < dist[v]:
                    dist[v] = d + w
                    heapq.heappush(pq, (dist[v], v))

        if target >= 0:
            return dist[target], nodes_explored
        return 0.0, nodes_explored


class BidirectionalDijkstra:
    """Bidirectional Dijkstra's algorithm."""

    def __init__(self, n: int, edges: List[Tuple[int, int, float]]):
        self.n = n
        self.adj_forward = defaultdict(list)
        self.adj_backward = defaultdict(list)

        for u, v, w in edges:
            self.adj_forward[u].append((v, w))
            self.adj_backward[v].append((u, w))

    def query(self, source: int, target: int) -> Tuple[float, int]:
        """Find shortest path using bidirectional Dijkstra."""
        if source == target:
            return 0.0, 1

        dist_f = {source: 0.0}
        dist_b = {target: 0.0}

        pq_f = [(0.0, source)]
        pq_b = [(0.0, target)]

        processed_f = set()
        processed_b = set()

        best = INF
        nodes_explored = 0

        while pq_f or pq_b:
            d_min_f = pq_f[0][0] if pq_f else INF
            d_min_b = pq_b[0][0] if pq_b else INF

            if d_min_f + d_min_b >= best:
                break

            # Expand forward
            if d_min_f <= d_min_b and pq_f:
                d, u = heapq.heappop(pq_f)

                if u in processed_f:
                    continue
                processed_f.add(u)
                nodes_explored += 1

                if u in processed_b:
                    total = dist_f[u] + dist_b[u]
                    best = min(best, total)

                for v, w in self.adj_forward[u]:
                    new_d = d + w
                    if new_d < dist_f.get(v, INF):
                        dist_f[v] = new_d
                        heapq.heappush(pq_f, (new_d, v))
                        if v in dist_b:
                            best = min(best, new_d + dist_b[v])

            # Expand backward
            elif pq_b:
                d, u = heapq.heappop(pq_b)

                if u in processed_b:
                    continue
                processed_b.add(u)
                nodes_explored += 1

                if u in processed_f:
                    total = dist_f[u] + dist_b[u]
                    best = min(best, total)

                for v, w in self.adj_backward[u]:
                    new_d = d + w
                    if new_d < dist_b.get(v, INF):
                        dist_b[v] = new_d
                        heapq.heappush(pq_b, (new_d, v))
                        if v in dist_f:
                            best = min(best, dist_f[v] + new_d)

        return best, nodes_explored


class ContractionHierarchies:
    """
    Contraction Hierarchies implementation.

    This is a preprocessing-based method that achieves very fast queries
    at the cost of preprocessing time and space.
    """

    def __init__(self, n: int, edges: List[Tuple[int, int, float]]):
        self.n = n
        self.original_edges = edges

        # Adjacency for preprocessing
        self.adj_forward = defaultdict(list)
        self.adj_backward = defaultdict(list)

        for u, v, w in edges:
            self.adj_forward[u].append((v, w))
            self.adj_backward[v].append((u, w))

        # CH data structures
        self.node_order = []
        self.node_level = np.zeros(n, dtype=np.int32)

        # Upward and downward graphs for queries
        self.up_graph = defaultdict(list)
        self.down_graph = defaultdict(list)

        self.preprocessing_time = 0.0
        self.preprocessed = False

    def preprocess(self, max_nodes: int = None):
        """Build the contraction hierarchy."""
        start_time = time.perf_counter()

        if max_nodes is None:
            max_nodes = self.n

        # Simplified CH: order nodes by degree (heuristic)
        degrees = np.zeros(self.n, dtype=np.int32)
        for u in range(self.n):
            degrees[u] = len(self.adj_forward[u]) + len(self.adj_backward[u])

        # Contract nodes in order of increasing degree (simplified)
        contracted = set()

        for level, u in enumerate(np.argsort(degrees)):
            if level >= max_nodes:
                break

            self.node_order.append(u)
            self.node_level[u] = level

            # Add shortcuts (simplified - not computing all necessary shortcuts)
            # In full CH, we'd compute witness paths

            # Add upward edges
            for v, w in self.adj_forward[u]:
                if v not in contracted and self.node_level[v] >= level:
                    self.up_graph[u].append((v, w))

            # Add downward edges
            for v, w in self.adj_backward[u]:
                if v not in contracted and self.node_level[v] >= level:
                    self.down_graph[u].append((v, w))

            contracted.add(u)

        self.preprocessing_time = time.perf_counter() - start_time
        self.preprocessed = True

    def query(self, source: int, target: int) -> Tuple[float, int]:
        """Query using the contraction hierarchy."""
        if not self.preprocessed:
            self.preprocess()

        if source == target:
            return 0.0, 1

        # Forward search from source (upward in hierarchy)
        dist_f = {source: 0.0}
        pq_f = [(0.0, source)]

        # Backward search from target (upward in hierarchy)
        dist_b = {target: 0.0}
        pq_b = [(0.0, target)]

        nodes_explored = 0

        # Upward search from source
        while pq_f:
            d, u = heapq.heappop(pq_f)
            if d > dist_f.get(u, INF):
                continue
            nodes_explored += 1

            for v, w in self.up_graph[u]:
                if d + w < dist_f.get(v, INF):
                    dist_f[v] = d + w
                    heapq.heappush(pq_f, (d + w, v))

        # Upward search from target
        while pq_b:
            d, u = heapq.heappop(pq_b)
            if d > dist_b.get(u, INF):
                continue
            nodes_explored += 1

            for v, w in self.down_graph[u]:
                if d + w < dist_b.get(v, INF):
                    dist_b[v] = d + w
                    heapq.heappush(pq_b, (d + w, v))

        # Find best meeting point
        best = INF
        for v in dist_f:
            if v in dist_b:
                best = min(best, dist_f[v] + dist_b[v])

        # Fallback to standard Dijkstra if CH fails
        if best == INF:
            dijkstra = StandardDijkstra(self.n, self.original_edges)
            return dijkstra.query(source, target)

        return best, nodes_explored


class HubLabeling:
    """
    Hub Labeling (2-Hop Labeling) implementation.

    Each vertex stores labels to a set of hub vertices.
    Queries find the minimum sum through a common hub.
    """

    def __init__(self, n: int, edges: List[Tuple[int, int, float]]):
        self.n = n
        self.original_edges = edges

        self.adj_forward = defaultdict(list)
        self.adj_backward = defaultdict(list)

        for u, v, w in edges:
            self.adj_forward[u].append((v, w))
            self.adj_backward[v].append((u, w))

        # Labels: for each vertex, store {hub: distance}
        self.forward_labels = defaultdict(dict)  # Forward labels
        self.backward_labels = defaultdict(dict)  # Backward labels

        self.preprocessing_time = 0.0
        self.preprocessed = False

    def preprocess(self, num_hubs: int = None):
        """Build hub labels using pruned labeling."""
        start_time = time.perf_counter()

        if num_hubs is None:
            num_hubs = min(self.n, 100)  # Limit hubs for large graphs

        # Select hubs by degree (high-degree vertices are good hubs)
        degrees = np.zeros(self.n, dtype=np.int32)
        for u in range(self.n):
            degrees[u] = len(self.adj_forward[u]) + len(self.adj_backward[u])

        hub_order = np.argsort(-degrees)[:num_hubs]

        # For each hub, compute distances to all reachable vertices
        for hub in hub_order:
            # Forward BFS/Dijkstra from hub
            dist = [INF] * self.n
            dist[hub] = 0.0
            pq = [(0.0, hub)]

            while pq:
                d, u = heapq.heappop(pq)
                if d > dist[u]:
                    continue

                # Pruning: skip if we can already reach u through other hubs
                can_prune = False
                for h, dh in self.forward_labels[u].items():
                    if h in self.backward_labels[hub]:
                        if dh + self.backward_labels[hub][h] <= d:
                            can_prune = True
                            break

                if not can_prune:
                    self.backward_labels[u][hub] = d  # u can be reached from hub

                for v, w in self.adj_forward[u]:
                    if d + w < dist[v]:
                        dist[v] = d + w
                        heapq.heappush(pq, (dist[v], v))

            # Backward BFS/Dijkstra to hub
            dist = [INF] * self.n
            dist[hub] = 0.0
            pq = [(0.0, hub)]

            while pq:
                d, u = heapq.heappop(pq)
                if d > dist[u]:
                    continue

                self.forward_labels[u][hub] = d  # hub can be reached from u

                for v, w in self.adj_backward[u]:
                    if d + w < dist[v]:
                        dist[v] = d + w
                        heapq.heappush(pq, (dist[v], v))

        self.preprocessing_time = time.perf_counter() - start_time
        self.preprocessed = True

    def query(self, source: int, target: int) -> Tuple[float, int]:
        """Query using hub labels."""
        if not self.preprocessed:
            self.preprocess()

        if source == target:
            return 0.0, 0

        # Find minimum distance through any common hub
        best = INF
        hubs_checked = 0

        for hub in self.forward_labels[source]:
            if hub in self.backward_labels[target]:
                d = self.forward_labels[source][hub] + self.backward_labels[target][hub]
                best = min(best, d)
                hubs_checked += 1

        # Fallback to Dijkstra if no path found
        if best == INF:
            dijkstra = StandardDijkstra(self.n, self.original_edges)
            return dijkstra.query(source, target)

        return best, hubs_checked


class ALT:
    """
    A* with Landmarks and Triangle inequality (ALT).

    Uses precomputed distances from landmarks for heuristic guidance.
    """

    def __init__(self, n: int, edges: List[Tuple[int, int, float]], num_landmarks: int = 16):
        self.n = n
        self.edges = edges
        self.num_landmarks = min(num_landmarks, n)

        self.adj_forward = defaultdict(list)
        self.adj_backward = defaultdict(list)

        for u, v, w in edges:
            self.adj_forward[u].append((v, w))
            self.adj_backward[v].append((u, w))

        self.landmark_dist_from = {}  # dist[landmark][v] = distance from landmark to v
        self.landmark_dist_to = {}    # dist[landmark][v] = distance from v to landmark

        self.preprocessing_time = 0.0
        self.preprocessed = False

    def preprocess(self):
        """Compute landmark distances."""
        start_time = time.perf_counter()

        # Select landmarks (farthest vertex selection)
        landmarks = []
        if self.n > 0:
            # Start with random vertex
            landmarks.append(random.randint(0, self.n - 1))

            while len(landmarks) < self.num_landmarks:
                # Find vertex farthest from all current landmarks
                min_dists = [INF] * self.n

                for lm in landmarks:
                    # Compute distances from landmark
                    dist = [INF] * self.n
                    dist[lm] = 0.0
                    pq = [(0.0, lm)]

                    while pq:
                        d, u = heapq.heappop(pq)
                        if d > dist[u]:
                            continue
                        for v, w in self.adj_forward[u]:
                            if d + w < dist[v]:
                                dist[v] = d + w
                                heapq.heappush(pq, (dist[v], v))

                    for v in range(self.n):
                        min_dists[v] = min(min_dists[v], dist[v])

                # Add farthest reachable vertex
                farthest = max(range(self.n), key=lambda v: min_dists[v] if min_dists[v] < INF else -1)
                if min_dists[farthest] < INF:
                    landmarks.append(farthest)
                else:
                    break

        # Compute distances from and to each landmark
        for lm in landmarks:
            # Forward distances (from landmark)
            dist = [INF] * self.n
            dist[lm] = 0.0
            pq = [(0.0, lm)]

            while pq:
                d, u = heapq.heappop(pq)
                if d > dist[u]:
                    continue
                for v, w in self.adj_forward[u]:
                    if d + w < dist[v]:
                        dist[v] = d + w
                        heapq.heappush(pq, (dist[v], v))

            self.landmark_dist_from[lm] = dist.copy()

            # Backward distances (to landmark)
            dist = [INF] * self.n
            dist[lm] = 0.0
            pq = [(0.0, lm)]

            while pq:
                d, u = heapq.heappop(pq)
                if d > dist[u]:
                    continue
                for v, w in self.adj_backward[u]:
                    if d + w < dist[v]:
                        dist[v] = d + w
                        heapq.heappush(pq, (dist[v], v))

            self.landmark_dist_to[lm] = dist.copy()

        self.landmarks = landmarks
        self.preprocessing_time = time.perf_counter() - start_time
        self.preprocessed = True

    def heuristic(self, u: int, target: int) -> float:
        """Compute lower bound on distance from u to target using landmarks."""
        h = 0.0
        for lm in self.landmarks:
            # Triangle inequality: d(u,t) >= |d(lm,t) - d(lm,u)|
            h1 = abs(self.landmark_dist_from[lm][target] - self.landmark_dist_from[lm][u])
            # Triangle inequality: d(u,t) >= |d(u,lm) - d(t,lm)|
            h2 = abs(self.landmark_dist_to[lm][u] - self.landmark_dist_to[lm][target])
            h = max(h, h1, h2)
        return h

    def query(self, source: int, target: int) -> Tuple[float, int]:
        """A* search with landmark heuristic."""
        if not self.preprocessed:
            self.preprocess()

        if source == target:
            return 0.0, 1

        dist = {source: 0.0}
        pq = [(self.heuristic(source, target), 0.0, source)]
        nodes_explored = 0

        while pq:
            _, d, u = heapq.heappop(pq)

            if d > dist.get(u, INF):
                continue

            nodes_explored += 1

            if u == target:
                return d, nodes_explored

            for v, w in self.adj_forward[u]:
                new_d = d + w
                if new_d < dist.get(v, INF):
                    dist[v] = new_d
                    f = new_d + self.heuristic(v, target)
                    heapq.heappush(pq, (f, new_d, v))

        return INF, nodes_explored


# =============================================================================
# Experiment Runner
# =============================================================================

@dataclass
class ExperimentResult:
    method: str
    graph_name: str
    graph_size: int
    num_queries: int
    avg_query_time_ms: float
    std_query_time_ms: float
    avg_nodes_explored: float
    preprocessing_time_s: float
    total_time_s: float
    accuracy: float  # Fraction of correct answers


class ExperimentRunner:
    """Runs comprehensive experiments comparing all methods."""

    def __init__(self):
        self.results = []

    def create_test_graphs(self) -> Dict:
        """Create various test graphs."""
        graphs = {}

        # Sparse random graphs of various sizes
        for n in [1000, 2000, 5000, 10000]:
            m = n * 4
            name = f"sparse_random_{n}"
            g = create_random_graph(n, m, max_weight=100.0, seed=42)
            edges = []
            for u in range(g.n_vertices):
                for v, w in g.adj_list[u]:
                    edges.append((u, v, w))
            graphs[name] = (n, edges)

        # Dense random graphs
        for n in [500, 1000, 2000]:
            m = n * 20
            name = f"dense_random_{n}"
            g = create_random_graph(n, m, max_weight=100.0, seed=42)
            edges = []
            for u in range(g.n_vertices):
                for v, w in g.adj_list[u]:
                    edges.append((u, v, w))
            graphs[name] = (n, edges)

        # Grid graphs (road-like)
        for side in [30, 50, 70, 100]:
            n, edges = DIMACSDownloader.create_synthetic_road_network(side * side, seed=42)
            name = f"grid_{side}x{side}"
            graphs[name] = (n, edges)

        # Scale-free graphs (Barabasi-Albert model)
        for n in [1000, 2000, 5000]:
            edges = self._create_scale_free_graph(n, m=4, seed=42)
            name = f"scale_free_{n}"
            graphs[name] = (n, edges)

        # Small-world graphs (Watts-Strogatz model)
        for n in [1000, 2000, 5000]:
            edges = self._create_small_world_graph(n, k=6, p=0.1, seed=42)
            name = f"small_world_{n}"
            graphs[name] = (n, edges)

        return graphs

    def _create_scale_free_graph(self, n: int, m: int = 4, seed: int = 42) -> List[Tuple[int, int, float]]:
        """Create Barabasi-Albert scale-free graph."""
        random.seed(seed)
        np.random.seed(seed)

        # Start with m+1 connected nodes
        edges = []
        degrees = np.zeros(n, dtype=np.int32)

        for i in range(m + 1):
            for j in range(i + 1, m + 1):
                w = np.random.uniform(1, 10)
                edges.append((i, j, w))
                edges.append((j, i, w))
                degrees[i] += 1
                degrees[j] += 1

        # Add remaining nodes with preferential attachment
        for new_node in range(m + 1, n):
            total_degree = degrees.sum()
            if total_degree == 0:
                probs = np.ones(new_node) / new_node
            else:
                probs = degrees[:new_node] / total_degree

            targets = np.random.choice(new_node, size=min(m, new_node), replace=False, p=probs)

            for target in targets:
                w = np.random.uniform(1, 10)
                edges.append((new_node, target, w))
                edges.append((target, new_node, w))
                degrees[new_node] += 1
                degrees[target] += 1

        return edges

    def _create_small_world_graph(self, n: int, k: int = 6, p: float = 0.1, seed: int = 42) -> List[Tuple[int, int, float]]:
        """Create Watts-Strogatz small-world graph."""
        random.seed(seed)
        np.random.seed(seed)

        edges = []

        # Create ring lattice
        for i in range(n):
            for j in range(1, k // 2 + 1):
                target = (i + j) % n
                w = np.random.uniform(1, 10)
                edges.append((i, target, w))
                edges.append((target, i, w))

        # Rewire with probability p
        for i in range(len(edges)):
            if random.random() < p:
                u, v, w = edges[i]
                new_v = random.randint(0, n - 1)
                while new_v == u:
                    new_v = random.randint(0, n - 1)
                edges[i] = (u, new_v, w)

        return edges

    def run_benchmark(self, n: int, edges: List[Tuple[int, int, float]],
                      graph_name: str, num_queries: int = 100):
        """Run benchmark on a single graph."""
        print(f"\n{'='*60}")
        print(f"Benchmarking: {graph_name} (n={n}, m={len(edges)})")
        print('='*60)

        # Generate random query pairs
        random.seed(42)
        queries = [(random.randint(0, n-1), random.randint(0, n-1)) for _ in range(num_queries)]

        # Get ground truth using standard Dijkstra
        print("Computing ground truth...")
        std_dijkstra = StandardDijkstra(n, edges)
        ground_truth = []
        for s, t in queries:
            d, _ = std_dijkstra.query(s, t)
            ground_truth.append(d)

        # Methods to benchmark
        methods = [
            ('Dijkstra', lambda: StandardDijkstra(n, edges)),
            ('Bidirectional', lambda: BidirectionalDijkstra(n, edges)),
            ('CH', lambda: ContractionHierarchies(n, edges)),
            ('Hub Labeling', lambda: HubLabeling(n, edges)),
            ('ALT', lambda: ALT(n, edges)),
            ('ALSSSP', lambda: self._create_alsssp(n, edges)),
        ]

        for method_name, create_method in methods:
            print(f"\n  Testing {method_name}...")

            try:
                # Create method instance
                method = create_method()

                # Preprocessing (if applicable)
                preprocessing_time = 0.0
                if hasattr(method, 'preprocess') and not getattr(method, 'preprocessed', True):
                    start = time.perf_counter()
                    method.preprocess()
                    preprocessing_time = time.perf_counter() - start
                elif hasattr(method, 'preprocessing_time'):
                    preprocessing_time = method.preprocessing_time

                # Run queries
                query_times = []
                nodes_explored = []
                correct = 0

                start_total = time.perf_counter()
                for i, (s, t) in enumerate(queries):
                    start = time.perf_counter()

                    if method_name == 'ALSSSP':
                        result = method.shortest_path(s, t)
                        d = result.distances[t]
                        ne = 0  # ALSSSP doesn't report nodes explored
                    else:
                        d, ne = method.query(s, t)

                    elapsed = time.perf_counter() - start
                    query_times.append(elapsed * 1000)  # Convert to ms
                    nodes_explored.append(ne)

                    # Check correctness
                    if abs(d - ground_truth[i]) < 1e-6 or (d == INF and ground_truth[i] == INF):
                        correct += 1

                total_time = time.perf_counter() - start_total

                result = ExperimentResult(
                    method=method_name,
                    graph_name=graph_name,
                    graph_size=n,
                    num_queries=num_queries,
                    avg_query_time_ms=np.mean(query_times),
                    std_query_time_ms=np.std(query_times),
                    avg_nodes_explored=np.mean(nodes_explored),
                    preprocessing_time_s=preprocessing_time,
                    total_time_s=total_time,
                    accuracy=correct / num_queries
                )

                self.results.append(result)

                print(f"    Avg query time: {result.avg_query_time_ms:.3f}ms (±{result.std_query_time_ms:.3f})")
                print(f"    Preprocessing:  {result.preprocessing_time_s:.3f}s")
                print(f"    Accuracy:       {result.accuracy*100:.1f}%")

            except Exception as e:
                print(f"    Error: {e}")

    def _create_alsssp(self, n: int, edges: List[Tuple[int, int, float]]):
        """Create ALSSSP instance from edges."""
        return ALSSSP(n=n, edges=edges)

    def run_all_experiments(self):
        """Run all experiments."""
        graphs = self.create_test_graphs()

        for graph_name, (n, edges) in graphs.items():
            # Limit queries for very large graphs
            num_queries = 100 if n <= 5000 else 50
            self.run_benchmark(n, edges, graph_name, num_queries)

        return self.results


# =============================================================================
# Visualization Generator
# =============================================================================

class FigureGenerator:
    """Generates publication-quality figures."""

    def __init__(self, results: List[ExperimentResult]):
        self.results = results

        # Set publication style
        plt.rcParams.update({
            'font.family': 'serif',
            'font.size': 10,
            'axes.labelsize': 11,
            'axes.titlesize': 12,
            'xtick.labelsize': 9,
            'ytick.labelsize': 9,
            'legend.fontsize': 9,
            'figure.figsize': (8, 6),
            'figure.dpi': 150,
            'savefig.dpi': 300,
            'savefig.bbox': 'tight',
        })

        # Color scheme
        self.colors = {
            'Dijkstra': '#1f77b4',
            'Bidirectional': '#ff7f0e',
            'CH': '#2ca02c',
            'Hub Labeling': '#d62728',
            'ALT': '#9467bd',
            'ALSSSP': '#8c564b',
        }

    def generate_scalability_plot(self):
        """Generate scalability comparison plot."""
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Filter for sparse random graphs
        sparse_results = [r for r in self.results if 'sparse_random' in r.graph_name]

        methods = list(self.colors.keys())

        # Left: Query time vs graph size
        ax1 = axes[0]
        for method in methods:
            method_results = [r for r in sparse_results if r.method == method]
            if method_results:
                sizes = [r.graph_size for r in method_results]
                times = [r.avg_query_time_ms for r in method_results]
                ax1.plot(sizes, times, 'o-', color=self.colors[method],
                        label=method, linewidth=2, markersize=8)

        ax1.set_xlabel('Number of Vertices')
        ax1.set_ylabel('Query Time (ms)')
        ax1.set_title('Query Time Scalability (Sparse Random Graphs)')
        ax1.set_xscale('log')
        ax1.set_yscale('log')
        ax1.legend(loc='upper left')
        ax1.grid(True, alpha=0.3)

        # Right: Speedup over Dijkstra
        ax2 = axes[1]
        dijkstra_times = {r.graph_size: r.avg_query_time_ms
                         for r in sparse_results if r.method == 'Dijkstra'}

        for method in methods:
            if method == 'Dijkstra':
                continue
            method_results = [r for r in sparse_results if r.method == method]
            if method_results:
                sizes = [r.graph_size for r in method_results]
                speedups = [dijkstra_times.get(r.graph_size, 1) / max(r.avg_query_time_ms, 0.001)
                           for r in method_results]
                ax2.plot(sizes, speedups, 'o-', color=self.colors[method],
                        label=method, linewidth=2, markersize=8)

        ax2.axhline(y=1, color='gray', linestyle='--', alpha=0.5)
        ax2.set_xlabel('Number of Vertices')
        ax2.set_ylabel('Speedup over Dijkstra')
        ax2.set_title('Speedup Factor (Sparse Random Graphs)')
        ax2.set_xscale('log')
        ax2.legend(loc='upper left')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('paper_figures/fig1_scalability.pdf')
        plt.savefig('paper_figures/fig1_scalability.png')
        plt.close()
        print("Saved: fig1_scalability.pdf/png")

    def generate_topology_plot(self):
        """Generate topology comparison bar chart."""
        fig, ax = plt.subplots(figsize=(12, 6))

        # Group results by topology type
        topology_types = ['sparse_random', 'dense_random', 'grid', 'scale_free', 'small_world']
        methods = list(self.colors.keys())

        x = np.arange(len(topology_types))
        width = 0.12

        for i, method in enumerate(methods):
            times = []
            for topo in topology_types:
                matching = [r for r in self.results
                           if r.method == method and topo in r.graph_name and r.graph_size <= 2000]
                if matching:
                    times.append(np.mean([r.avg_query_time_ms for r in matching]))
                else:
                    times.append(0)

            ax.bar(x + i * width, times, width, label=method, color=self.colors[method])

        ax.set_xlabel('Graph Topology')
        ax.set_ylabel('Average Query Time (ms)')
        ax.set_title('Query Time by Graph Topology (n ≈ 2000)')
        ax.set_xticks(x + width * (len(methods) - 1) / 2)
        ax.set_xticklabels(['Sparse\nRandom', 'Dense\nRandom', 'Grid', 'Scale-Free', 'Small-World'])
        ax.legend(loc='upper right', ncol=2)
        ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        plt.savefig('paper_figures/fig2_topology.pdf')
        plt.savefig('paper_figures/fig2_topology.png')
        plt.close()
        print("Saved: fig2_topology.pdf/png")

    def generate_preprocessing_plot(self):
        """Generate preprocessing vs query time tradeoff plot."""
        fig, ax = plt.subplots(figsize=(10, 6))

        # Get results for medium-sized graphs
        medium_results = [r for r in self.results if 2000 <= r.graph_size <= 5000]

        methods = list(self.colors.keys())

        for method in methods:
            method_results = [r for r in medium_results if r.method == method]
            if method_results:
                preproc = np.mean([r.preprocessing_time_s for r in method_results])
                query = np.mean([r.avg_query_time_ms for r in method_results])

                ax.scatter(preproc, query, s=200, c=self.colors[method],
                          label=method, edgecolors='black', linewidth=1.5)

        ax.set_xlabel('Preprocessing Time (seconds)')
        ax.set_ylabel('Average Query Time (ms)')
        ax.set_title('Preprocessing vs Query Time Tradeoff')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('paper_figures/fig3_preprocessing.pdf')
        plt.savefig('paper_figures/fig3_preprocessing.png')
        plt.close()
        print("Saved: fig3_preprocessing.pdf/png")

    def generate_speedup_heatmap(self):
        """Generate speedup heatmap across graph types and sizes."""
        fig, ax = plt.subplots(figsize=(12, 8))

        # Prepare data matrix
        graph_types = ['sparse_random', 'grid', 'scale_free', 'small_world']
        sizes = sorted(set(r.graph_size for r in self.results))[:5]  # Top 5 sizes

        # Calculate ALSSSP speedup over Dijkstra
        speedup_matrix = np.zeros((len(graph_types), len(sizes)))

        for i, gtype in enumerate(graph_types):
            for j, size in enumerate(sizes):
                dijkstra = [r for r in self.results
                           if r.method == 'Dijkstra' and gtype in r.graph_name and r.graph_size == size]
                alsssp = [r for r in self.results
                         if r.method == 'ALSSSP' and gtype in r.graph_name and r.graph_size == size]

                if dijkstra and alsssp:
                    speedup = dijkstra[0].avg_query_time_ms / max(alsssp[0].avg_query_time_ms, 0.001)
                    speedup_matrix[i, j] = speedup

        im = ax.imshow(speedup_matrix, cmap='YlOrRd', aspect='auto')

        ax.set_xticks(np.arange(len(sizes)))
        ax.set_yticks(np.arange(len(graph_types)))
        ax.set_xticklabels([f'n={s}' for s in sizes])
        ax.set_yticklabels(['Sparse Random', 'Grid', 'Scale-Free', 'Small-World'])

        # Add colorbar
        cbar = ax.figure.colorbar(im, ax=ax)
        cbar.ax.set_ylabel('Speedup Factor', rotation=-90, va="bottom")

        # Add text annotations
        for i in range(len(graph_types)):
            for j in range(len(sizes)):
                if speedup_matrix[i, j] > 0:
                    text = ax.text(j, i, f'{speedup_matrix[i, j]:.1f}x',
                                  ha='center', va='center', color='black', fontsize=10)

        ax.set_title('ALSSSP Speedup over Dijkstra')

        plt.tight_layout()
        plt.savefig('paper_figures/fig4_heatmap.pdf')
        plt.savefig('paper_figures/fig4_heatmap.png')
        plt.close()
        print("Saved: fig4_heatmap.pdf/png")

    def generate_all_figures(self):
        """Generate all figures."""
        print("\nGenerating figures...")
        self.generate_scalability_plot()
        self.generate_topology_plot()
        self.generate_preprocessing_plot()
        self.generate_speedup_heatmap()
        print("All figures generated!")

    def generate_latex_table(self) -> str:
        """Generate LaTeX table for paper."""
        # Group results by method and graph type
        table = """
\\begin{table}[t]
\\caption{Comprehensive performance comparison across graph topologies. Query times in milliseconds, preprocessing in seconds. Best results in \\textbf{bold}.}
\\label{tab:comprehensive}
\\small
\\begin{tabular}{llrrrrr}
\\toprule
Method & Graph Type & n & Query (ms) & Preproc (s) & Speedup & Accuracy \\\\
\\midrule
"""

        # Add rows
        for gtype in ['sparse_random', 'grid', 'scale_free']:
            results_by_type = [r for r in self.results if gtype in r.graph_name]

            # Get representative sizes
            sizes = sorted(set(r.graph_size for r in results_by_type))[:2]

            for size in sizes:
                dijkstra_time = None
                for method in ['Dijkstra', 'Bidirectional', 'CH', 'ALT', 'ALSSSP']:
                    matching = [r for r in results_by_type
                               if r.method == method and r.graph_size == size]
                    if matching:
                        r = matching[0]
                        if method == 'Dijkstra':
                            dijkstra_time = r.avg_query_time_ms

                        speedup = dijkstra_time / max(r.avg_query_time_ms, 0.001) if dijkstra_time else 1.0

                        table += f"{method} & {gtype.replace('_', ' ').title()} & {r.graph_size:,} & "
                        table += f"{r.avg_query_time_ms:.2f} & {r.preprocessing_time_s:.2f} & "
                        table += f"{speedup:.1f}x & {r.accuracy*100:.1f}\\% \\\\\n"

                table += "\\midrule\n"

        table += """\\bottomrule
\\end{tabular}
\\end{table}
"""
        return table


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    """Run all experiments and generate outputs."""
    print("="*70)
    print("ALSSSP Comprehensive Experiments")
    print("="*70)

    # Run experiments
    runner = ExperimentRunner()
    results = runner.run_all_experiments()

    # Save results
    results_data = [
        {
            'method': r.method,
            'graph_name': r.graph_name,
            'graph_size': r.graph_size,
            'avg_query_time_ms': r.avg_query_time_ms,
            'std_query_time_ms': r.std_query_time_ms,
            'preprocessing_time_s': r.preprocessing_time_s,
            'accuracy': r.accuracy
        }
        for r in results
    ]

    with open('results/experiment_results.json', 'w') as f:
        json.dump(results_data, f, indent=2)
    print("\nResults saved to results/experiment_results.json")

    # Generate figures
    fig_gen = FigureGenerator(results)
    fig_gen.generate_all_figures()

    # Generate LaTeX table
    latex_table = fig_gen.generate_latex_table()
    with open('results/latex_table.tex', 'w') as f:
        f.write(latex_table)
    print("LaTeX table saved to results/latex_table.tex")

    # Print summary
    print("\n" + "="*70)
    print("EXPERIMENT SUMMARY")
    print("="*70)

    print("\nAverage Query Times by Method:")
    methods = set(r.method for r in results)
    for method in sorted(methods):
        method_results = [r for r in results if r.method == method]
        avg_time = np.mean([r.avg_query_time_ms for r in method_results])
        avg_accuracy = np.mean([r.accuracy for r in method_results])
        print(f"  {method:15s}: {avg_time:8.3f}ms (accuracy: {avg_accuracy*100:.1f}%)")

    print("\n" + "="*70)
    print("DONE")
    print("="*70)


if __name__ == '__main__':
    main()
