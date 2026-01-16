"""
Road Network Visualization for ALSSSP
======================================

Creates visualizations showing how ALSSSP solves shortest path problems
on road network-like data, demonstrating the bidirectional search
exploration pattern and efficiency gains.

Author: Dr. Ankush Rai
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.animation as animation
from collections import defaultdict
import heapq
import time
import os
import sys
import random

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fast_dijkstra.alsssp import ALSSSP

os.makedirs('paper_figures', exist_ok=True)

INF = float('inf')


class RoadNetworkGenerator:
    """Generates realistic road network-like graphs with coordinates."""

    @staticmethod
    def create_grid_road_network(rows: int, cols: int, seed: int = 42):
        """
        Create a grid-based road network with coordinates.

        Returns:
            n: number of vertices
            edges: list of (u, v, weight) tuples
            coords: array of (x, y) coordinates for each vertex
        """
        np.random.seed(seed)
        random.seed(seed)

        n = rows * cols
        edges = []
        coords = np.zeros((n, 2))

        # Assign coordinates with some perturbation
        for i in range(rows):
            for j in range(cols):
                node = i * cols + j
                # Add slight random perturbation for realism
                coords[node] = [j + np.random.uniform(-0.2, 0.2),
                               i + np.random.uniform(-0.2, 0.2)]

        # Create grid edges (local roads)
        for i in range(rows):
            for j in range(cols):
                node = i * cols + j

                # Right neighbor
                if j < cols - 1:
                    neighbor = i * cols + (j + 1)
                    # Weight based on Euclidean distance + randomness (traffic)
                    dist = np.linalg.norm(coords[node] - coords[neighbor])
                    weight = dist * np.random.uniform(0.8, 1.5)
                    edges.append((node, neighbor, weight))
                    edges.append((neighbor, node, weight))

                # Down neighbor
                if i < rows - 1:
                    neighbor = (i + 1) * cols + j
                    dist = np.linalg.norm(coords[node] - coords[neighbor])
                    weight = dist * np.random.uniform(0.8, 1.5)
                    edges.append((node, neighbor, weight))
                    edges.append((neighbor, node, weight))

        # Add some diagonal shortcuts (expressways)
        num_shortcuts = n // 20
        for _ in range(num_shortcuts):
            i1, j1 = random.randint(0, rows-1), random.randint(0, cols-1)
            i2, j2 = random.randint(0, rows-1), random.randint(0, cols-1)

            # Make sure shortcut spans some distance
            if abs(i1 - i2) + abs(j1 - j2) > 3:
                u = i1 * cols + j1
                v = i2 * cols + j2
                dist = np.linalg.norm(coords[u] - coords[v])
                # Highways are faster (lower weight per distance)
                weight = dist * np.random.uniform(0.3, 0.6)
                edges.append((u, v, weight))
                edges.append((v, u, weight))

        return n, edges, coords


class SearchVisualizer:
    """Visualizes the search exploration patterns."""

    def __init__(self, n: int, edges: list, coords: np.ndarray):
        self.n = n
        self.edges = edges
        self.coords = coords

        # Build adjacency lists
        self.adj_forward = defaultdict(list)
        self.adj_backward = defaultdict(list)

        for u, v, w in edges:
            self.adj_forward[u].append((v, w))
            self.adj_backward[v].append((u, w))

    def trace_dijkstra_exploration(self, source: int, target: int):
        """Trace which nodes Dijkstra explores."""
        dist = [INF] * self.n
        dist[source] = 0.0

        pq = [(0.0, source)]
        exploration_order = []

        while pq:
            d, u = heapq.heappop(pq)

            if d > dist[u]:
                continue

            exploration_order.append(u)

            if u == target:
                break

            for v, w in self.adj_forward[u]:
                if d + w < dist[v]:
                    dist[v] = d + w
                    heapq.heappush(pq, (dist[v], v))

        return exploration_order, dist[target]

    def trace_bidirectional_exploration(self, source: int, target: int):
        """Trace which nodes bidirectional search explores."""
        dist_f = {source: 0.0}
        dist_b = {target: 0.0}

        pq_f = [(0.0, source)]
        pq_b = [(0.0, target)]

        processed_f = set()
        processed_b = set()

        exploration_forward = []
        exploration_backward = []

        best = INF

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
                exploration_forward.append(u)

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
                exploration_backward.append(u)

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

        return exploration_forward, exploration_backward, best

    def create_comparison_visualization(self, source: int, target: int,
                                        output_file: str = 'paper_figures/road_network_comparison.png'):
        """Create side-by-side comparison of Dijkstra vs Bidirectional."""
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))

        # Trace explorations
        dijkstra_nodes, dist_dijk = self.trace_dijkstra_exploration(source, target)
        forward_nodes, backward_nodes, dist_bidir = self.trace_bidirectional_exploration(source, target)

        for ax_idx, (ax, title) in enumerate(zip(axes, ['Standard Dijkstra', 'Bidirectional Search (ALSSSP)'])):
            # Draw all edges in light gray
            edge_segments = []
            for u, v, w in self.edges:
                if u < v:  # Avoid duplicates
                    edge_segments.append([self.coords[u], self.coords[v]])

            lc = LineCollection(edge_segments, colors='lightgray', linewidths=0.5, alpha=0.5)
            ax.add_collection(lc)

            # Draw all nodes as small dots
            ax.scatter(self.coords[:, 0], self.coords[:, 1],
                      c='lightgray', s=10, alpha=0.3, zorder=1)

            if ax_idx == 0:  # Dijkstra
                # Color explored nodes by order
                explored_coords = self.coords[dijkstra_nodes]
                colors = plt.cm.YlOrRd(np.linspace(0.2, 1, len(dijkstra_nodes)))
                ax.scatter(explored_coords[:, 0], explored_coords[:, 1],
                          c=colors, s=30, alpha=0.8, zorder=2)

                explored_count = len(dijkstra_nodes)

            else:  # Bidirectional
                # Forward search in blue
                if forward_nodes:
                    forward_coords = self.coords[forward_nodes]
                    colors_f = plt.cm.Blues(np.linspace(0.3, 1, len(forward_nodes)))
                    ax.scatter(forward_coords[:, 0], forward_coords[:, 1],
                              c=colors_f, s=30, alpha=0.8, zorder=2, label='Forward search')

                # Backward search in green
                if backward_nodes:
                    backward_coords = self.coords[backward_nodes]
                    colors_b = plt.cm.Greens(np.linspace(0.3, 1, len(backward_nodes)))
                    ax.scatter(backward_coords[:, 0], backward_coords[:, 1],
                              c=colors_b, s=30, alpha=0.8, zorder=2, label='Backward search')

                explored_count = len(forward_nodes) + len(backward_nodes)

            # Highlight source and target
            ax.scatter([self.coords[source, 0]], [self.coords[source, 1]],
                      c='green', s=200, marker='o', edgecolors='black',
                      linewidths=2, zorder=5, label='Source')
            ax.scatter([self.coords[target, 0]], [self.coords[target, 1]],
                      c='red', s=200, marker='*', edgecolors='black',
                      linewidths=2, zorder=5, label='Target')

            ax.set_title(f'{title}\n(Explored: {explored_count} nodes, Distance: {dist_dijk if ax_idx==0 else dist_bidir:.2f})',
                        fontsize=14, fontweight='bold')
            ax.set_xlabel('X Coordinate')
            ax.set_ylabel('Y Coordinate')
            ax.legend(loc='upper left')
            ax.set_aspect('equal')
            ax.grid(True, alpha=0.3)

        # Add overall title
        speedup = len(dijkstra_nodes) / max(len(forward_nodes) + len(backward_nodes), 1)
        fig.suptitle(f'Road Network Shortest Path: Dijkstra vs ALSSSP\n'
                    f'ALSSSP explores {speedup:.1f}x fewer nodes',
                    fontsize=16, fontweight='bold')

        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.savefig(output_file.replace('.png', '.pdf'), bbox_inches='tight')
        print(f"Saved: {output_file}")
        plt.close()

        return {
            'dijkstra_nodes': len(dijkstra_nodes),
            'bidir_nodes': len(forward_nodes) + len(backward_nodes),
            'speedup': speedup
        }

    def create_exploration_heatmap(self, num_queries: int = 100,
                                   output_file: str = 'paper_figures/exploration_heatmap.png'):
        """Create heatmap showing where each algorithm explores most."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        dijkstra_visits = np.zeros(self.n)
        bidir_visits = np.zeros(self.n)

        random.seed(42)

        for _ in range(num_queries):
            source = random.randint(0, self.n - 1)
            target = random.randint(0, self.n - 1)

            if source != target:
                dijk_nodes, _ = self.trace_dijkstra_exploration(source, target)
                for node in dijk_nodes:
                    dijkstra_visits[node] += 1

                fwd, bwd, _ = self.trace_bidirectional_exploration(source, target)
                for node in fwd + bwd:
                    bidir_visits[node] += 1

        for ax_idx, (ax, visits, title) in enumerate(zip(
            axes,
            [dijkstra_visits, bidir_visits],
            ['Dijkstra Exploration Frequency', 'ALSSSP Exploration Frequency']
        )):
            # Draw edges
            edge_segments = []
            for u, v, w in self.edges:
                if u < v:
                    edge_segments.append([self.coords[u], self.coords[v]])
            lc = LineCollection(edge_segments, colors='lightgray', linewidths=0.3, alpha=0.3)
            ax.add_collection(lc)

            # Scatter with color by visit count
            scatter = ax.scatter(self.coords[:, 0], self.coords[:, 1],
                               c=visits, cmap='hot', s=20, alpha=0.8)

            plt.colorbar(scatter, ax=ax, label='Visit Count')
            ax.set_title(f'{title}\n(Total visits: {int(visits.sum())})', fontsize=12, fontweight='bold')
            ax.set_xlabel('X Coordinate')
            ax.set_ylabel('Y Coordinate')
            ax.set_aspect('equal')

        plt.suptitle(f'Exploration Pattern Comparison ({num_queries} Random Queries)',
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.savefig(output_file.replace('.png', '.pdf'), bbox_inches='tight')
        print(f"Saved: {output_file}")
        plt.close()


def create_scalability_analysis(sizes=[20, 30, 40, 50],
                               output_file='paper_figures/road_scalability.png'):
    """Analyze scalability on road networks of various sizes."""
    results = []

    for side in sizes:
        print(f"\nTesting {side}x{side} grid road network...")
        n, edges, coords = RoadNetworkGenerator.create_grid_road_network(side, side)

        # Run queries
        random.seed(42)
        num_queries = 50

        dijkstra_times = []
        dijkstra_nodes_explored = []
        bidir_times = []
        bidir_nodes_explored = []
        alsssp_times = []

        vis = SearchVisualizer(n, edges, coords)
        alsssp = ALSSSP(n=n, edges=edges)

        for _ in range(num_queries):
            source = random.randint(0, n-1)
            target = random.randint(0, n-1)

            if source == target:
                continue

            # Dijkstra timing
            start = time.perf_counter()
            dijk_nodes, _ = vis.trace_dijkstra_exploration(source, target)
            dijkstra_times.append((time.perf_counter() - start) * 1000)
            dijkstra_nodes_explored.append(len(dijk_nodes))

            # Bidirectional timing
            start = time.perf_counter()
            fwd, bwd, _ = vis.trace_bidirectional_exploration(source, target)
            bidir_times.append((time.perf_counter() - start) * 1000)
            bidir_nodes_explored.append(len(fwd) + len(bwd))

            # ALSSSP timing
            start = time.perf_counter()
            alsssp.shortest_path(source, target)
            alsssp_times.append((time.perf_counter() - start) * 1000)

        results.append({
            'n': n,
            'side': side,
            'dijkstra_time': np.mean(dijkstra_times),
            'dijkstra_nodes': np.mean(dijkstra_nodes_explored),
            'bidir_time': np.mean(bidir_times),
            'bidir_nodes': np.mean(bidir_nodes_explored),
            'alsssp_time': np.mean(alsssp_times),
        })

        print(f"  n={n}: Dijkstra={results[-1]['dijkstra_time']:.2f}ms, "
              f"Bidir={results[-1]['bidir_time']:.2f}ms, "
              f"ALSSSP={results[-1]['alsssp_time']:.2f}ms")

    # Create visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    ns = [r['n'] for r in results]

    # Query time comparison
    ax1 = axes[0]
    ax1.plot(ns, [r['dijkstra_time'] for r in results], 'b-o',
            label='Dijkstra', linewidth=2, markersize=8)
    ax1.plot(ns, [r['bidir_time'] for r in results], 'g-s',
            label='Bidirectional', linewidth=2, markersize=8)
    ax1.plot(ns, [r['alsssp_time'] for r in results], 'r-^',
            label='ALSSSP', linewidth=2, markersize=8)
    ax1.set_xlabel('Number of Nodes')
    ax1.set_ylabel('Query Time (ms)')
    ax1.set_title('Query Time on Road Networks')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xscale('log')
    ax1.set_yscale('log')

    # Nodes explored
    ax2 = axes[1]
    ax2.plot(ns, [r['dijkstra_nodes'] for r in results], 'b-o',
            label='Dijkstra', linewidth=2, markersize=8)
    ax2.plot(ns, [r['bidir_nodes'] for r in results], 'g-s',
            label='Bidirectional', linewidth=2, markersize=8)
    ax2.set_xlabel('Number of Nodes')
    ax2.set_ylabel('Nodes Explored')
    ax2.set_title('Exploration Efficiency')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Speedup
    ax3 = axes[2]
    speedup_bidir = [r['dijkstra_time'] / max(r['bidir_time'], 0.001) for r in results]
    speedup_alsssp = [r['dijkstra_time'] / max(r['alsssp_time'], 0.001) for r in results]

    x = np.arange(len(ns))
    width = 0.35
    ax3.bar(x - width/2, speedup_bidir, width, label='Bidirectional', color='green', alpha=0.7)
    ax3.bar(x + width/2, speedup_alsssp, width, label='ALSSSP', color='red', alpha=0.7)
    ax3.axhline(y=1, color='gray', linestyle='--')
    ax3.set_xlabel('Network Size')
    ax3.set_ylabel('Speedup over Dijkstra')
    ax3.set_title('Speedup Factor')
    ax3.set_xticks(x)
    ax3.set_xticklabels([f'{r["side"]}×{r["side"]}' for r in results])
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')

    plt.suptitle('Road Network Performance Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.savefig(output_file.replace('.png', '.pdf'), bbox_inches='tight')
    print(f"\nSaved: {output_file}")
    plt.close()

    return results


def main():
    """Generate all road network visualizations."""
    print("="*70)
    print("Road Network Visualization for ALSSSP")
    print("="*70)

    # Create a sample road network
    print("\n1. Creating road network...")
    n, edges, coords = RoadNetworkGenerator.create_grid_road_network(30, 30)
    print(f"   Created network with {n} nodes and {len(edges)} edges")

    # Create visualizer
    vis = SearchVisualizer(n, edges, coords)

    # Choose source and target (opposite corners for maximum path)
    source = 0  # Top-left
    target = n - 1  # Bottom-right

    # Create comparison visualization
    print("\n2. Creating comparison visualization...")
    stats = vis.create_comparison_visualization(source, target)
    print(f"   Dijkstra explored: {stats['dijkstra_nodes']} nodes")
    print(f"   ALSSSP explored:   {stats['bidir_nodes']} nodes")
    print(f"   Speedup:           {stats['speedup']:.2f}x")

    # Create exploration heatmap
    print("\n3. Creating exploration heatmap...")
    vis.create_exploration_heatmap(num_queries=50)

    # Create scalability analysis
    print("\n4. Running scalability analysis...")
    results = create_scalability_analysis(sizes=[15, 20, 30, 40, 50])

    # Create a summary figure
    print("\n5. Creating summary visualization...")
    create_summary_figure(results)

    print("\n" + "="*70)
    print("Road Network Visualizations Complete!")
    print("="*70)
    print("\nGenerated files:")
    print("  - paper_figures/road_network_comparison.pdf/png")
    print("  - paper_figures/exploration_heatmap.pdf/png")
    print("  - paper_figures/road_scalability.pdf/png")
    print("  - paper_figures/road_summary.pdf/png")


def create_summary_figure(results):
    """Create a summary figure for the paper."""
    fig = plt.figure(figsize=(14, 10))

    # Create grid layout
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

    # 1. Query time comparison (top-left)
    ax1 = fig.add_subplot(gs[0, 0])
    ns = [r['n'] for r in results]
    ax1.plot(ns, [r['dijkstra_time'] for r in results], 'b-o',
            label='Standard Dijkstra', linewidth=2, markersize=8)
    ax1.plot(ns, [r['alsssp_time'] for r in results], 'r-^',
            label='ALSSSP', linewidth=2, markersize=8)
    ax1.set_xlabel('Number of Vertices (n)')
    ax1.set_ylabel('Average Query Time (ms)')
    ax1.set_title('Query Time Scalability', fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. Nodes explored (top-right)
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(ns, [r['dijkstra_nodes'] for r in results], 'b-o',
            label='Dijkstra', linewidth=2, markersize=8)
    ax2.plot(ns, [r['bidir_nodes'] for r in results], 'r-^',
            label='ALSSSP (Bidirectional)', linewidth=2, markersize=8)
    ax2.set_xlabel('Number of Vertices (n)')
    ax2.set_ylabel('Average Nodes Explored')
    ax2.set_title('Search Space Reduction', fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. Speedup factors (bottom-left)
    ax3 = fig.add_subplot(gs[1, 0])
    speedups = [r['dijkstra_time'] / max(r['alsssp_time'], 0.001) for r in results]
    node_reduction = [r['dijkstra_nodes'] / max(r['bidir_nodes'], 1) for r in results]

    x = np.arange(len(ns))
    width = 0.35
    bars1 = ax3.bar(x - width/2, speedups, width, label='Time Speedup',
                   color='red', alpha=0.7)
    bars2 = ax3.bar(x + width/2, node_reduction, width, label='Node Reduction',
                   color='blue', alpha=0.7)

    ax3.axhline(y=1, color='gray', linestyle='--', alpha=0.5)
    ax3.set_xlabel('Road Network Size')
    ax3.set_ylabel('Improvement Factor (×)')
    ax3.set_title('ALSSSP Performance Gains', fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels([f'{r["side"]}×{r["side"]}' for r in results])
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax3.annotate(f'{height:.1f}×',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=8)

    # 4. Theoretical vs empirical (bottom-right)
    ax4 = fig.add_subplot(gs[1, 1])

    # Theoretical sqrt(n) reduction
    theoretical = [np.sqrt(n) for n in ns]
    empirical = node_reduction

    ax4.plot(ns, theoretical, 'g--', label='Theoretical √n', linewidth=2)
    ax4.plot(ns, empirical, 'r-o', label='Empirical', linewidth=2, markersize=8)
    ax4.set_xlabel('Number of Vertices (n)')
    ax4.set_ylabel('Node Reduction Factor')
    ax4.set_title('Theory vs Practice', fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.suptitle('ALSSSP Performance on Road Networks\n'
                 'Bidirectional Search Achieves Significant Speedup',
                 fontsize=14, fontweight='bold')

    plt.savefig('paper_figures/road_summary.png', dpi=300, bbox_inches='tight')
    plt.savefig('paper_figures/road_summary.pdf', bbox_inches='tight')
    print("Saved: paper_figures/road_summary.pdf/png")
    plt.close()


if __name__ == '__main__':
    main()
