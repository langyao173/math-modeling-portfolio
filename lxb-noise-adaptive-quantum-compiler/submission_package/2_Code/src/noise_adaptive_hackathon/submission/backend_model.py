from __future__ import annotations

from dataclasses import dataclass
import math

import networkx as nx

from noise_adaptive_hackathon.formats.backend import CompetitionBackend


@dataclass(frozen=True)
class BackendNoiseModel:
    backend: CompetitionBackend
    graph: nx.Graph
    coupling_edges: frozenset[tuple[int, int]]
    neighbors: dict[int, tuple[int, ...]]
    hop_lengths: dict[int, dict[int, int]]
    hop_paths: dict[int, dict[int, list[int]]]
    noise_lengths: dict[int, dict[int, float]]
    noise_paths: dict[int, dict[int, list[int]]]

    @classmethod
    def from_backend(cls, backend: CompetitionBackend) -> "BackendNoiseModel":
        graph = backend.graph()
        return cls(
            backend=backend,
            graph=graph,
            coupling_edges=frozenset(tuple(sorted(edge)) for edge in backend.coupling_map),
            neighbors={node: tuple(sorted(graph.neighbors(node))) for node in graph.nodes},
            hop_lengths=dict(nx.all_pairs_shortest_path_length(graph)),
            hop_paths=dict(nx.all_pairs_shortest_path(graph)),
            noise_lengths=dict(nx.all_pairs_dijkstra_path_length(graph, weight="cost")),
            noise_paths=dict(nx.all_pairs_dijkstra_path(graph, weight="cost")),
        )

    def is_coupled(self, left: int, right: int) -> bool:
        return tuple(sorted((left, right))) in self.coupling_edges

    def readout_cost(self, qubit: int) -> float:
        return -math.log(max(1.0 - self.backend.readout(qubit), 1e-12))

    def edge_cost(self, left: int, right: int) -> float:
        return -math.log(max(1.0 - self.backend.edge_error(left, right), 1e-12))

    def swap_cost(self, left: int, right: int) -> float:
        return 3.0 * self.edge_cost(left, right)

    def hop_distance(self, left: int, right: int) -> int:
        return self.hop_lengths[left][right]

    def noise_distance(self, left: int, right: int) -> float:
        return self.noise_lengths[left][right]

    def noise_path(self, left: int, right: int) -> list[int]:
        return list(self.noise_paths[left][right])

    def hop_path(self, left: int, right: int) -> list[int]:
        return list(self.hop_paths[left][right])
