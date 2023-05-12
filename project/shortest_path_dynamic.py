import heapq
import itertools
import math
from typing import Hashable

import boltons.queueutils
import networkx as nx


def dijkstra_sssp(graph: nx.Graph, start: Hashable) -> dict[Hashable, float]:
    """
    Finds single-source shortest paths using classical Dijkstra's algorithm.
    :param graph: graph to run the algorithm on. If its edges have weight attribute, it
    will be used as weights, or 1 will be assumed otherwise. The weights are assumed to
    be non-negative.
    :param start: vertex of the graph from which the paths are calculated.
    :return: distances from start to each vertex in the graph, or +inf if a vertex is
    unreachable.
    """
    if graph is nx.MultiGraph:
        raise ValueError("Multigraphs are unsupported")

    dists = {node: math.inf for node in graph.nodes}
    dists[start] = 0

    queue = [(0, start)]

    while queue:
        dist, node = heapq.heappop(queue)

        if dist > dists[node]:
            continue

        for neighbor in graph.neighbors(node):  # neighbours == successors when directed
            weight = graph[node][neighbor].get("weight", 1)
            new_dist = dists[node] + weight

            if new_dist < dists[neighbor]:
                dists[neighbor] = new_dist
                heapq.heappush(queue, (new_dist, neighbor))

    return dists


class DynamicSSSP:
    """
    Dynamic Dijkstra-like single-source shortest paths algorithm, based on
    https://doi.org/10.1006/jagm.1996.0046.
    """

    def __init__(self, graph: nx.DiGraph, start: Hashable):
        """
        Initializes the algorithm to run on the provided graph and start vertex.
        :param graph: graph to run the algorithm on. If its edges have weight attribute,
        it will be used as weights, or 1 will be assumed otherwise. The weights are
        assumed to be non-negative.
        :param start: vertex of the graph from which the paths are calculated.
        """
        self._graph = graph
        self._start = start
        # Current distances
        self._dists = dijkstra_sssp(graph, start)
        # Set of all possibly inconsistent vertices w.r.t. current graph and dists
        self._modified_nodes = set()

    def insert_edge(self, u: Hashable, v: Hashable, weight: float = 1):
        """Insert or update the edge in the graph."""
        self._graph.add_edge(u, v, weight=weight)
        self._modified_nodes.add(v)

        # NetworkX allows adding edges to non-existent vertices
        if u not in self._dists:
            self._dists[u] = math.inf
        if v not in self._dists:
            self._dists[v] = math.inf

    def delete_edge(self, u: Hashable, v: Hashable):
        """Delete the edge from the graph."""
        self._graph.remove_edge(u, v)
        self._modified_nodes.add(v)

    def query_dists(self) -> dict[Hashable, float]:
        """
        Returns the distances from start to each vertex in the graph, or +inf if a
        vertex is unreachable.
        """
        if len(self._modified_nodes) > 0:
            self._update_dists()
            self._modified_nodes = set()
        return self._dists

    def _update_dists(self):
        """Applies the accumulated graph updates to the stored distances"""
        rhs: dict[Hashable, float] = {}
        # This queue implementation ensures element uniqueness: when an element being
        # added is already in the queue, its priority is replaced instead -- this
        # corresponds to the behavior of AdjustHeap from the original algorithm, though
        # asymptotic estimations may vary
        queue = boltons.queueutils.HeapPriorityQueue(priority_key=lambda x: x)
        for u in self._modified_nodes:
            rhs[u] = self._calculate_rhs(u)
            if rhs[u] != self._dists[u]:
                queue.add(u, priority=min(rhs[u], self._dists[u]))

        while len(queue) > 0:
            u = queue.pop()

            if rhs[u] < self._dists[u]:
                self._dists[u] = rhs[u]
                to_update_rhs = self._graph.successors(u)
            else:
                self._dists[u] = math.inf
                to_update_rhs = itertools.chain(self._graph.successors(u), [u])

            for v in to_update_rhs:
                rhs[v] = self._calculate_rhs(v)
                if rhs[v] != self._dists[v]:
                    queue.add(v, priority=min(rhs[v], self._dists[v]))
                else:
                    if v in queue._entry_map:  # No public way to check this efficiently
                        queue.remove(v)

    def _calculate_rhs(self, u: Hashable):
        """Calculates the current rhs value for the provided vertex."""
        if u == self._start:
            return 0
        return min(
            (
                self._dists[v] + self._graph[v][u].get("weight", 1)
                for v in self._graph.predecessors(u)
            ),
            default=math.inf,
        )
