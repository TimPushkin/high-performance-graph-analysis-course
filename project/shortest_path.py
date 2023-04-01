import math
from typing import Collection

import pygraphblas as gb


def sssp(graph: gb.Matrix, start: int) -> list[int]:
    """
    Finds single-source shortest paths using an algebraic Bellman-Ford algorithm.
    :param graph: NxN real adjacency matrix of the graph with edge lengths, where a
    no-value is treated as +inf (i.e. no path). Elements on the main diagonal should
    usually all be zeroes until other task-specific requirements are present. The graph
    should not have negative-length cycles.
    :param start: index of the starting vertex as it is in the adjacency matrix.
    :return: list of numbers representing the length of the sortest path from start to
    the corresponding vertex, or +inf if it is unreachable.
    """
    return mssp(graph, [start])[0][1]


def mssp(graph: gb.Matrix, starts: Collection[int]) -> list[tuple[int, list[int]]]:
    """
    Finds multiple-source shortest paths using an algebraic Bellman-Ford algorithm.
    :param graph: NxN real adjacency matrix of the graph with edge lengths, where a
    no-value is treated as +inf (i.e. no path). Elements on the main diagonal should
    usually all be zeroes until other task-specific requirements are present. The graph
    should not have negative-length cycles reachable from any of the start vertices.
    :param starts: indices of starting vertices as they are in the adjacency matrix.
    :return: list of (start, distances) pairs where for each start vertex there is a
    list of numbers representing the length of the sortest path from this start to
    the corresponding vertex, or +inf if it is unreachable.
    """
    if graph.type != gb.FP64:
        raise ValueError("Adjacency matrix has unexpected type")
    if not graph.square:
        raise ValueError("Adjacency matrix must be square")
    graph.select("!=", thunk=math.inf, out=graph)  # Remove explicit infinities

    dists = gb.Matrix.sparse(gb.FP64, nrows=len(starts), ncols=graph.ncols)
    for row, start in enumerate(starts):
        if start < 0 or start >= graph.ncols:
            raise ValueError(
                f"Vertex {start} is out of range for graph of {graph.ncols} vertices"
            )
        dists[row, start] = 0

    for _ in range(graph.ncols - 1):
        dists.mxm(graph, semiring=gb.FP64.MIN_PLUS, out=dists)

    # Detect reachable negative-length cycles
    if dists.isne(dists.mxm(graph, semiring=gb.FP64.MIN_PLUS)):
        raise ValueError("Graph has a reachable negative-length cycle")

    return [
        (start, [dists.get(row, col, default=math.inf) for col in range(graph.ncols)])
        for row, start in enumerate(starts)
    ]


def apsp(graph: gb.Matrix) -> list[tuple[int, list[int]]]:
    """
    Finds all-pairs shortest paths using an algebraic Floyd–Warshall algorithm.
    :param graph: NxN real adjacency matrix of the graph with edge lengths, where a
    no-value is treated as +inf (i.e. no path). Elements on the main diagonal should
    usually all be zeroes until other task-specific requirements are present. The graph
    should not have negative-length cycles reachable from any of the start vertices.
    :return: list of (start, distances) pairs where for each start vertex there is a
    list of numbers representing the length of the sortest path from this start to
    the corresponding vertex, or +inf if it is unreachable.
    """
    if graph.type != gb.FP64:
        raise ValueError("Adjacency matrix has unexpected type")
    if not graph.square:
        raise ValueError("Adjacency matrix must be square")
    graph.select("!=", thunk=math.inf, out=graph)  # Remove explicit infinities

    dists = graph.dup()

    for k in range(graph.ncols):
        step = dists.extract_matrix(col_index=k).mxm(
            dists.extract_matrix(row_index=k), semiring=gb.FP64.MIN_PLUS
        )
        dists.eadd(step, add_op=gb.FP64.MIN, out=dists)

    # Detect reachable negative-length cycles
    for k in range(graph.ncols):
        step = dists.extract_matrix(col_index=k).mxm(
            dists.extract_matrix(row_index=k), semiring=gb.FP64.MIN_PLUS
        )
        if dists.isne(dists.eadd(step, add_op=gb.FP64.MIN)):
            raise ValueError("Graph has a reachable negative-length cycle")

    return [
        (row, [dists.get(row, col, default=math.inf) for col in range(graph.ncols)])
        for row in range(graph.nrows)
    ]
