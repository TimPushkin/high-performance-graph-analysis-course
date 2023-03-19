from typing import Collection

import pygraphblas as gb


def bfs(graph: gb.Matrix, start: int) -> list[int]:
    """
    BFS on a directed graph.
    :param graph: NxN bool adjacency matrix of the graph.
    :param start: index of the starting vertex as it is in the adjacency matrix.
    :return: list of numbers representing the step on which the corresponding vertex was
    visited the first time during BFS, or -1 if it is unreachable; the starting vertex
    is visited on step 0, its adjacent vertices are visited on step 1, and so on.
    """
    if graph.type != gb.BOOL:
        raise ValueError("Adjacency matrix has unexpected type")
    if not graph.square:
        raise ValueError("Adjacency matrix must be square")
    if start < 0 or start >= graph.nrows:
        raise ValueError(
            f"Vertex {start} is out of range for graph of {graph.nrows} vertices"
        )
    graph = graph.nonzero()

    steps = gb.Vector.sparse(gb.INT64, size=graph.nrows)
    steps[start] = 0
    front = gb.Vector.sparse(gb.BOOL, size=graph.nrows)
    front[start] = True

    step = 1
    while front.nvals:
        front.vxm(graph, out=front, mask=steps.S, desc=gb.descriptor.RC)
        steps.assign_scalar(step, mask=front)
        step += 1

    return [steps.get(i, default=-1) for i in range(steps.size)]


def msbfs(graph: gb.Matrix, starts: Collection[int]) -> list[tuple[int, list[int]]]:
    """
    Multi-source BFS on a directed graph.
    :param graph: NxN bool adjacency matrix of the graph.
    :param starts: indices of starting vertices as they are in the adjacency matrix.
    :return: list of (start, parents) pairs where for each start vertex there is a list
    of N parent vertices for the corresponding vertex. If there are several possible
    parent vertices, the one with the smaller index will be picked. The start vertices
    will have -1 in these lists while unreachable vertices will have -2.
    """
    if graph.type != gb.BOOL:
        raise ValueError("Adjacency matrix has unexpected type")
    if not graph.square:
        raise ValueError("Adjacency matrix must be square")
    graph = graph.nonzero()

    parents = gb.Matrix.sparse(gb.INT64, nrows=len(starts), ncols=graph.ncols)
    front = gb.Matrix.sparse(gb.INT64, nrows=len(starts), ncols=graph.ncols)
    for row, start in enumerate(starts):
        if start < 0 or start >= graph.ncols:
            raise ValueError(
                f"Vertex {start} is out of range for graph of {graph.ncols} vertices"
            )
        parents[row, start] = -1
        front[row, start] = start

    while front.nvals > 0:
        front.mxm(
            graph,
            out=front,
            semiring=gb.INT64.MIN_FIRST,
            mask=parents.S,
            desc=gb.descriptor.RC,
        )
        parents.assign(front, mask=front.S)
        front.apply(gb.INT64.POSITIONJ, out=front, mask=front.S)

    return [
        (start, [parents.get(row, col, default=-2) for col in range(graph.ncols)])
        for row, start in enumerate(starts)
    ]
