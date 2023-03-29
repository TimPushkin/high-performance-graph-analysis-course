import math

import pygraphblas as gb


def count_triangles_per_vertex(graph: gb.Matrix) -> list[int]:
    """
    Counts the number of triangles in which each graph vertex participates.
    :param graph: NxN bool undirected (i.e. symmetric, which is not checked) adjacency
    matrix of a graph.
    :return: list of N numbers of triangles in which the corresponding vertex
    participates, triangles are counted in a single direction, self-loops always form
    triangles.
    """
    if graph.type != gb.BOOL:
        raise ValueError("Adjacency matrix has unexpected type")
    if not graph.square:
        raise ValueError("Adjacency matrix must be square")
    graph = graph.nonzero()

    squared = graph.mxm(graph, semiring=gb.INT64.PLUS_TIMES, mask=graph.S)
    triangles = squared.reduce_vector()
    return [math.ceil(triangles.get(i, default=0) / 2) for i in range(triangles.size)]


def count_triangles_cohen(graph: gb.Matrix) -> int:
    """
    Counts the number of triangles in a graph using Cohen's algorithm.
    :param graph: bool undirected (i.e. symmetric, which is not checked) adjacency
    matrix of a graph.
    :return: the number of unique triangles in a graph, note that if the graph contains
    self loops, the resulting number of triangles may depend on its vertex numeration.
    """
    if graph.type != gb.BOOL:
        raise ValueError("Adjacency matrix has unexpected type")
    if not graph.square:
        raise ValueError("Adjacency matrix must be square")
    graph = graph.nonzero()

    counts = graph.tril().mxm(graph.triu(), semiring=gb.INT64.PLUS_TIMES, mask=graph)
    return math.ceil(counts.reduce_int() / 2)


def count_triangles_sandia(graph: gb.Matrix) -> int:
    """
    Counts the number of triangles in a graph using Sandia's algorithm.
    :param graph: bool undirected (i.e. symmetric, which is not checked) adjacency
    matrix of a graph.
    :return: the number of unique triangles in a graph, note that self loops also form
    triangles.
    """
    if graph.type != gb.BOOL:
        raise ValueError("Adjacency matrix has unexpected type")
    if not graph.square:
        raise ValueError("Adjacency matrix must be square")
    graph = graph.nonzero()

    tril = graph.tril()
    counts = tril.mxm(tril, semiring=gb.INT64.PLUS_TIMES, mask=tril)
    return counts.reduce_int()
