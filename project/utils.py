import pygraphblas as gb


def is_undirected(graph: gb.Matrix) -> bool:
    """
    Checks whether a graph is undirected, i.e. its matrix is symmetric.
    :param graph: bool adjacency matrix of the graph.
    :return: True is the graph is undirected and false otherwise.
    """
    for i, j in zip(graph.I, graph.J):
        forward = graph.get(i, j, default=graph.type.default_zero)
        backward = graph.get(j, i, default=graph.type.default_zero)
        if forward != backward:
            return False
    return True
