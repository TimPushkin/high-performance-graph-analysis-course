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

    steps = gb.Vector.sparse(gb.INT64, size=graph.nrows)
    steps[start] = 0
    front = gb.Vector.sparse(gb.BOOL, size=graph.nrows)
    front[start] = True

    step = 1
    while True:
        old_steps_nvals = steps.nvals

        front.vxm(graph, out=front)

        new_visited_mask = front.eadd(steps.S, add_op=gb.BOOL.GT, mask=front.S)
        steps.assign_scalar(step, mask=new_visited_mask)
        step += 1

        if steps.nvals == old_steps_nvals:
            break

    return [steps.get(i, default=-1) for i in range(steps.size)]
