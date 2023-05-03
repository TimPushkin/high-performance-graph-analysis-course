from typing import Union

import networkx as nx
import pytest

import utils
from project import shortest_path_dynamic


@pytest.mark.parametrize(
    "graph, start, expected",
    utils.load_test_data(
        "test_dijkstra_sssp",
        lambda d: (
            nx.node_link_graph(d["graph"], directed=True, multigraph=False),
            d["start"],
            {v: float(l) for v, l in d["expected"].items()},
        ),
    ),
    ids=utils.load_test_ids("test_dijkstra_sssp"),
)
def test_dijkstra_sssp(graph: nx.DiGraph, start: str, expected: dict[str, float]):
    actual = shortest_path_dynamic.dijkstra_sssp(graph, start)

    assert actual == expected


@pytest.mark.parametrize(
    "graph, start, update_chunks, expecteds",
    utils.load_test_data(
        "test_dynamic_sssp",
        lambda d: (
            nx.node_link_graph(d["graph"], directed=True, multigraph=False),
            d["start"],
            d["updates"],
            [{v: float(l) for v, l in exp.items()} for exp in d["expecteds"]],
        ),
    ),
    ids=utils.load_test_ids("test_dynamic_sssp"),
)
def test_dynamic_sssp(
    graph: nx.DiGraph,
    start: str,
    update_chunks: list[list[Union[tuple[str, str, float], tuple[str, str]]]],
    expecteds: list[dict[str, float]],
):
    algo = shortest_path_dynamic.DynamicSSSP(graph, start)

    actual = algo.query_dists()
    assert actual == expecteds[0]

    for updates, expected in zip(update_chunks, expecteds[1:]):
        for update in updates:
            if len(update) == 3:
                algo.insert_edge(*update)
            elif len(update) == 2:
                algo.delete_edge(*update)
            else:
                raise ValueError("Illegal update size")

        actual = algo.query_dists()
        assert actual == expected
