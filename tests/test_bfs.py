import pygraphblas as gb
import pytest

import utils
from project import bfs


@pytest.mark.parametrize(
    "graph, start",
    utils.load_test_data(
        "test_bfs_incorrect_inputs",
        lambda d: (
            utils.matrix_from_dense_list(
                d["graph"],
                gb.types.BOOL if d["matrix_type"] == "BOOL" else gb.types.INT64,
            ),
            d["start"],
        ),
    ),
    ids=utils.load_test_ids("test_bfs_incorrect_inputs"),
)
def test_bfs_incorrect_inputs(graph: gb.Matrix, start: int):
    with pytest.raises(ValueError):
        bfs.bfs(graph, start)


@pytest.mark.parametrize(
    "graph, start, expected",
    utils.load_test_data(
        "test_bfs_correct_inputs",
        lambda d: (utils.matrix_from_dense_list(d["graph"]), d["start"], d["expected"]),
    ),
    ids=utils.load_test_ids("test_bfs_correct_inputs"),
)
def test_bfs_correct_inputs(graph: gb.Matrix, start: int, expected: list[int]):
    actual = bfs.bfs(graph, start)

    assert actual == expected


@pytest.mark.parametrize(
    "graph, starts",
    utils.load_test_data(
        "test_msbfs_incorrect_inputs",
        lambda d: (
            utils.matrix_from_dense_list(
                d["graph"],
                gb.types.BOOL if d["matrix_type"] == "BOOL" else gb.types.INT64,
            ),
            d["starts"],
        ),
    ),
    ids=utils.load_test_ids("test_msbfs_incorrect_inputs"),
)
def test_msbfs_incorrect_inputs(graph: gb.Matrix, starts: list[int]):
    with pytest.raises(ValueError):
        bfs.msbfs(graph, starts)


@pytest.mark.parametrize(
    "graph, starts, expected",
    utils.load_test_data(
        "test_msbfs_correct_inputs",
        lambda d: (
            utils.matrix_from_dense_list(d["graph"]),
            d["starts"],
            [tuple(pair_list) for pair_list in d["expected"]],
        ),
    ),
    ids=utils.load_test_ids("test_msbfs_correct_inputs"),
)
def test_msbfs_correct_inputs(graph: gb.Matrix, starts: list[int], expected: list[int]):
    actual = bfs.msbfs(graph, starts)

    assert actual == expected
