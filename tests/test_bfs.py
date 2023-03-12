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
