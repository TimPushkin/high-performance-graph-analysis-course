from typing import Callable, Union

import pygraphblas as gb
import pytest

import utils
from project import triangles


@pytest.mark.parametrize(
    "algo",
    [
        triangles.count_triangles_per_vertex,
        triangles.count_triangles_cohen,
        triangles.count_triangles_sandia,
    ],
)
@pytest.mark.parametrize(
    "graph",
    utils.load_test_data(
        "test_incorrect_inputs",
        lambda d: utils.matrix_from_dense_list(
            d["graph"],
            gb.types.BOOL if d["matrix_type"] == "BOOL" else gb.types.INT64,
        ),
    ),
    ids=utils.load_test_ids("test_incorrect_inputs"),
)
def test_incorrect_inputs(
    algo: Callable[[gb.Matrix], Union[list[int], int]], graph: gb.Matrix
):
    with pytest.raises(ValueError):
        algo(graph)


@pytest.mark.parametrize(
    "graph, expected",
    utils.load_test_data(
        "test_per_vertex",
        lambda d: (utils.matrix_from_dense_list(d["graph"]), d["expected"]),
    ),
    ids=utils.load_test_ids("test_per_vertex"),
)
def test_per_vertex(graph: gb.Matrix, expected: list[int]):
    actual = triangles.count_triangles_per_vertex(graph)

    assert actual == expected


@pytest.mark.parametrize(
    "graph, expected",
    utils.load_test_data(
        "test_overall",
        lambda d: (utils.matrix_from_dense_list(d["graph"]), d["expected_cohen"]),
    ),
    ids=utils.load_test_ids("test_overall"),
)
def test_cohen(graph: gb.Matrix, expected: int):
    actual = triangles.count_triangles_cohen(graph)

    assert actual == expected


@pytest.mark.parametrize(
    "graph, expected",
    utils.load_test_data(
        "test_overall",
        lambda d: (utils.matrix_from_dense_list(d["graph"]), d["expected_sandia"]),
    ),
    ids=utils.load_test_ids("test_overall"),
)
def test_sandia(graph: gb.Matrix, expected: int):
    actual = triangles.count_triangles_sandia(graph)

    assert actual == expected
