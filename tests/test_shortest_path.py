import pygraphblas as gb
import pytest

import utils
from project import shortest_path


@pytest.mark.parametrize(
    "graph, start",
    utils.load_test_data(
        "test_sssp_incorrect_inputs",
        lambda d: (
            utils.matrix_from_dense_list(
                d["graph"],
                gb.types.FP64 if d["matrix_type"] == "FP64" else gb.types.BOOL,
            ),
            d["start"],
        ),
    ),
    ids=utils.load_test_ids("test_sssp_incorrect_inputs"),
)
def test_sssp_incorrect_inputs(graph: gb.Matrix, start: int):
    with pytest.raises(ValueError):
        shortest_path.sssp(graph, start)


@pytest.mark.parametrize(
    "graph, starts",
    utils.load_test_data(
        "test_mssp_incorrect_inputs",
        lambda d: (
            utils.matrix_from_dense_list(
                d["graph"],
                gb.types.FP64 if d["matrix_type"] == "FP64" else gb.types.BOOL,
            ),
            d["starts"],
        ),
    ),
    ids=utils.load_test_ids("test_mssp_incorrect_inputs"),
)
def test_mssp_incorrect_inputs(graph: gb.Matrix, starts: list[int]):
    with pytest.raises(ValueError):
        shortest_path.mssp(graph, starts)


@pytest.mark.parametrize(
    "graph",
    utils.load_test_data(
        "test_apsp_incorrect_inputs",
        lambda d: utils.matrix_from_dense_list(
            d["graph"], gb.types.FP64 if d["matrix_type"] == "FP64" else gb.types.BOOL
        ),
    ),
    ids=utils.load_test_ids("test_apsp_incorrect_inputs"),
)
def test_apsp_incorrect_inputs(graph: gb.Matrix):
    with pytest.raises(ValueError):
        shortest_path.apsp(graph)


@pytest.mark.parametrize(
    "graph, start, expected",
    utils.load_test_data(
        "test_sssp_correct_inputs",
        lambda d: (
            utils.matrix_from_dense_list(
                [
                    [float(v) if v is not None else None for v in row]
                    for row in d["graph"]
                ],
                typ=gb.types.FP64,
            ),
            d["start"],
            [float(x) for x in d["expected"]],
        ),
    ),
    ids=utils.load_test_ids("test_sssp_correct_inputs"),
)
def test_sssp_correct_inputs(graph: gb.Matrix, start: int, expected: list[float]):
    actual = shortest_path.sssp(graph, start)

    assert actual == expected


@pytest.mark.parametrize(
    "graph, starts, expected",
    utils.load_test_data(
        "test_mssp_correct_inputs",
        lambda d: (
            utils.matrix_from_dense_list(
                [
                    [float(v) if v is not None else None for v in row]
                    for row in d["graph"]
                ],
                typ=gb.types.FP64,
            ),
            d["starts"],
            [(start, [float(x) for x in dists]) for [start, dists] in d["expected"]],
        ),
    ),
    ids=utils.load_test_ids("test_mssp_correct_inputs"),
)
def test_mssp_correct_inputs(
    graph: gb.Matrix, starts: list[int], expected: list[float]
):
    actual = shortest_path.mssp(graph, starts)

    assert actual == expected


@pytest.mark.parametrize(
    "graph, expected",
    utils.load_test_data(
        "test_apsp_correct_inputs",
        lambda d: (
            utils.matrix_from_dense_list(
                [
                    [float(v) if v is not None else None for v in row]
                    for row in d["graph"]
                ],
                typ=gb.types.FP64,
            ),
            [(start, [float(x) for x in dists]) for [start, dists] in d["expected"]],
        ),
    ),
    ids=utils.load_test_ids("test_apsp_correct_inputs"),
)
def test_apsp_correct_inputs(graph: gb.Matrix, expected: list[float]):
    actual = shortest_path.apsp(graph)

    assert actual == expected
