import pygraphblas as gb
import pytest

import utils as testing_utils
from project import utils


@pytest.mark.parametrize(
    "graph, expected",
    testing_utils.load_test_data(
        "test_is_undirected",
        lambda d: (
            testing_utils.matrix_from_dense_list(
                d["graph"],
                gb.types.BOOL if d["matrix_type"] == "BOOL" else gb.types.INT64,
            ),
            d["expected"],
        ),
    ),
    ids=testing_utils.load_test_ids("test_is_undirected"),
)
def test_is_undirected(graph: gb.Matrix, expected: bool):
    actual = utils.is_undirected(graph)

    assert actual == expected
