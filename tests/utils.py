import inspect
import json
import pathlib
from collections.abc import Callable
from typing import TextIO, TypeVar, Union

import pygraphblas as gb


def _open_test_json(
    test_name: str, filename: Union[str, None], add_filename_suffix: bool
) -> TextIO:
    with pathlib.Path(inspect.stack()[2].filename) as f:
        parent = f.parent
        if filename is None:
            filename = f.stem
        if add_filename_suffix:
            filename += f"_{test_name}"
    return open(parent / "data" / f"{filename}.json")


_T = TypeVar("_T")


def load_test_data(
    test_name: str,
    transform: Callable[[dict], _T],
    *,
    data_filename: Union[str, None] = None,
    add_filename_suffix: bool = False,
) -> Union[list[_T], list[tuple]]:
    with _open_test_json(test_name, data_filename, add_filename_suffix) as f:
        test_data = json.load(f)
    return [transform(test_datum) for test_datum in test_data[test_name]]


def load_test_ids(
    test_name: str,
    *,
    data_filename: Union[str, None] = None,
    add_filename_suffix: bool = False,
) -> list[str]:
    with _open_test_json(test_name, data_filename, add_filename_suffix) as f:
        test_data = json.load(f)
    return [test_datum["description"] for test_datum in test_data[test_name]]


def matrix_from_dense_list(
    vs: list[list], typ: gb.types.MetaType = gb.types.BOOL
) -> gb.Matrix:
    if len(vs) == 0:
        return gb.Matrix.sparse(typ, 0, 0)

    rows = []
    cols = []
    vals = []
    for i, row in enumerate(vs):
        for j, v in enumerate(row):
            if v is not None:
                rows.append(i)
                cols.append(j)
                vals.append(v)
    return gb.Matrix.from_lists(rows, cols, vals, typ=typ)
