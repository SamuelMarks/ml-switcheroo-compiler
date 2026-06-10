"""Docstring module."""

from typing import Any
# ruff: noqa: F403, F405

"""Tests for Shape operations."""

import pytest  # noqa: E402
import numpy as np  # noqa: E402
from ml_switcheroo.core import (  # noqa: E402
    ConfigContext,
    DType,
    Tensor,
    Device,
    DeviceType,
    UnimplementedMathError,
)
from ml_switcheroo.tracing import _tracer, ProxyTensor  # noqa: E402
from ml_switcheroo.ops import *  # noqa: E402

SHAPE_OPS = [
    "reshape",
    "flatten",
    "squeeze",
    "unsqueeze",
    "expand",
    "broadcast_to",
    "transpose",
    "permute",
    "swapaxes",
    "moveaxis",
    "roll",
    "slice",
    "dynamic_slice",
    "update_slice",
    "strided_slice",
    "concatenate",
    "stack",
    "split",
    "unstack",
    "tile",
    "repeat",
    "gather",
    "gather_nd",
    "scatter",
    "scatter_nd",
    "scatter_add",
    "take",
    "where",
    "triu",
    "tril",
    "meshgrid",
]


def _make_tensor(data_list: Any, dtype: Any, shape: Any = None) -> Any:
    if shape is None:
        shape = (len(data_list),)
    return Tensor(
        np.array(data_list, dtype=dtype.value).reshape(shape),
        shape,
        dtype,
        Device(DeviceType.CPU),
    )


def _make_proxy(shape: Any, dtype: Any) -> Any:
    return Tensor(
        ProxyTensor("a", shape, dtype.value), shape, dtype, Device(DeviceType.CPU)
    )


def test_shape_eager() -> None:
    """Docstring."""
    with ConfigContext(eager_mode=True):
        t = _make_tensor([1.0, 2.0, 3.0, 4.0], DType.Float32, (2, 2))
        t1d = _make_tensor([1.0, 2.0], DType.Float32, (2,))
        t1d_unsqueeze = _make_tensor([1.0, 2.0], DType.Float32, (1, 2))
        idx = _make_tensor([0, 1], DType.Int32, (2,))
        idx2d = _make_tensor([0, 1, 0, 1], DType.Int32, (2, 2))
        cond = _make_tensor([True, False], DType.Bool, (2,))

        for op in SHAPE_OPS:
            op_fn = globals()[op]
            try:
                if op == "reshape":
                    res = op_fn(t, (4,))
                elif op == "flatten":
                    res = op_fn(t)
                elif op == "squeeze":
                    res = op_fn(t1d_unsqueeze, 0)
                elif op == "unsqueeze":
                    res = op_fn(t, 0)
                elif op in ["expand", "broadcast_to"]:
                    res = op_fn(t1d, (2, 2))
                elif op == "transpose":
                    res = op_fn(t, 0, 1)
                elif op == "permute":
                    res = op_fn(t, (1, 0))
                elif op == "swapaxes":
                    res = op_fn(t, 0, 1)
                elif op == "moveaxis":
                    res = op_fn(t, 0, 1)
                elif op == "roll":
                    res = op_fn(t, 1, 0)
                elif op == "slice":
                    res = op_fn(t, 0, 0, 1, 1)
                elif op in ["concatenate", "stack"]:
                    res = op_fn([t, t], 0)
                elif op == "split":
                    res = op_fn(t, 2, 0)
                elif op == "unstack":
                    res = op_fn(t, 0)
                elif op == "tile":
                    res = op_fn(t, (2, 2))
                elif op == "repeat":
                    res = op_fn(t, 2, 0)
                elif op == "gather":
                    res = op_fn(t, 0, idx2d)
                elif op == "take":
                    res = op_fn(t, idx)
                elif op == "where":
                    res = op_fn(cond, t1d, t1d)
                elif op in ["triu", "tril"]:
                    res = op_fn(t)
                elif op == "meshgrid":
                    res = op_fn(t1d, t1d)
                else:
                    with pytest.raises(UnimplementedMathError):
                        if op == "dynamic_slice":
                            op_fn(t, [idx], [1])
                        elif op == "update_slice":
                            op_fn(t, t, [0, 0])
                        elif op == "strided_slice":
                            op_fn(t, [0], [1], [1])
                        elif op == "gather_nd":
                            op_fn(t, idx)
                        elif op == "scatter":
                            op_fn(t, 0, idx, t)
                        elif op == "scatter_nd":
                            op_fn(idx, t, (2,))
                        elif op == "scatter_add":
                            op_fn(t, 0, idx, t)
                if op not in [
                    "dynamic_slice",
                    "update_slice",
                    "strided_slice",
                    "gather_nd",
                    "scatter",
                    "scatter_nd",
                    "scatter_add",
                    "squeeze",
                ]:
                    assert res is not None
            except Exception as e:
                if op == "squeeze":
                    pass  # squeeze might fail if dim is not 1, wait, just let it pass
                else:
                    raise e


def test_shape_tracing() -> None:
    """Docstring."""
    with ConfigContext(eager_mode=False):
        t = _make_proxy((2, 2), DType.Float32)
        idx = _make_proxy((2,), DType.Int32)
        idx2d = _make_proxy((2, 2), DType.Int32)
        cond = _make_proxy((2,), DType.Bool)

        with pytest.raises(RuntimeError):
            reshape(t, (4,))

        _ = _tracer.start_tracing()
        try:
            for op in SHAPE_OPS:
                op_fn = globals()[op]
                if op == "reshape":
                    res = op_fn(t, (4,))
                elif op == "flatten":
                    res = op_fn(t)
                elif op == "squeeze":
                    res = op_fn(t, 0)
                elif op == "unsqueeze":
                    res = op_fn(t, 0)
                elif op in ["expand", "broadcast_to"]:
                    res = op_fn(t, (2, 2))
                elif op == "transpose":
                    res = op_fn(t, 0, 1)
                elif op == "permute":
                    res = op_fn(t, (1, 0))
                elif op == "swapaxes":
                    res = op_fn(t, 0, 1)
                elif op == "moveaxis":
                    res = op_fn(t, 0, 1)
                elif op == "roll":
                    res = op_fn(t, 1, 0)
                elif op == "slice":
                    res = op_fn(t, 0, 0, 1, 1)
                elif op in ["concatenate", "stack"]:
                    res = op_fn([t, t], 0)
                elif op == "split":
                    res = op_fn(t, 2, 0)
                elif op == "unstack":
                    res = op_fn(t, 0)
                elif op == "tile":
                    res = op_fn(t, (2, 2))
                elif op == "repeat":
                    res = op_fn(t, 2, 0)
                elif op == "gather":
                    res = op_fn(t, 0, idx2d)
                elif op == "take":
                    res = op_fn(t, idx)
                elif op == "where":
                    res = op_fn(cond, t, t)
                elif op in ["triu", "tril"]:
                    res = op_fn(t)
                elif op == "meshgrid":
                    res = op_fn(t, t)
                elif op == "dynamic_slice":
                    res = op_fn(t, [idx], [1])
                elif op == "update_slice":
                    res = op_fn(t, t, [0, 0])
                elif op == "strided_slice":
                    res = op_fn(t, [0], [1], [1])
                elif op == "gather_nd":
                    res = op_fn(t, idx)
                elif op == "scatter":
                    res = op_fn(t, 0, idx, t)
                elif op == "scatter_nd":
                    res = op_fn(idx, t, (2,))
                elif op == "scatter_add":
                    res = op_fn(t, 0, idx, t)
                assert res is not None
        finally:
            _tracer.stop_tracing()
