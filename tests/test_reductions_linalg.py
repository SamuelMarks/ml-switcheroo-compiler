"""Docstring module."""

from typing import Any
# ruff: noqa: F403, F405

"""Tests for Reductions and Linear Algebra operations."""

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

REDUCTIONS = [
    "sum",
    "prod",
    "mean",
    "variance",
    "std",
    "max",
    "min",
    "argmax",
    "argmin",
    "all",
    "any",
    "logsumexp",
    "count_nonzero",
    "norm",
]

LINALG = [
    "matmul",
    "dot",
    "tensordot",
    "vdot",
    "inner",
    "outer",
    "einsum",
    "cholesky",
    "svd",
    "qr",
    "inv",
    "pinv",
    "det",
    "slogdet",
    "eigh",
    "eigvalsh",
    "matrix_power",
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


def test_reductions_eager() -> None:
    """Docstring."""
    with ConfigContext(eager_mode=True):
        t_float = _make_tensor([0.5, 0.2, 1.0, 2.0], DType.Float32, (2, 2))

        for op_name in REDUCTIONS:
            op_fn = globals()[op_name]
            if op_name == "logsumexp":
                with pytest.raises(UnimplementedMathError):
                    op_fn(t_float)
            else:
                res = op_fn(t_float)
                assert isinstance(res, Tensor)


def test_reductions_tracing() -> None:
    """Docstring."""
    with ConfigContext(eager_mode=False):
        t_float = _make_proxy((2, 2), DType.Float32)

        with pytest.raises(RuntimeError):
            sum(t_float)

        _ = _tracer.start_tracing()
        try:
            for op_name in REDUCTIONS:
                op_fn = globals()[op_name]
                res = op_fn(t_float, axis=0, keepdims=True)
                res2 = op_fn(t_float, axis=1, keepdims=False)
                assert isinstance(res2, Tensor)
                res3 = op_fn(t_float, axis=(0, 1), keepdims=False)
                assert isinstance(res3, Tensor)
                res4 = op_fn(t_float, axis=None, keepdims=True)
                assert isinstance(res4, Tensor)
                assert isinstance(res, Tensor)
        finally:
            _tracer.stop_tracing()


def test_linalg_eager() -> None:
    """Docstring."""
    with ConfigContext(eager_mode=True):
        t2 = _make_tensor([[2.0, 0.0], [0.0, 2.0]], DType.Float32, (2, 2))
        t1 = _make_tensor([1.0, 2.0], DType.Float32, (2,))
        assert t1 is not None

        for op_name in LINALG:
            op_fn = globals()[op_name]
            if op_name in ["matmul", "dot", "vdot", "inner", "outer"]:
                res = op_fn(t2, t2)
                assert isinstance(res, Tensor)
            elif op_name == "tensordot":
                res = op_fn(t2, t2)
                assert isinstance(res, Tensor)
            elif op_name == "einsum":
                res = op_fn("ii->i", t2)
                assert isinstance(res, Tensor)
            elif op_name in ["svd", "qr", "slogdet", "eigh"]:
                res = op_fn(t2)
                assert isinstance(res, tuple)
            elif op_name == "matrix_power":
                res = op_fn(t2, 2)
                assert isinstance(res, Tensor)
            else:
                res = op_fn(t2)
                assert isinstance(res, Tensor)


def test_linalg_tracing() -> None:
    """Docstring."""
    with ConfigContext(eager_mode=False):
        t2 = _make_proxy((2, 2), DType.Float32)

        with pytest.raises(RuntimeError):
            matmul(t2, t2)

        _ = _tracer.start_tracing()
        try:
            for op_name in LINALG:
                op_fn = globals()[op_name]
                if op_name in ["matmul", "dot", "vdot", "inner", "outer"]:
                    res = op_fn(t2, t2)
                    assert isinstance(res, Tensor)
                elif op_name == "tensordot":
                    res = op_fn(t2, t2)
                    assert isinstance(res, Tensor)
                elif op_name == "einsum":
                    res = op_fn("ii->i", t2)
                    assert isinstance(res, Tensor)
                elif op_name in ["svd", "qr", "slogdet", "eigh"]:
                    res = op_fn(t2)
                    assert isinstance(res, tuple)
                elif op_name == "matrix_power":
                    res = op_fn(t2, 2)
                    assert isinstance(res, Tensor)
                else:
                    res = op_fn(t2)
                    assert isinstance(res, Tensor)
        finally:
            _tracer.stop_tracing()
