"""Docstring module."""

from typing import Any

# ruff: noqa: F403, F405
"""Tests for Unary and Binary operations."""

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

UNARY_OPS = [
    "abs",
    "acos",
    "acosh",
    "asin",
    "asinh",
    "atan",
    "atanh",
    "bitwise_not",
    "cbrt",
    "ceil",
    "conj",
    "cos",
    "cosh",
    "deg2rad",
    "erf",
    "erfc",
    "exp",
    "exp2",
    "expm1",
    "fix",
    "floor",
    "imag",
    "isfinite",
    "isinf",
    "isnan",
    "lgamma",
    "log",
    "log10",
    "log1p",
    "log2",
    "logical_not",
    "negative",
    "positive",
    "rad2deg",
    "real",
    "reciprocal",
    "round",
    "sign",
    "sin",
    "sinc",
    "sinh",
    "sqrt",
    "square",
    "tan",
    "tanh",
    "trunc",
]

BINARY_OPS = [
    "add",
    "bitwise_and",
    "bitwise_or",
    "bitwise_xor",
    "copysign",
    "divide",
    "equal",
    "float_power",
    "floor_divide",
    "fmax",
    "fmin",
    "fmod",
    "gcd",
    "greater",
    "greater_equal",
    "heaviside",
    "hypot",
    "lcm",
    "ldexp",
    "left_shift",
    "less",
    "less_equal",
    "logaddexp",
    "logaddexp2",
    "logical_and",
    "logical_or",
    "logical_xor",
    "maximum",
    "minimum",
    "mod",
    "multiply",
    "nextafter",
    "not_equal",
    "power",
    "remainder",
    "right_shift",
    "subtract",
]

UNIMPLEMENTED_UNARY = ["erfc", "lgamma", "digamma"]


def _make_tensor(data_list: Any, dtype: Any) -> Any:
    return Tensor(
        np.array(data_list, dtype=dtype.value),
        (len(data_list),),
        dtype,
        Device(DeviceType.CPU),
    )


def _make_proxy(shape: Any, dtype: Any) -> Any:
    return Tensor(
        ProxyTensor("a", shape, dtype.value), shape, dtype, Device(DeviceType.CPU)
    )


def test_unary_eager() -> None:
    """Docstring."""
    with ConfigContext(eager_mode=True):
        t_float = _make_tensor([0.5, 0.2], DType.Float32)
        t_int = _make_tensor([1, 2], DType.Int32)
        t_bool = _make_tensor([True, False], DType.Bool)

        for op_name in UNARY_OPS:
            op_fn = globals()[op_name]
            if op_name in UNIMPLEMENTED_UNARY:
                with pytest.raises(UnimplementedMathError):
                    op_fn(t_float)
            else:
                t = (
                    t_bool
                    if "logical" in op_name or "bitwise" in op_name
                    else (t_int if "bitwise" in op_name else t_float)
                )
                res = op_fn(t)
                assert isinstance(res, Tensor)

        # Test specials
        with pytest.raises(UnimplementedMathError):
            digamma(t_float)

        res = rsqrt(t_float)
        assert res.dtype == DType.Float32

        res = atan2(t_float, t_float)
        assert res.dtype == DType.Float32

        m, e = frexp(t_float)
        assert m.dtype == DType.Float32
        assert e.dtype == DType.Int32

        res = cast(t_int, DType.Float32)
        assert res.dtype == DType.Float32

        res = bitcast(t_int, DType.Float32)
        assert res.dtype == DType.Float32


def test_binary_eager() -> None:
    """Docstring."""
    with ConfigContext(eager_mode=True):
        t_float = _make_tensor([0.5, 0.2], DType.Float32)
        t_int = _make_tensor([1, 2], DType.Int32)
        t_bool = _make_tensor([True, False], DType.Bool)

        for op_name in BINARY_OPS:
            op_fn = globals()[op_name]
            t = (
                t_bool
                if "logical" in op_name or "bitwise" in op_name
                else (t_int if "bitwise" in op_name or "shift" in op_name else t_float)
            )
            # numpy logaddexp/etc need float
            if op_name in ["logaddexp", "logaddexp2", "nextafter", "copysign", "ldexp"]:
                t = t_float
                t_other = t_float if op_name != "ldexp" else t_int
            elif op_name in ["gcd", "lcm"]:
                t = t_int
                t_other = t_int
            else:
                t_other = t
            res = op_fn(t, t_other)
            assert isinstance(res, Tensor)

        res = isclose(t_float, t_float)
        assert res.dtype == DType.Bool

        q, r = divmod(t_float, t_float)
        assert q.dtype == DType.Float32
        assert r.dtype == DType.Float32

        res = allclose(t_float, t_float)
        assert res is True


def test_unary_tracing() -> None:
    """Docstring."""
    with ConfigContext(eager_mode=False):
        t_float = _make_proxy((2,), DType.Float32)

        # Ensure RuntimeError outside tracing
        with pytest.raises(RuntimeError):
            abs(t_float)
        with pytest.raises(RuntimeError):
            atan2(t_float, t_float)
        with pytest.raises(RuntimeError):
            frexp(t_float)
        with pytest.raises(RuntimeError):
            cast(t_float, DType.Int32)
        with pytest.raises(RuntimeError):
            bitcast(t_float, DType.Int32)

        _ = _tracer.start_tracing()
        try:
            for op_name in UNARY_OPS:
                op_fn = globals()[op_name]
                res = op_fn(t_float)
                assert isinstance(res, Tensor)

            digamma(t_float)
            rsqrt(t_float)
            atan2(t_float, t_float)
            m, e = frexp(t_float)
            cast(t_float, DType.Int32)
            bitcast(t_float, DType.Int32)
        finally:
            _tracer.stop_tracing()


def test_binary_tracing() -> None:
    """Docstring."""
    with ConfigContext(eager_mode=False):
        t_float = _make_proxy((2,), DType.Float32)

        with pytest.raises(RuntimeError):
            add(t_float, t_float)
        with pytest.raises(RuntimeError):
            divmod(t_float, t_float)
        with pytest.raises(RuntimeError):
            allclose(t_float, t_float)

        _ = _tracer.start_tracing()
        try:
            for op_name in BINARY_OPS:
                op_fn = globals()[op_name]
                res = op_fn(t_float, t_float)
                assert isinstance(res, Tensor)

            isclose(t_float, t_float)
            divmod(t_float, t_float)
            allclose(t_float, t_float)
        finally:
            _tracer.stop_tracing()
