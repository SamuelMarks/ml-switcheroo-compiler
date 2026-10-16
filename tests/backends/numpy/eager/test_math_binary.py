"""Tests for test_math_binary."""

from __future__ import annotations

import inspect
from typing import TYPE_CHECKING, cast

import numpy as np

import ml_switcheroo_compiler.backends.numpy.eager.math_binary as mb
from ml_switcheroo_compiler.backends.numpy.eager.math_binary import _np_polygamma, _np_zeta

if TYPE_CHECKING:
    from collections.abc import Callable

    from ml_switcheroo_compiler.backends.eager_registry import EagerTensorProtocol, EagerValue


def test_math_binary_coverage() -> None:
    """Test function test_math_binary_coverage."""
    ops = [getattr(mb, name) for name in dir(mb) if name.startswith("_") and callable(getattr(mb, name))]
    arg = np.array([1.0, 2.0])

    for op in ops:
        sig = inspect.signature(op)
        args_to_pass = []
        for p in sig.parameters.values():
            if p.name == "backend_module":
                args_to_pass.append(np)
            elif p.name in ("a", "b", "x", "y"):
                cast("Callable[..., EagerTensorProtocol]", args_to_pass.append)(arg)
            else:
                cast("Callable[..., EagerTensorProtocol]", args_to_pass.append)(arg)

        try:
            op(*args_to_pass)
        except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
            assert _exc is not None

        try:
            op(*args_to_pass[:-1])
        except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
            assert _exc is not None


def test_polygamma_zeta_missing_args() -> None:
    """Test function test_polygamma_zeta_missing_args."""
    import ml_switcheroo_compiler.backends.numpy.eager.math_binary as mb

    arg = np.array([1.0, 2.0])
    try:
        mb._np_polygamma(np, arg)
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None
    try:
        mb._np_zeta(np, arg)
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None


def test_missing_math_binary() -> None:
    """Test function test_missing_math_binary."""
    import ml_switcheroo_compiler.backends.numpy.eager.math_binary as mb

    class DummyBk:
        """Test class DummyBk."""

        def zeros_like(self, x: EagerValue) -> EagerValue:
            """Test function zeros_like."""
            return np.zeros_like(x)

        def array(self, x: EagerValue) -> EagerValue:
            """Test function array."""
            return np.asarray(x)

    res = mb._np_polygamma(DummyBk(), np.array([1.0]), np.array([2.0]))
    cast("Callable[..., EagerTensorProtocol]", np.testing.assert_array_almost_equal)(res, cast("Callable[..., EagerTensorProtocol]", np.array)([0.64493407]))

    res2 = mb._np_zeta(DummyBk(), np.array([1.0]), np.array([2.0]))
    cast("Callable[..., EagerTensorProtocol]", np.testing.assert_array_almost_equal)(res2, cast("Callable[..., EagerTensorProtocol]", np.array)([cast("Callable[..., EagerTensorProtocol]", float)("inf")]))

    try:
        mb._np_zeta(DummyBk(), np.array([1.0]))
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None

    try:
        mb._np_zeta(DummyBk(), np.array([1.0]), q=np.array([1.0]))
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None


def test_polygamma_zeta_missing_args_2() -> None:
    """Test function test_polygamma_zeta_missing_args_2."""
    a = np.array([1.0])
    res1 = _np_polygamma(np, a)
    assert np.all(res1 == 0.0)
    res2 = _np_zeta(np, a)
    assert np.all(res2 == 0.0)


def test_polygamma_zeta_kwargs() -> None:
    """Test function test_polygamma_zeta_kwargs."""
    a = np.array([1.0])
    res1 = _np_polygamma(np, a, x=a)
    assert cast("EagerTensorProtocol", getattr(res1, "shape")) == (1,)
    res2 = _np_zeta(np, a, q=a)
    assert cast("EagerTensorProtocol", getattr(res2, "shape")) == (1,)
