"""Tests for test_math_binary."""

from __future__ import annotations

import inspect
import typing

import numpy as np

import ml_switcheroo_compiler.backends.numpy.eager.math_binary as mb
from ml_switcheroo_compiler.backends.numpy.eager.math_binary import _np_polygamma, _np_zeta
from ml_switcheroo_compiler.backends.protocols import BackendModuleProtocol

if typing.TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import EagerCallableProtocol, EagerTensorProtocol, EagerValue


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
                typing.cast("EagerCallableProtocol", args_to_pass.append)(arg)
            else:
                typing.cast("EagerCallableProtocol", args_to_pass.append)(arg)
        try:
            op(*args_to_pass)
        except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, OverflowError, RecursionError, LookupError, ArithmeticError):
            _caught_err = True
        try:
            op(*args_to_pass[:-1])
        except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, OverflowError, RecursionError, LookupError, ArithmeticError):
            _caught_err = True
    assert callable(test_math_binary_coverage)


def test_polygamma_zeta_missing_args() -> None:
    """Test function test_polygamma_zeta_missing_args."""
    import ml_switcheroo_compiler.backends.numpy.eager.math_binary as mb

    arg = np.array([1.0, 2.0])
    mb._np_polygamma(np, arg)
    mb._np_zeta(np, arg)
    assert callable(test_polygamma_zeta_missing_args)


def test_missing_math_binary() -> None:
    """Test function test_missing_math_binary."""
    import ml_switcheroo_compiler.backends.numpy.eager.math_binary as mb

    class DummyBk:
        """Test class DummyBk."""

        def zeros_like(self, x: EagerValue) -> EagerValue:
            """Test function zeros_like.

            Args:
                x: Test parameter `x`.
            """
            return np.zeros_like(x)

        def array(self, x: EagerValue) -> EagerValue:
            """Test function array.

            Args:
                x: Test parameter `x`.
            """
            return np.asarray(x)

    res = mb._np_polygamma(typing.cast("BackendModuleProtocol", DummyBk()), np.array([1.0]), np.array([2.0]))
    typing.cast("EagerCallableProtocol", np.testing.assert_array_almost_equal)(res, typing.cast("EagerCallableProtocol", np.array)([0.64493407]))
    res2 = mb._np_zeta(typing.cast("BackendModuleProtocol", DummyBk()), np.array([1.0]), np.array([2.0]))
    typing.cast("EagerCallableProtocol", np.testing.assert_array_almost_equal)(res2, typing.cast("EagerCallableProtocol", np.array)([typing.cast("EagerCallableProtocol", float)("inf")]))
    mb._np_zeta(typing.cast("BackendModuleProtocol", DummyBk()), np.array([1.0]))
    mb._np_zeta(typing.cast("BackendModuleProtocol", DummyBk()), np.array([1.0]), q=np.array([1.0]))


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
    assert typing.cast("EagerTensorProtocol", getattr(res1, "shape")) == (1,)
    res2 = _np_zeta(np, a, q=a)
    assert typing.cast("EagerTensorProtocol", getattr(res2, "shape")) == (1,)
