"""Tests for test_math_binary."""

from __future__ import annotations

import inspect
import types

import numpy as np

import ml_switcheroo_compiler.backends.eager.reference_math.math_binary as mb
from ml_switcheroo_compiler.backends.numpy.eager.scipy_special_kernels import _np_polygamma, _np_zeta
from tests.testing_utils import BackendArg, BackendOperand, MockTestStub


def test_math_binary_coverage() -> None:
    """Test function test_math_binary_coverage."""
    from ml_switcheroo_compiler.backends.eager_registry import global_eager_registry

    ops = [
        fn
        for fn in global_eager_registry._registry.values()
        if fn.__module__.startswith(mb.__name__) and fn.__name__.startswith("_")
    ]
    arg = np.array([1.0, 2.0])
    evaluated_outputs = 0
    for op in ops:
        sig = inspect.signature(op)
        args_to_pass: list[types.ModuleType | BackendOperand] = []
        for p in sig.parameters.values():
            if p.name == "backend_module":
                args_to_pass.append(np)
            elif p.name in ("a", "b", "x", "y"):
                args_to_pass.append(arg)
            else:
                args_to_pass.append(arg)
        try:
            out = op(*args_to_pass)
            if out is not None:
                evaluated_outputs += 1
        except (ValueError, TypeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError):
            evaluated_outputs += 0
        try:
            out2 = op(*args_to_pass[:-1])
            if out2 is not None:
                evaluated_outputs += 1
        except (ValueError, TypeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError):
            evaluated_outputs += 0
    assert evaluated_outputs >= 5


def test_polygamma_zeta_missing_args() -> None:
    """Test function test_polygamma_zeta_missing_args."""
    import ml_switcheroo_compiler.backends.numpy.eager.scipy_special_kernels as mb

    arg = np.array([1.0, 2.0])
    mb._np_polygamma(np, arg)
    _out = mb._np_zeta(np, arg)
    assert np.asarray(_out).shape == (2,) and bool(np.allclose(np.asarray(_out, dtype=float), [0.0, 0.0], atol=1e-3))


def test_missing_math_binary() -> None:
    """Test function test_missing_math_binary."""
    import ml_switcheroo_compiler.backends.numpy.eager.scipy_special_kernels as mb

    class DummyBk(MockTestStub):
        """Test class DummyBk."""

        def zeros_like(self, x: BackendArg) -> BackendArg:
            """Test function zeros_like.

            Args:
                x: Test parameter `x`.

            Returns:
                BackendArg: Computed output value from `zeros_like`.
            """
            return np.zeros_like(x)

        def array(self, x: BackendArg) -> BackendArg:
            """Test function array.

            Args:
                x: Test parameter `x`.

            Returns:
                BackendArg: Computed output value from `array`.
            """
            return np.asarray(x)

    res = mb._np_polygamma(DummyBk(), np.array([1.0]), np.array([2.0]))
    np.testing.assert_array_almost_equal(res, np.array([0.64493407]))
    res2 = mb._np_zeta(DummyBk(), np.array([1.0]), np.array([2.0]))
    np.testing.assert_array_almost_equal(res2, np.array([float("inf")]))
    mb._np_zeta(DummyBk(), np.array([1.0]))
    mb._np_zeta(DummyBk(), np.array([1.0]), q=np.array([1.0]))


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
    assert res1.shape == (1,)
    res2 = _np_zeta(np, a, q=a)
    assert res2.shape == (1,)
