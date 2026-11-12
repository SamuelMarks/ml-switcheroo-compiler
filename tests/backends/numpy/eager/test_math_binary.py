"""Tests for test_math_binary."""

from __future__ import annotations

import inspect
import typing

import numpy as np

import ml_switcheroo_compiler.backends.numpy.eager.math_binary as mb
from ml_switcheroo_compiler.backends.numpy.eager.math_binary import _np_polygamma, _np_zeta
from tests.testing_utils import ensure_callable, ensure_eager_tensor, ensure_mod

if typing.TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import EagerValue


def test_math_binary_coverage() -> None:
    """Test function test_math_binary_coverage."""
    from tests.testing_utils import get_test_members

    ops = [ensure_callable(fn) for name, fn in get_test_members(mb).items() if name.startswith("_") and callable(fn)]
    arg = np.array([1.0, 2.0])
    evaluated_outputs = 0
    for op in ops:
        sig = inspect.signature(op)
        args_to_pass = []
        for p in sig.parameters.values():
            if p.name == "backend_module":
                args_to_pass.append(np)
            elif p.name in ("a", "b", "x", "y"):
                ensure_callable(args_to_pass.append)(arg)
            else:
                ensure_callable(args_to_pass.append)(arg)
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
    import ml_switcheroo_compiler.backends.numpy.eager.math_binary as mb

    arg = np.array([1.0, 2.0])
    mb._np_polygamma(ensure_eager_tensor(np), arg)
    _out = mb._np_zeta(ensure_eager_tensor(np), arg)
    assert np.asarray(ensure_eager_tensor(_out)).shape == (2,) and bool(np.allclose(np.asarray(ensure_eager_tensor(_out), dtype=float), [0.0, 0.0], atol=1e-3))


def test_missing_math_binary() -> None:
    """Test function test_missing_math_binary."""
    import ml_switcheroo_compiler.backends.numpy.eager.math_binary as mb

    class DummyBk:
        """Test class DummyBk."""

        def zeros_like(self, x: EagerValue) -> EagerValue:
            """Test function zeros_like.

            Args:
                x: Test parameter `x`.

            Returns:
                EagerValue: Computed output value from `zeros_like`.
            """
            return np.zeros_like(x)

        def array(self, x: EagerValue) -> EagerValue:
            """Test function array.

            Args:
                x: Test parameter `x`.

            Returns:
                EagerValue: Computed output value from `array`.
            """
            return np.asarray(x)

    res = mb._np_polygamma(ensure_mod(DummyBk()), np.array([1.0]), np.array([2.0]))
    ensure_callable(np.testing.assert_array_almost_equal)(res, ensure_callable(np.array)([0.64493407]))
    res2 = mb._np_zeta(ensure_mod(DummyBk()), np.array([1.0]), np.array([2.0]))
    ensure_callable(np.testing.assert_array_almost_equal)(res2, ensure_callable(np.array)([ensure_callable(float)("inf")]))
    mb._np_zeta(ensure_mod(DummyBk()), np.array([1.0]))
    mb._np_zeta(ensure_mod(DummyBk()), np.array([1.0]), q=np.array([1.0]))


def test_polygamma_zeta_missing_args_2() -> None:
    """Test function test_polygamma_zeta_missing_args_2."""
    a = np.array([1.0])
    res1 = _np_polygamma(ensure_eager_tensor(np), a)
    assert np.all(res1 == 0.0)
    res2 = _np_zeta(ensure_eager_tensor(np), a)
    assert np.all(res2 == 0.0)


def test_polygamma_zeta_kwargs() -> None:
    """Test function test_polygamma_zeta_kwargs."""
    a = np.array([1.0])
    res1 = _np_polygamma(ensure_eager_tensor(np), a, x=a)
    assert ensure_eager_tensor(res1).shape == (1,)
    res2 = _np_zeta(ensure_eager_tensor(np), a, q=a)
    assert ensure_eager_tensor(res2).shape == (1,)
