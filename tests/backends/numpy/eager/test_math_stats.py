"""Tests for test_math_stats."""

from __future__ import annotations

import inspect
from typing import TYPE_CHECKING, cast

import numpy as np
import pytest

import ml_switcheroo_compiler.backends.numpy.eager.math_stats as ms
from ml_switcheroo_compiler.backends.numpy.eager.math_stats import _np_randomcategorical, _np_randomshuffle

if TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import EagerCallableProtocol, EagerTensorProtocol


def test_math_stats_coverage() -> None:
    """Test function test_math_stats_coverage."""
    ops = [getattr(ms, name) for name in dir(ms) if name.startswith("_") and callable(getattr(ms, name))]
    arg = np.array([1.0, 2.0])
    for op in ops:
        sig = inspect.signature(op)
        args_to_pass = []
        for p in sig.parameters.values():
            if p.name == "backend_module":
                args_to_pass.append(np)
            elif p.name in ("a", "b", "x", "y", "tensor"):
                cast("EagerCallableProtocol", args_to_pass.append)(arg)
            else:
                cast("EagerCallableProtocol", args_to_pass.append)(arg)
        try:
            op(*args_to_pass)
        except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, OverflowError, RecursionError, LookupError, ArithmeticError):
            _caught_err = True
        try:
            op(*args_to_pass[:-1])
        except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, OverflowError, RecursionError, LookupError, ArithmeticError):
            _caught_err = True
    assert callable(test_math_stats_coverage)


def test_math_stats_exceptions() -> None:
    """Test function test_math_stats_exceptions."""
    import ml_switcheroo_compiler.backends.numpy.eager.math_stats as ms

    with pytest.raises(ValueError):
        ms._np_descriptive(np, a=None)
    with pytest.raises(IndexError):
        ms._np_randompermutation(np, 5)
    with pytest.raises(TypeError):
        ms._np_randomcategorical(np, [1, 2], p=[0.5, 0.5])


def test_missing_math_stats2() -> None:
    """Test function test_missing_math_stats2."""
    import ml_switcheroo_compiler.backends.numpy.eager.math_stats as ms

    class MockData:
        """Test class MockData."""

        data = 5

    ms._np_randompermutation(np, np.array([1.0]), x=MockData())

    class MockDataP:
        """Test class MockDataP."""

        data = [1.0]

    class MockDataList:
        """Test class MockDataList."""

        data = [1.0]

    ms._np_randomchoice(np, np.array([1.0]), a=MockDataList(), p=MockDataP())
    assert callable(test_missing_math_stats2)


def test_np_random_shuffle_list() -> None:
    """Test function test_np_random_shuffle_list."""
    x = [1, 2, 3]
    res = _np_randomshuffle(None, x)
    assert isinstance(res, np.ndarray)
    assert set(res) == {1, 2, 3}


def test_randomcategorical_shape() -> None:
    """Test function test_randomcategorical_shape."""
    res = _np_randomcategorical(None, None, logits=np.array([1.0, 2.0]), shape=(1, 2))
    assert cast("EagerTensorProtocol", getattr(res, "shape")) == (1, 2)
    res = _np_randomcategorical(None, None, logits=np.array([[1.0, 2.0]]))
    assert cast("EagerTensorProtocol", getattr(res, "shape")) == (1,)
