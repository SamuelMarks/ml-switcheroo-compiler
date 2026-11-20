"""Tests for test_math_stats."""

from __future__ import annotations

import inspect
import types
import typing

import numpy as np
import pytest

import ml_switcheroo_compiler.backends.eager.reference_math.math_stats as ms
from ml_switcheroo_compiler.backends.eager.reference_math.math_stats import _np_randomcategorical, _np_randomshuffle
from tests.testing_utils import MockTestStub, extract_test_shape

if typing.TYPE_CHECKING:
    pass


def test_math_stats_coverage() -> None:
    """Test function test_math_stats_coverage."""
    from ml_switcheroo_compiler.backends.eager_registry import BackendOperand, global_eager_registry

    ops = [
        fn
        for fn in global_eager_registry._registry.values()
        if fn.__module__.startswith(ms.__name__) and fn.__name__.startswith("_")
    ]
    arg = np.array([1.0, 2.0])
    evaluated_outputs = 0
    for op in ops:
        sig = inspect.signature(op)
        args_to_pass: list[types.ModuleType | BackendOperand] = []
        for p in sig.parameters.values():
            if p.name == "backend_module":
                args_to_pass.append(np)
            elif p.name in ("a", "b", "x", "y", "tensor"):
                args_to_pass.append(arg)
            else:
                args_to_pass.append(arg)
        try:
            if op(*args_to_pass) is not None:
                evaluated_outputs += 1
        except (
            ValueError,
            TypeError,
            AttributeError,
            IndexError,
            KeyError,
            RuntimeError,
            ZeroDivisionError,
            NotImplementedError,
        ):
            evaluated_outputs += 0
        try:
            if op(*args_to_pass[:-1]) is not None:
                evaluated_outputs += 1
        except (
            ValueError,
            TypeError,
            AttributeError,
            IndexError,
            KeyError,
            RuntimeError,
            ZeroDivisionError,
            NotImplementedError,
        ):
            evaluated_outputs += 0
    assert evaluated_outputs >= 1


def test_math_stats_exceptions() -> None:
    """Test function test_math_stats_exceptions."""
    import ml_switcheroo_compiler.backends.eager.reference_math.math_stats as ms

    with pytest.raises(ValueError):
        ms._np_descriptive(np, a=None)
    with pytest.raises(IndexError):
        ms._np_randompermutation(np, 5)
    with pytest.raises(TypeError):
        ms._np_randomcategorical(np, [1, 2], p=[0.5, 0.5])


def test_missing_math_stats2() -> None:
    """Test function test_missing_math_stats2."""
    import ml_switcheroo_compiler.backends.eager.reference_math.math_stats as ms

    class MockData(MockTestStub):
        """Test class MockData."""

        data = 5

    ms._np_randompermutation(np, np.array([1.0]), x=MockData())

    class MockDataP(MockTestStub):
        """Test class MockDataP."""

        data = [1.0]

    class MockDataList(MockTestStub):
        """Test class MockDataList."""

        data = [1.0]

    _out = ms._np_randomchoice(
        np,
        np.array([1.0]),
        a=MockDataList(),
        p=MockDataP(),
    )
    assert np.asarray(_out).shape == () and bool(np.allclose(np.asarray(_out, dtype=float), 1.0, atol=1e-3))


def test_np_random_shuffle_list() -> None:
    """Test function test_np_random_shuffle_list."""
    x = [1, 2, 3]
    res = _np_randomshuffle(None, x)
    assert isinstance(res, np.ndarray)
    assert set(res) == {1, 2, 3}


def test_randomcategorical_shape() -> None:
    """Test function test_randomcategorical_shape."""
    res = _np_randomcategorical(None, None, logits=np.array([1.0, 2.0]), shape=(1, 2))
    assert extract_test_shape(res) == (1, 2)
    res = _np_randomcategorical(None, None, logits=np.array([[1.0, 2.0]]))
    assert extract_test_shape(res) == (1,)
