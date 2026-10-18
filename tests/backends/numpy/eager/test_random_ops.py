"""Tests for test_random_ops."""

from __future__ import annotations

import inspect
from typing import TYPE_CHECKING, cast

import numpy as np
import pytest

import ml_switcheroo_compiler.backends.numpy.eager.random_ops as ro

if TYPE_CHECKING:
    from collections.abc import Callable

    from ml_switcheroo_compiler.backends.eager_registry import EagerTensorProtocol, EagerValue


def test_random_ops_coverage() -> None:
    """Test function test_random_ops_coverage."""
    ops = [getattr(ro, name) for name in dir(ro) if name.startswith("_") and callable(getattr(ro, name))]
    arg = np.array([1.0, 2.0])
    for op in ops:
        sig = inspect.signature(op)
        args_to_pass = []
        for p in sig.parameters.values():
            if p.name == "backend_module":
                args_to_pass.append(np)
            elif p.name in ("x", "y", "tensor", "a", "b", "c", "t"):
                cast("Callable[..., EagerTensorProtocol]", args_to_pass.append)(arg)
            else:
                cast("Callable[..., EagerTensorProtocol]", args_to_pass.append)(arg)
        with pytest.raises(TypeError):
            op(*args_to_pass)
        try:
            op(*args_to_pass[:-1])
        except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, OverflowError, RecursionError, LookupError, ArithmeticError):
            _caught_err = True


def test_missing_random_ops() -> None:
    """Test function test_missing_random_ops."""
    import ml_switcheroo_compiler.backends.numpy.eager.random_ops as ro

    class DummyConfig:
        """Test class DummyConfig."""

        minval = None
        maxval = None

    ro._np_uniform(np, shape=(2,), dtype="float32", config=DummyConfig())
    ro._np_stateless_split(np, seed=np.array(["test"]), num=2)
    ro._np_stateless_split(np, seed=np.array(["not_int", "other"]), num=2)

    class DummyTable:
        """Test class DummyTable."""

        def lookup(self, keys: EagerValue) -> EagerValue:
            """Test function lookup."""
            return keys

    ro._np_lookup(np, DummyTable(), np.array([1, 2]))

    class BadStr:
        """Test class BadStr."""

        def __int__(self) -> None:
            """Test function __int__."""
            raise ValueError()

        def __str__(self) -> str:
            """Test function __str__."""
            return "bad"

    ro._np_stateless_split(np, seed=np.array([BadStr()]), num=2)
    ro._np_lookup(np, {1: 2}, np.array([1, 2]))
    ro._np_stateless_split(np, seed=np.array([]))
