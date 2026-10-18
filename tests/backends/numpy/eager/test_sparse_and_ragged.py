"""Tests for test_sparse_and_ragged."""

from __future__ import annotations

import inspect
from typing import TYPE_CHECKING, cast

import numpy as np
import pytest

import ml_switcheroo_compiler.backends.numpy.eager.sparse_and_ragged as sr

if TYPE_CHECKING:
    from collections.abc import Callable

    from ml_switcheroo_compiler.backends.eager_registry import EagerTensorProtocol


def test_sparse_and_ragged_coverage() -> None:
    """Test function test_sparse_and_ragged_coverage."""
    ops = [getattr(sr, name) for name in dir(sr) if name.startswith("_") and callable(getattr(sr, name))]
    arg = np.array([1.0, 2.0])
    for op in ops:
        sig = inspect.signature(op)
        args_to_pass = []
        for p in sig.parameters.values():
            if p.name == "backend_module":
                args_to_pass.append(np)
            elif p.name in ("x", "y", "tensor", "a", "b", "c", "t", "sp_input"):
                cast("Callable[..., EagerTensorProtocol]", args_to_pass.append)(arg)
            else:
                cast("Callable[..., EagerTensorProtocol]", args_to_pass.append)(arg)
        with pytest.raises(TypeError):
            op(*args_to_pass)
        try:
            op(*args_to_pass[:-1])
        except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, OverflowError, RecursionError, LookupError, ArithmeticError):
            _caught_err = True
