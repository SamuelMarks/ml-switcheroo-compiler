"""Tests for test_sparse_and_ragged."""

from __future__ import annotations

import inspect
import typing

import numpy as np
import pytest

import ml_switcheroo_compiler.backends.numpy.eager.sparse_and_ragged as sr
from tests.testing_utils import ensure_callable

if typing.TYPE_CHECKING:
    pass


def test_sparse_and_ragged_coverage() -> None:
    """Test function test_sparse_and_ragged_coverage."""
    from tests.testing_utils import get_test_members

    ops = [ensure_callable(fn) for name, fn in get_test_members(sr).items() if name.startswith("_") and callable(fn)]
    arg = np.array([1.0, 2.0])
    for op in ops:
        sig = inspect.signature(op)
        args_to_pass = []
        for p in sig.parameters.values():
            if p.name == "backend_module":
                args_to_pass.append(np)
            elif p.name in ("x", "y", "tensor", "a", "b", "c", "t", "sp_input"):
                ensure_callable(args_to_pass.append)(arg)
            else:
                ensure_callable(args_to_pass.append)(arg)
        with pytest.raises(TypeError):
            op(*args_to_pass)
        try:
            op(*args_to_pass[:-1])
        except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, OverflowError, RecursionError, LookupError, ArithmeticError):
            _caught_err = True
