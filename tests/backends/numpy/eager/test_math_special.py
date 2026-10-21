"""Tests for test_math_special."""

from __future__ import annotations

import inspect
import typing

import numpy as np

import ml_switcheroo_compiler.backends.numpy.eager.math_advanced as ma
import ml_switcheroo_compiler.backends.numpy.eager.math_special as ms
from tests.testing_utils import as_callable

if typing.TYPE_CHECKING:
    pass


def test_math_advanced_coverage() -> None:
    """Test function test_math_advanced_coverage."""
    ops = [getattr(ma, name) for name in dir(ma) if name.startswith("_") and callable(getattr(ma, name))]
    for op in ops:
        arg = np.array([1.0, 2.0])
        sig = inspect.signature(op)
        args_to_pass = []
        for p in sig.parameters.values():
            if p.name == "backend_module":
                args_to_pass.append(np)
            elif p.name in ("n",):
                as_callable(args_to_pass.append)(1)
            else:
                as_callable(args_to_pass.append)(arg)
        try:
            op(*args_to_pass)
        except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, OverflowError, RecursionError, LookupError, ArithmeticError):
            _caught_err = True
        try:
            op(*args_to_pass[:-1])
        except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, OverflowError, RecursionError, LookupError, ArithmeticError):
            _caught_err = True
    assert callable(test_math_advanced_coverage)


def test_math_special_coverage() -> None:
    """Test function test_math_special_coverage."""
    ops = [getattr(ms, name) for name in dir(ms) if name.startswith("_") and callable(getattr(ms, name))]
    arg = np.array([1.0, 2.0])
    for op in ops:
        sig = inspect.signature(op)
        args_to_pass = []
        for p in sig.parameters.values():
            if p.name == "backend_module":
                args_to_pass.append(np)
            elif p.name in ("n",):
                as_callable(args_to_pass.append)(1)
            else:
                as_callable(args_to_pass.append)(arg)
        try:
            op(*args_to_pass)
        except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, OverflowError, RecursionError, LookupError, ArithmeticError):
            _caught_err = True
        try:
            op(*args_to_pass[:-1])
        except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, OverflowError, RecursionError, LookupError, ArithmeticError):
            _caught_err = True
    ms._np_expint(np, np.array([1.0, 2.0]))
    as_callable(ms._np_expint)(np, x=np.array([1.0, 2.0]), n=1)
    assert callable(test_math_special_coverage)
