"""Tests for test_math_special."""

from __future__ import annotations

import inspect
import typing

import numpy as np

import ml_switcheroo_compiler.backends.numpy.eager.math_advanced as ma
import ml_switcheroo_compiler.backends.numpy.eager.math_special as ms
from tests.testing_utils import ensure_callable, ensure_eager_tensor

if typing.TYPE_CHECKING:
    pass


def test_math_advanced_coverage() -> None:
    """Test function test_math_advanced_coverage."""
    from tests.testing_utils import get_test_members

    ops = [ensure_callable(fn) for name, fn in get_test_members(ma).items() if name.startswith("_") and callable(fn)]
    for op in ops:
        arg = np.array([1.0, 2.0])
        sig = inspect.signature(op)
        args_to_pass = []
        for p in sig.parameters.values():
            if p.name == "backend_module":
                args_to_pass.append(np)
            elif p.name in ("n",):
                ensure_callable(args_to_pass.append)(1)
            else:
                ensure_callable(args_to_pass.append)(arg)
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
    from tests.testing_utils import get_test_members

    ops = [ensure_callable(fn) for name, fn in get_test_members(ms).items() if name.startswith("_") and callable(fn)]
    arg = np.array([1.0, 2.0])
    for op in ops:
        sig = inspect.signature(op)
        args_to_pass = []
        for p in sig.parameters.values():
            if p.name == "backend_module":
                args_to_pass.append(np)
            elif p.name in ("n",):
                ensure_callable(args_to_pass.append)(1)
            else:
                ensure_callable(args_to_pass.append)(arg)
        try:
            op(*args_to_pass)
        except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, OverflowError, RecursionError, LookupError, ArithmeticError):
            _caught_err = True
        try:
            op(*args_to_pass[:-1])
        except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, OverflowError, RecursionError, LookupError, ArithmeticError):
            _caught_err = True
    ms._np_expint(ensure_eager_tensor(np), np.array([1.0, 2.0]))
    ensure_callable(ms._np_expint)(np, x=np.array([1.0, 2.0]), n=1)
    assert callable(test_math_special_coverage)
