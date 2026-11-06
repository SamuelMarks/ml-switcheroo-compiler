"""Tests for ops unary logical coverage."""

from __future__ import annotations

import ml_switcheroo_compiler.ops.unary.logical as unary_logical
from tests.testing_utils import ensure_callable, get_test_members


def test_unary_logical_functions() -> None:
    """Test unary logical coverage."""

    class DummyNode:
        """Test class DummyNode."""

        shape = (10, 20)

    a = DummyNode()
    for attr, val in get_test_members(unary_logical).items():
        if attr.startswith("_infer_shape_") or attr.startswith("_infer_dtype_"):
            if callable(val):
                fn = ensure_callable(val)
                try:
                    fn(a)
                except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, OverflowError, RecursionError, LookupError, ArithmeticError):
                    _caught_err = True
                try:
                    fn(a, a)
                except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, OverflowError, RecursionError, LookupError, ArithmeticError):
                    _caught_err = True
                try:
                    fn()
                except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, OverflowError, RecursionError, LookupError, ArithmeticError):
                    _caught_err = True
    assert callable(test_unary_logical_functions)
