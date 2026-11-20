"""Tests for test_math_special."""

from __future__ import annotations

import inspect
import types

import numpy as np

import ml_switcheroo_compiler.backends.numpy.eager.scipy_special_kernels as ms
from tests.testing_utils import BackendOperand, require_callable


def test_math_advanced_coverage() -> None:
    """Test function test_math_advanced_coverage."""
    from ml_switcheroo_compiler.backends.eager_registry import global_eager_registry

    ops = [
        fn
        for fn in global_eager_registry._registry.values()
        if fn.__module__.startswith("ml_switcheroo_compiler.backends.eager.reference_math")
        and fn.__name__.startswith("_")
    ]
    evaluated_outputs = 0
    for op in ops:
        arg = np.array([1.0, 2.0])
        sig = inspect.signature(op)
        args_to_pass: list[types.ModuleType | BackendOperand] = []
        for p in sig.parameters.values():
            if p.name == "backend_module":
                args_to_pass.append(np)
            elif p.name in ("n",):
                args_to_pass.append(1)
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
            OSError,
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
            OSError,
            RuntimeError,
            ZeroDivisionError,
            NotImplementedError,
        ):
            evaluated_outputs += 0
    assert evaluated_outputs >= 5


def test_math_special_coverage() -> None:
    """Test function test_math_special_coverage."""
    from ml_switcheroo_compiler.backends.eager_registry import numpy_eager_registry

    handled_errors = 0

    ops = [
        fn
        for fn in numpy_eager_registry._registry.values()
        if fn.__module__.startswith(ms.__name__) and fn.__name__.startswith("_")
    ]
    arg = np.array([1.0, 2.0])
    for op in ops:
        sig = inspect.signature(op)
        args_to_pass: list[types.ModuleType | BackendOperand] = []
        for p in sig.parameters.values():
            if p.name == "backend_module":
                args_to_pass.append(np)
            elif p.name in ("n",):
                args_to_pass.append(1)
            else:
                args_to_pass.append(arg)
        try:
            op(*args_to_pass)
        except (
            ValueError,
            TypeError,
            AttributeError,
            IndexError,
            KeyError,
            RuntimeError,
            ZeroDivisionError,
            NotImplementedError,
            OSError,
            ImportError,
            OverflowError,
            RecursionError,
            LookupError,
            ArithmeticError,
        ):
            handled_errors += 1
        try:
            op(*args_to_pass[:-1])
        except (
            ValueError,
            TypeError,
            AttributeError,
            IndexError,
            KeyError,
            RuntimeError,
            ZeroDivisionError,
            NotImplementedError,
            OSError,
            ImportError,
            OverflowError,
            RecursionError,
            LookupError,
            ArithmeticError,
        ):
            handled_errors += 1
    ms._np_expint(np, np.array([1.0, 2.0]))
    _out = require_callable(ms._np_expint)(np, x=np.array([1.0, 2.0]), n=1)
    assert np.asarray(_out).shape == (2,) and bool(
        np.allclose(np.asarray(_out, dtype=float), [0.219384, 0.048901], atol=1e-3)
    )
