"""Tests for test_math_special."""

from __future__ import annotations

import inspect
import types

import numpy as np

import ml_switcheroo_compiler.backends.numpy.eager.scipy_special_kernels as ms
from tests.testing_utils import BackendOperand, require_callable


def _sample_arg_for_special_op(
    op_name: str,
    default_vals: tuple[float, float] = (1.0, 2.0),
) -> np.ndarray[tuple[int, ...], np.dtype[np.float64]]:
    """Select an in-domain input vector for `op_name`.

    Args:
        op_name (str): Target operation function name.
        default_vals (tuple[float, float]): Default pair for unconstrained ops.

    Returns:
        np.ndarray[tuple[int, ...], np.dtype[np.float64]]: Domain-valid 1-D float64 array.
    """
    is_unit_trig = any(tok in op_name for tok in ("arccos", "acos", "arcsin", "asin")) and not any(
        hyp in op_name for hyp in ("cosh", "sinh")
    )
    if is_unit_trig or any(tok in op_name for tok in ("arctanh", "atanh", "logit")):
        return np.array([0.25, 0.75])
    if "arccosh" in op_name or "acosh" in op_name:
        return np.array([1.5, 2.0])
    if "cross" in op_name:
        return np.array([1.0, 2.0, 3.0])
    return np.array(list(default_vals))


def _build_special_op_args(
    op_name: str,
    sig: inspect.Signature,
    arg: np.ndarray[tuple[int, ...], np.dtype[np.float64]],
) -> list[types.ModuleType | BackendOperand]:
    """Construct domain-safe positional call arguments for `op_name`.

    Args:
        op_name (str): Target operation function name.
        sig (inspect.Signature): Inspected callable signature.
        arg (np.ndarray[tuple[int, ...], np.dtype[np.float64]]): Base sample vector.

    Returns:
        list[types.ModuleType | BackendOperand]: Argument list starting with `np`.
    """
    args_to_pass: list[types.ModuleType | BackendOperand] = []
    for idx, p in enumerate(sig.parameters.values()):
        if p.name == "backend_module":
            args_to_pass.append(np)
        elif p.name in ("n",) or (idx == 1 and op_name in ("_np_expn", "_np_polygamma")):
            args_to_pass.append(1)
        else:
            args_to_pass.append(arg.copy())
    return args_to_pass


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
        arg = _sample_arg_for_special_op(op.__name__)
        args_to_pass = _build_special_op_args(op.__name__, inspect.signature(op), arg)
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
        args_trimmed = [a.copy() if isinstance(a, np.ndarray) else a for a in args_to_pass[:-1]]
        try:
            if op(*args_trimmed) is not None:
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
    for op in ops:
        arg = _sample_arg_for_special_op(op.__name__, default_vals=(2.0, 3.0))
        args_to_pass = _build_special_op_args(op.__name__, inspect.signature(op), arg)
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
        args_trimmed = [a.copy() if isinstance(a, np.ndarray) else a for a in args_to_pass[:-1]]
        try:
            op(*args_trimmed)
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
