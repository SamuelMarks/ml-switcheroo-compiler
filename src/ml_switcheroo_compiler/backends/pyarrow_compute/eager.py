"""Eager evaluation dispatch for Apache Arrow Compute backend."""

from __future__ import annotations

import importlib

from ml_switcheroo_compiler.core.errors import BackendNotSupportedError


def execute_op(
    cls_or_op: object,
    op_type_or_first: object = None,
    *args: object,
    **kwargs: object,
) -> object:
    """Execute operation eagerly on Apache Arrow Compute backend.

    Args:
        cls_or_op (object): Caller class context or operation name string.
        op_type_or_first (object): Operation name string or first argument.
        *args (object): Input tensor parameters.
        **kwargs (object): Additional keyword attributes.

    Returns:
        object: Evaluated Arrow array or scalar.

    Raises:
        BackendNotSupportedError: When the operation cannot be resolved.
    """
    if isinstance(cls_or_op, type):
        op_type = str(op_type_or_first)
        actual_args = args
    else:
        op_type = str(cls_or_op)
        actual_args = (op_type_or_first,) + args if op_type_or_first is not None else args

    try:
        pc_mod = importlib.import_module("pyarrow.compute")
    except Exception:
        pc_mod = None

    fn_name = op_type.lower()
    if pc_mod is not None and hasattr(pc_mod, fn_name):
        return getattr(pc_mod, fn_name)(*actual_args, **kwargs)

    # Common Arrow Compute name mappings
    pc_map = {
        "add": "add",
        "sub": "subtract",
        "mul": "multiply",
        "div": "divide",
        "sum": "sum",
        "mean": "mean",
        "min": "min",
        "max": "max",
        "equal": "equal",
        "greater": "greater",
        "less": "less",
    }
    target = pc_map.get(fn_name)
    if pc_mod is not None and target and hasattr(pc_mod, target):
        return getattr(pc_mod, target)(*actual_args, **kwargs)

    # Fallback to NumPy
    numpy_mod = importlib.import_module("numpy")
    if hasattr(numpy_mod, fn_name):
        return getattr(numpy_mod, fn_name)(*actual_args, **kwargs)

    alias_map = {
        "neg": "negative",
        "sub": "subtract",
        "mul": "multiply",
        "div": "divide",
        "truedivide": "true_divide",
    }
    target_np = alias_map.get(fn_name)
    if target_np and hasattr(numpy_mod, target_np):
        return getattr(numpy_mod, target_np)(*actual_args, **kwargs)

    raise BackendNotSupportedError(f"Operation '{op_type}' not supported in PyArrow Compute backend.")
