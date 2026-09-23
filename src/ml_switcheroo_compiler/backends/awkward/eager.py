"""Eager evaluation dispatch for Awkward Array backend."""

from __future__ import annotations

import importlib

from ml_switcheroo_compiler.core.errors import BackendNotSupportedError


def execute_op(
    cls_or_op: object,
    op_type_or_first: object = None,
    *args: object,
    **kwargs: object,
) -> object:
    """Execute operation eagerly on Awkward Array backend supporting ragged structures.

    Args:
        cls_or_op (object): Caller class context or operation name string.
        op_type_or_first (object): Operation name string or first argument.
        *args (object): Input tensor parameters.
        **kwargs (object): Additional keyword attributes.

    Returns:
        object: Evaluated Awkward array or NumPy array.

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
        ak_mod = importlib.import_module("awkward")
    except Exception:
        ak_mod = None

    fn_name = op_type.lower()
    if ak_mod is not None and hasattr(ak_mod, fn_name):
        return getattr(ak_mod, fn_name)(*actual_args, **kwargs)

    # Fallback to NumPy evaluation for standard tensor ops
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
    target = alias_map.get(fn_name)
    if target and hasattr(numpy_mod, target):
        return getattr(numpy_mod, target)(*actual_args, **kwargs)

    raise BackendNotSupportedError(f"Operation '{op_type}' not supported in Awkward backend.")
