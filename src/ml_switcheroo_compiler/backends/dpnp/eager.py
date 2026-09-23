"""Eager evaluation dispatch for Data Parallel NumPy (dpnp)."""

from __future__ import annotations

import importlib

from ml_switcheroo_compiler.core.errors import BackendNotSupportedError


def execute_op(
    cls_or_op: object,
    op_type_or_first: object = None,
    *args: object,
    **kwargs: object,
) -> object:
    """Execute operation eagerly on DPNP backend with Intel SYCL target.

    Args:
        cls_or_op (object): Caller class context or operation name string.
        op_type_or_first (object): Operation name string or first argument.
        *args (object): Input tensor parameters.
        **kwargs (object): Additional keyword attributes.

    Returns:
        object: Evaluated DPNP array.

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
        dpnp_mod = importlib.import_module("dpnp")
    except Exception:
        dpnp_mod = importlib.import_module("numpy")

    fn_name = op_type.lower()
    if hasattr(dpnp_mod, fn_name):
        return getattr(dpnp_mod, fn_name)(*actual_args, **kwargs)

    # Common aliases
    alias_map = {
        "neg": "negative",
        "sub": "subtract",
        "mul": "multiply",
        "div": "divide",
        "truedivide": "true_divide",
    }
    target = alias_map.get(fn_name)
    if target and hasattr(dpnp_mod, target):
        return getattr(dpnp_mod, target)(*actual_args, **kwargs)

    raise BackendNotSupportedError(f"Operation '{op_type}' not supported in DPNP backend.")
