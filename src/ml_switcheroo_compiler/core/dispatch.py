"""Dispatch utilities for the ml-switcheroo compiler."""

from __future__ import annotations

import typing
from collections.abc import Sequence

from ml_switcheroo_compiler.backends.registry import get_active_backend
from ml_switcheroo_compiler.core.config import config as core_config
from ml_switcheroo_compiler.core.shape import AttrValue, TensorOrScalar


def dispatch(module_name: str, func_name: str, *args: TensorOrScalar, **kwargs: AttrValue) -> TensorOrScalar:
    """Dynamically dispatch a function to the active backend.

    Args:
        module_name (str): The name of the backend submodule (e.g., 'lax').
        func_name (str): The name of the function to execute (e.g., 'abs_p').
        *args (TensorOrScalar): Positional arguments for the function.
        **kwargs (AttrValue): Keyword arguments for the function.

    Returns:
        TensorOrScalar: The result of the function execution.

    Raises:
        ValueError: If not supported in the active backend or in tracing mode.
    """
    if core_config.eager_mode:
        backend = get_active_backend()
        if hasattr(backend.module, module_name):
            submodule = getattr(backend.module, module_name)
            if hasattr(submodule, func_name):
                return typing.cast(TensorOrScalar, getattr(submodule, func_name)(*args, **kwargs))
        raise ValueError(f"{func_name} is not supported in the active backend.")
    from ml_switcheroo_compiler.ops.registry import get_op
    from ml_switcheroo_compiler.ops.shape.utils import _emit_shape_node

    try:
        op_def = get_op(func_name)
        out_shape = getattr(op_def, "infer_shape")(*args, **kwargs)
    except Exception:
        out_shape = ()
    out_dtype = getattr(args[0], "dtype", "float32") if args else "float32"
    return _emit_shape_node(func_name, list(args), kwargs, typing.cast("Sequence[int | str]", out_shape), out_dtype)
