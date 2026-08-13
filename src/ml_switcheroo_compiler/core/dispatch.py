# ruff: noqa: E402, D100, D103, D104, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, D101, D102, D107, E701, E722, F403, E711, E712, PLR0913, PLR0915
"""Dispatch utilities for the ml-switcheroo compiler."""

from typing import Any

from ml_switcheroo_compiler.backends.registry import get_active_backend
from ml_switcheroo_compiler.core.config import config as core_config


def dispatch(module_name: str, func_name: str, *args: Any, **kwargs: Any) -> Any:
    """Dynamically dispatch a function to the active backend.

    Args:
        module_name (str): The name of the backend submodule (e.g., 'lax').
        func_name (str): The name of the function to execute (e.g., 'abs_p').
        *args (object): Positional arguments for the function.
        **kwargs (object): Keyword arguments for the function.

    Returns: Any: The result of the function execution.

    Raises:
        ValueError: If not supported in the active backend or in tracing mode.
    """
    if core_config.eager_mode:
        backend = get_active_backend()
        if hasattr(backend.module, module_name):  # type: ignore  # Justification: Polymorphic / Duck Typing for Framework Agnosticism
            submodule = getattr(backend.module, module_name)  # type: ignore  # Justification: Polymorphic / Duck Typing for Framework Agnosticism
            if hasattr(submodule, func_name):
                return getattr(submodule, func_name)(*args, **kwargs)
        raise ValueError(f"{func_name} is not supported in the active backend.")
    from ml_switcheroo_compiler.ops.registry import get_op
    from ml_switcheroo_compiler.ops.shape.utils import _emit_shape_node

    try:
        op_def = get_op(func_name)
        out_shape = op_def.infer_shape(*args, **kwargs)  # type: ignore
    except Exception:
        out_shape = ()
    out_dtype = getattr(args[0], "dtype", "float32") if args else "float32"
    return _emit_shape_node(func_name, list(args), kwargs, out_shape, out_dtype)
