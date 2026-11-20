"""Dispatch utilities delegating to the unified operation dispatcher."""

from __future__ import annotations

from ml_switcheroo_compiler.backends.registry import get_active_backend
from ml_switcheroo_compiler.core.shape import IRAttrValue, OpValue
from ml_switcheroo_compiler.ops.dispatcher import dispatch_submodule_op


def dispatch(module_name: str, func_name: str, *args: OpValue, **kwargs: IRAttrValue) -> OpValue:
    """Dynamically dispatch a submodule function to the active backend or tracing builder.

    Args:
        module_name (str): The name of the backend submodule (e.g., 'lax').
        func_name (str): The name of the function to execute (e.g., 'abs_p').
        *args (OpValue): Positional arguments for the function.
        **kwargs (IRAttrValue): Keyword arguments for the function.

    Returns:
        OpValue: The result of the function execution.

    Raises:
        ValueError: If not supported in the active backend in eager mode.
    """
    return dispatch_submodule_op(
        module_name,
        func_name,
        args,
        kwargs,
        backend_getter=get_active_backend,
    )


__all__ = ["dispatch"]
