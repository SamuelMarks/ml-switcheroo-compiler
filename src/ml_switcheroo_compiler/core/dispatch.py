"""Dispatch utilities for the ml-switcheroo compiler."""

from __future__ import annotations

import typing

from ml_switcheroo_compiler.backends.registry import (
    adapt_backend_module,
    get_active_backend,
)
from ml_switcheroo_compiler.core.config import config as core_config
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.shape import IRAttrValue, OpValue, is_variadic_callable


@typing.runtime_checkable
class _HasDTypeProtocol(typing.Protocol):
    """Protocol for objects exposing a `.dtype` attribute."""

    @property
    def dtype(self) -> DType | str | type | None:
        """Return data type.

        Returns:
            DType | str | type | None: Data type.
        """
        ...


def dispatch(module_name: str, func_name: str, *args: OpValue, **kwargs: IRAttrValue) -> OpValue:
    """Dynamically dispatch a function to the active backend.

    Args:
        module_name (str): The name of the backend submodule (e.g., 'lax').
        func_name (str): The name of the function to execute (e.g., 'abs_p').
        *args (OpValue): Positional arguments for the function.
        **kwargs (IRAttrValue): Keyword arguments for the function.

    Returns:
        OpValue: The result of the function execution.

    Raises:
        ValueError: If not supported in the active backend or in tracing mode.
    """
    if core_config.eager_mode:
        backend = get_active_backend()
        has_mod, submodule = adapt_backend_module(backend.module).resolve_symbol(module_name)
        if has_mod and submodule is not None:
            fn = adapt_backend_module(submodule).get_op(func_name)
            if fn is not None:
                return fn(*args, **kwargs)
        raise ValueError(f"{func_name} is not supported in the active backend.")
    from ml_switcheroo_compiler.ops.registry import get_op
    from ml_switcheroo_compiler.ops.shape.utils import _emit_shape_node

    try:
        op_cls = get_op(func_name)
        _callable_fn_2: OpValue = op_cls().infer_shape
        assert is_variadic_callable(_callable_fn_2)
        out_shape = _callable_fn_2(*args, **kwargs)
    except (
        AttributeError,
        ImportError,
        LookupError,
        OSError,
        ArithmeticError,
        RuntimeError,
        SyntaxError,
        TypeError,
        ValueError,
    ):
        out_shape = ()
    first_arg: OpValue = args[0] if args else None
    raw_dtype = first_arg.dtype if isinstance(first_arg, _HasDTypeProtocol) else "float32"
    out_dtype: DType | str = raw_dtype if isinstance(raw_dtype, DType) else str(raw_dtype or "float32")
    return _emit_shape_node(func_name, list(args), kwargs, out_shape, out_dtype)
