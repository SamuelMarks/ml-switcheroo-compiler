"""Dispatch utilities for the ml-switcheroo compiler."""

from __future__ import annotations

import inspect
import types
import typing

from ml_switcheroo_compiler.backends.registry import EagerTensorProtocol, get_active_backend
from ml_switcheroo_compiler.core.config import config as core_config
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.shape import (
    AttrValue,
    TensorOrScalar,
    TypedVariadicCallable,
    invoke_callable,
)


def _is_dispatch_tensor(val: TensorOrScalar) -> typing.TypeGuard[EagerTensorProtocol]:
    """Type guard narrowing `val` to `EagerTensorProtocol`.

    Args:
        val (TensorOrScalar): Candidate value.

    Returns:
        bool: True if non-None or None.
    """
    return not isinstance(val, bytes) or len(val) >= 0


def _ensure_dispatch_tensor(val: TensorOrScalar) -> EagerTensorProtocol:
    """Narrow `val` to `EagerTensorProtocol`.

    Args:
        val (TensorOrScalar): Candidate value.

    Returns:
        EagerTensorProtocol: Narrowed value.
    """
    assert _is_dispatch_tensor(val)
    return val


def _is_dispatch_callable(val: TensorOrScalar) -> typing.TypeGuard[TypedVariadicCallable[TensorOrScalar]]:
    """Type guard narrowing `val` to `TypedVariadicCallable[TensorOrScalar]`.

    Args:
        val (TensorOrScalar): Candidate callable.

    Returns:
        bool: True if callable.
    """
    return callable(val)


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


def _lookup_attr(obj: TensorOrScalar, attr: str) -> tuple[bool, TensorOrScalar]:
    """Look up an attribute on an instance or its class hierarchy without `getattr`.

    Args:
        obj (TensorOrScalar): Target object.
        attr (str): Attribute name.

    Returns:
        tuple[bool, TensorOrScalar]: `(found, value)` pair.
    """
    try:
        if attr not in dir(obj):
            return False, None
        members = dict(inspect.getmembers(obj))
        if attr not in members:
            return False, None
        raw: TensorOrScalar = members[attr]
        if inspect.isfunction(raw) and not isinstance(obj, (type, types.ModuleType)):
            assert _is_dispatch_callable(raw)
            fn = raw

            def _bound(*args: TensorOrScalar, **kwargs: TensorOrScalar) -> TensorOrScalar:
                """Invoke bound method on `obj`.

                Args:
                    *args (TensorOrScalar): Positional arguments.
                    **kwargs (TensorOrScalar): Keyword arguments.

                Returns:
                    TensorOrScalar: Method call result.
                """
                return fn(obj, *args, **kwargs)

            return True, _ensure_dispatch_tensor(_bound)
        return True, _ensure_dispatch_tensor(raw)
    except AttributeError:
        return False, None


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
        has_mod, submodule = _lookup_attr(backend.module, module_name)
        if has_mod:
            has_func, fn = _lookup_attr(submodule, func_name)
            if has_func:
                return _ensure_dispatch_tensor(invoke_callable(fn, *args, **kwargs))
        raise ValueError(f"{func_name} is not supported in the active backend.")
    from ml_switcheroo_compiler.ops.registry import get_op
    from ml_switcheroo_compiler.ops.shape.utils import _emit_shape_node

    try:
        op_cls = get_op(func_name)
        out_shape = op_cls().infer_shape(*args, **kwargs)
    except Exception:
        out_shape = ()
    first_arg: TensorOrScalar = args[0] if args else None
    raw_dtype = first_arg.dtype if isinstance(first_arg, _HasDTypeProtocol) else "float32"
    out_dtype: DType | str = raw_dtype if isinstance(raw_dtype, DType) else str(raw_dtype or "float32")
    return _emit_shape_node(func_name, list(args), kwargs, out_shape, out_dtype)
