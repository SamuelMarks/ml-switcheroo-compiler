"""Unified operation dispatcher routing eager execution and symbolic tracing."""

from __future__ import annotations

import typing
from collections.abc import Callable

import ml_switcheroo_compiler.backends.registry as _backend_registry
from ml_switcheroo_compiler.backends.registry import adapt_backend_module
from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.shape import (
    IRAttrValue,
    OpValue,
    extract_dtype,
    extract_shape,
    is_variadic_callable,
)
from ml_switcheroo_compiler.ops.eager_evaluator import EagerEvaluator
from ml_switcheroo_compiler.tracing.builder import TracingNodeBuilder
from ml_switcheroo_compiler.tracing.state import global_tracing_state

if typing.TYPE_CHECKING:
    from ml_switcheroo_compiler.core.tensor import Tensor

DispatchArg = OpValue


@typing.runtime_checkable
class _HasDTypeAttrProtocol(typing.Protocol):
    """Protocol for operands exposing a `.dtype` attribute."""

    @property
    def dtype(self) -> DType | str | type | None:
        """Return operand data type.

        Returns:
            DType | str | type | None: Operand data type.
        """
        ...


def is_eager_execution_active() -> bool:
    """Return whether eager execution mode is active in the current context.

    Returns:
        bool: True when operations should execute eagerly rather than emit IR nodes.
    """
    return bool(config.eager_mode)


def dispatch_op(
    op_type: str,
    *args: Tensor | DispatchArg,
    **kwargs: Tensor | DispatchArg,
) -> Tensor:
    """Route a single-output canonical operation to `EagerEvaluator` or `TracingNodeBuilder`.

    Args:
        op_type (str): Canonical PascalCase operator name.
        *args (Tensor | DispatchArg): Positional operands.
        **kwargs (Tensor | DispatchArg): Keyword operator attributes.

    Returns:
        Tensor: Resulting single tensor.

    Raises:
        RuntimeError: If called outside of an active tracing context when eager mode is disabled.
    """
    if is_eager_execution_active():
        res_eager = EagerEvaluator.evaluate(op_type, *args, **kwargs)
        return res_eager[0] if isinstance(res_eager, tuple) else res_eager

    if not global_tracing_state.is_tracing:
        msg = f"Cannot emit {op_type} node outside of a tracing context."
        raise RuntimeError(msg)

    return TracingNodeBuilder.emit_tracing_node(op_type, *args, **kwargs)


def dispatch_multi_op(
    op_type: str,
    *args: Tensor | DispatchArg,
    **kwargs: Tensor | DispatchArg,
) -> Tensor | tuple[Tensor, ...]:
    """Route an operation that may return multiple output tensors (e.g. `Svd`, `Qr`).

    Args:
        op_type (str): Canonical PascalCase operator name.
        *args (Tensor | DispatchArg): Positional operands.
        **kwargs (Tensor | DispatchArg): Keyword operator attributes.

    Returns:
        Tensor | tuple[Tensor, ...]: Single result tensor or multi-output tensor tuple.

    Raises:
        RuntimeError: If called outside of an active tracing context when eager mode is disabled.
    """
    if is_eager_execution_active():
        res_eager = EagerEvaluator.evaluate(op_type, *args, **kwargs)
        if isinstance(res_eager, tuple):
            return res_eager[0] if len(res_eager) == 1 else res_eager
        return res_eager
    return dispatch_op(op_type, *args, **kwargs)


def dispatch_backend_eager_op(
    op_name: str,
    args: tuple[DispatchArg, ...],
    kwargs: dict[str, DispatchArg],
) -> DispatchArg:
    """Execute `op_name` eagerly via the active backend and wrap result(s) as `Tensor`.

    Args:
        op_name (str): Canonical PascalCase operation name.
        args (tuple[DispatchArg, ...]): Positional operands.
        kwargs (dict[str, DispatchArg]): Keyword operation attributes.

    Returns:
        DispatchArg: Wrapped `Tensor` or `tuple[Tensor, ...]` output.
    """
    from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig

    backend = _backend_registry.get_active_backend()
    raw_args = [a.data if isinstance(a, Tensor) else a for a in args]
    data = backend.execute_op(op_name, *raw_args, **kwargs)
    first_tensor = next((a for a in args if isinstance(a, Tensor)), None)
    device = first_tensor.device if first_tensor is not None else None
    dtype = first_tensor.dtype if first_tensor is not None else "float32"
    if isinstance(data, tuple):
        return tuple(
            Tensor(
                backend.array(d),
                TensorConfig(extract_shape(backend.array(d)), extract_dtype(d, dtype), device),
            )
            for d in data
        )
    return Tensor(
        backend.array(data),
        TensorConfig(extract_shape(backend.array(data)), dtype if first_tensor is not None else None, device),
    )


def dispatch_submodule_op(
    module_name: str,
    func_name: str,
    args: tuple[OpValue, ...],
    kwargs: dict[str, IRAttrValue],
    backend_getter: Callable[[], OpValue] | None = None,
) -> OpValue:
    """Route a submodule-scoped backend operation to eager execution or shape-inferred tracing.

    Args:
        module_name (str): Target backend submodule name (e.g. `'linalg'` or `'lax'`).
        func_name (str): Function identifier within `module_name`.
        args (tuple[OpValue, ...]): Positional operands.
        kwargs (dict[str, IRAttrValue]): Keyword attributes.
        backend_getter (Callable[[], OpValue] | None): Optional active-backend provider override.

    Returns:
        OpValue: Eager result or emitted symbolic shape node output.

    Raises:
        ValueError: If `module_name.func_name` is unsupported in eager mode.
    """
    if is_eager_execution_active():
        provider = backend_getter if backend_getter is not None else _backend_registry.get_active_backend
        backend = provider()
        has_raw_mod, raw_mod = adapt_backend_module(backend).resolve_symbol("module")
        target_mod = raw_mod if has_raw_mod and raw_mod is not None else backend
        has_mod, submodule = adapt_backend_module(target_mod).resolve_symbol(module_name)
        if has_mod and submodule is not None:
            fn = adapt_backend_module(submodule).get_op(func_name)
            if fn is not None:
                return fn(*args, **kwargs)
        raise ValueError(f"{func_name} is not supported in the active backend.")

    from ml_switcheroo_compiler.ops.registry import get_op
    from ml_switcheroo_compiler.ops.shape.utils import _emit_shape_node

    try:
        op_cls = get_op(func_name)
        infer_fn: OpValue = op_cls().infer_shape
        assert is_variadic_callable(infer_fn)
        out_shape = infer_fn(*args, **kwargs)
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
    raw_dtype = first_arg.dtype if isinstance(first_arg, _HasDTypeAttrProtocol) else "float32"
    out_dtype: DType | str = raw_dtype if isinstance(raw_dtype, DType) else str(raw_dtype or "float32")
    return _emit_shape_node(func_name, list(args), kwargs, out_shape, out_dtype)


__all__ = [
    "DispatchArg",
    "dispatch_backend_eager_op",
    "dispatch_multi_op",
    "dispatch_op",
    "dispatch_submodule_op",
    "is_eager_execution_active",
]
