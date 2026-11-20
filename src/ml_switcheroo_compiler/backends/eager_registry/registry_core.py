"""Core eager operation registry class, universal fallback dispatcher, and singleton instances."""

from __future__ import annotations

import typing
from collections.abc import Callable

from ml_switcheroo_compiler.backends.eager_registry import require_callable
from ml_switcheroo_compiler.core.shape import CallableLike

from .protocols import EagerRegisteredCallable
from .types_base import BackendArg, BackendOperand

_F = typing.TypeVar("_F")

UNIVERSAL_FALLBACK_OP_TABLE: dict[str, str] = {
    "matmul": "matmul",
    "add": "add",
    "sub": "subtract",
    "mul": "multiply",
    "div": "divide",
    "truedivide": "divide",
}


class EagerOpRegistry:
    """Registry mapping canonical IR operation names to eager runtime implementations."""

    def __init__(self) -> None:
        """Initialize an empty eager operation registry."""
        self._registry: dict[str, EagerRegisteredCallable] = {}

    def register(self, op_type: str) -> Callable[[_F], _F]:
        """Return a decorator registering an eager implementation for `op_type`.

        Args:
            op_type (str): Canonical operation identifier (e.g., `'Add'`, `'CustomVJP'`).

        Returns:
            Callable[[_F], _F]: Decorator preserving the original function signature.
        """

        def decorator(func: _F) -> _F:
            """Store `func` in the registry under `op_type`.

            Args:
                func (_F): Eager kernel callable.

            Returns:
                _F: Unmodified `func`.
            """
            cand_fn: CallableLike[BackendOperand] | EagerRegisteredCallable | _F = func
            assert isinstance(cand_fn, EagerRegisteredCallable)
            self._registry[op_type] = cand_fn
            return func

        return decorator

    def get(self, op_type: str) -> EagerRegisteredCallable | None:
        """Look up a registered eager operation by canonical name, falling back to `global_eager_registry`.

        Args:
            op_type (str): Canonical operation identifier.

        Returns:
            EagerRegisteredCallable | None: Registered eager callable or `None`.
        """
        local_fn = self._registry.get(op_type)
        if local_fn is not None or self is global_eager_registry:
            return local_fn
        return global_eager_registry.get(op_type)

    def dispatch(self, op_type: str, *args: BackendArg, **kwargs: BackendArg) -> BackendArg:
        """Dispatch `op_type` to a registered kernel, global registry fallback, or backend method.

        Args:
            op_type (str): Canonical operation identifier.
            *args (BackendArg): Positional eager operands.
            **kwargs (BackendArg): Keyword attributes.

        Returns:
            BackendArg: Evaluated eager tensor or scalar output.

        Raises:
            ValueError: If `op_type` is not registered and no backend fallback exists.
        """
        func = self.get(op_type)
        if func is not None:
            return func(*args, **kwargs)

        if self is not global_eager_registry:
            func = global_eager_registry.get(op_type)
            if func is not None:
                return func(*args, **kwargs)

        return _dispatch_universal_fallback(op_type, args, kwargs)


def _dispatch_universal_fallback(
    op_type: str,
    args: tuple[BackendArg, ...],
    kwargs: dict[str, BackendArg],
) -> BackendArg:
    """Resolve and invoke universal method or backend-module fallback for `op_type`.

    Args:
        op_type (str): Canonical operation name.
        args (tuple[BackendArg, ...]): Positional operands.
        kwargs (dict[str, BackendArg]): Keyword operands (including optional `backend_module`).

    Returns:
        BackendArg: Result of invoking the resolved fallback method.

    Raises:
        ValueError: If no matching method or alias exists on the operand or backend module.
    """
    from ml_switcheroo_compiler.backends.protocols import adapt_backend_module

    op_name = op_type.lower()
    if args:
        has_arg_op, arg_op = adapt_backend_module(args[0]).resolve_symbol(op_name)
        if has_arg_op and arg_op is not None:
            return require_callable(arg_op)(*args[1:], **kwargs)

    backend = kwargs.get("backend_module")
    op_args = args[1:] if backend is None and len(args) > 0 else args
    if backend is None and len(args) > 0:
        backend = args[0]

    if backend is not None:
        has_bk_op, bk_op = adapt_backend_module(backend).resolve_symbol(op_name)
        if has_bk_op and bk_op is not None:
            return require_callable(bk_op)(*op_args, **kwargs)

        mapped_name = UNIVERSAL_FALLBACK_OP_TABLE.get(op_name)
        has_mapped_op, mapped_op = (
            adapt_backend_module(backend).resolve_symbol(mapped_name) if mapped_name is not None else (False, None)
        )
        if has_mapped_op and mapped_op is not None:
            return require_callable(mapped_op)(*op_args, **kwargs)

    msg = (
        f"Operation '{op_type}' not found in registry and no universal fallback could be inferred for "
        f"backend {backend}."
    )
    raise ValueError(msg)


global_eager_registry = EagerOpRegistry()
mlx_eager_registry = EagerOpRegistry()
numpy_eager_registry = EagerOpRegistry()
pure_python_eager_registry = EagerOpRegistry()
