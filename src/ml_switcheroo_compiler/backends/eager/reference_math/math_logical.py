"""Generic backend-parametric eager reference operations for `reference_math.math_logical`."""

from __future__ import annotations

import types

from ml_switcheroo_compiler.backends.eager.utils import _resolve_eager_mod
from ml_switcheroo_compiler.backends.eager_registry import (
    BackendArg,
    global_eager_registry,
)
from ml_switcheroo_compiler.backends.protocols import BackendModuleProtocol, adapt_backend_module


@global_eager_registry.register("NotEqual")
def _np_not_equal(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager np not equal operation on the active backend array.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None):
            Input `backend_module` argument (backend module).
        *args (BackendArg): Positional args.
        **kwargs (BackendArg): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    return adapt_backend_module(backend_module).require_symbol("not_equal")(*args, **kwargs)


@global_eager_registry.register("Greater")
def _np_greater(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager np greater operation on the active backend array.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None):
            Input `backend_module` argument (backend module).
        *args (BackendArg): Positional args.
        **kwargs (BackendArg): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    return adapt_backend_module(backend_module).require_symbol("greater")(*args, **kwargs)


@global_eager_registry.register("GreaterEqual")
def _np_greater_equal(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager np greater equal operation on the active backend array.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None):
            Input `backend_module` argument (backend module).
        *args (BackendArg): Positional args.
        **kwargs (BackendArg): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    return adapt_backend_module(backend_module).require_symbol("greater_equal")(*args, **kwargs)


@global_eager_registry.register("Less")
def _np_less(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager np less operation on the active backend array.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None):
            Input `backend_module` argument (backend module).
        *args (BackendArg): Positional args.
        **kwargs (BackendArg): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    return adapt_backend_module(backend_module).require_symbol("less")(*args, **kwargs)


@global_eager_registry.register("LessEqual")
def _np_less_equal(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager np less equal operation on the active backend array.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None):
            Input `backend_module` argument (backend module).
        *args (BackendArg): Positional args.
        **kwargs (BackendArg): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    return adapt_backend_module(backend_module).require_symbol("less_equal")(*args, **kwargs)


@global_eager_registry.register("LogicalAnd")
def _np_logical_and(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager np logical and operation on the active backend array.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None):
            Input `backend_module` argument (backend module).
        *args (BackendArg): Positional args.
        **kwargs (BackendArg): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    return adapt_backend_module(backend_module).require_symbol("logical_and")(*args, **kwargs)


@global_eager_registry.register("LogicalOr")
def _np_logical_or(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager np logical or operation on the active backend array.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None):
            Input `backend_module` argument (backend module).
        *args (BackendArg): Positional args.
        **kwargs (BackendArg): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    return adapt_backend_module(backend_module).require_symbol("logical_or")(*args, **kwargs)


@global_eager_registry.register("LogicalNot")
def _np_logical_not(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager np logical not operation on the active backend array.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None):
            Input `backend_module` argument (backend module).
        *args (BackendArg): Positional args.
        **kwargs (BackendArg): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    return adapt_backend_module(backend_module).require_symbol("logical_not")(*args, **kwargs)


@global_eager_registry.register("LogicalXor")
def _np_logical_xor(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager np logical xor operation on the active backend array.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None):
            Input `backend_module` argument (backend module).
        *args (BackendArg): Positional args.
        **kwargs (BackendArg): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    return adapt_backend_module(backend_module).require_symbol("logical_xor")(*args, **kwargs)


@global_eager_registry.register("Where")
def _np_where(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager np where operation on the active backend array.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None):
            Input `backend_module` argument (backend module).
        *args (BackendArg): Positional args.
        **kwargs (BackendArg): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    mod = adapt_backend_module(backend_module)
    norm_args: tuple[BackendArg, ...] = (
        (mod.require_symbol("atleast_1d")(args[0]),)
        if len(args) == 1 and not kwargs and mod.has_symbol("atleast_1d")
        else args
    )
    return mod.require_symbol("where")(*norm_args, **kwargs)


@global_eager_registry.register("Assert")
def _np_assert(
    backend_module: BackendModuleProtocol | types.ModuleType | None, condition: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager np assert operation on the active backend array.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None):
            Input `backend_module` argument (backend module).
        condition (BackendArg): Input `condition` argument (condition).
        **kwargs (BackendArg): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.

    Raises:
        AssertionError: If invalid inputs or unsupported runtime conditions trigger `AssertionError`.
    """
    arr_mod = _resolve_eager_mod(backend_module)
    if not adapt_backend_module(arr_mod).require_symbol("all")(condition):
        data = kwargs.get("data", ["Assertion failed."])
        raise AssertionError(data)
    return adapt_backend_module(backend_module).require_symbol("array")(0)
