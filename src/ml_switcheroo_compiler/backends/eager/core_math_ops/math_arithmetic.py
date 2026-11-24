"""Operations and utilities for `ml_switcheroo_compiler.backends.eager.core_math_ops.math_arithmetic`."""

from __future__ import annotations

import builtins  # noqa: F401
import types

from ml_switcheroo_compiler.backends.eager.utils import _resolve_eager_mod
from ml_switcheroo_compiler.backends.eager_registry import (
    BackendArg,
    coerce_numeric,
    coerce_operand_list,
    global_eager_registry,
)
from ml_switcheroo_compiler.backends.protocols import BackendModuleProtocol, adapt_backend_module


@global_eager_registry.register("ScalarMul")
def _scalar_mul(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Multiply a tensor and a scalar operand elementwise on the active backend.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None): Active backend execution module.
        *args (BackendArg): Positional operands ``(x, scalar)`` or ``(scalar, x)``.
        **kwargs (BackendArg): Optional keyword arguments forwarded to ``multiply``.

    Returns:
        BackendArg: Scaled tensor result.
    """
    return adapt_backend_module(backend_module).require_symbol("multiply")(args[0], args[1], **kwargs)


@global_eager_registry.register("TrueDivide")
def _true_divide(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager true divide operation on the active backend array.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            BackendArg: Computed `BackendArg` output.
    """
    func = adapt_backend_module(backend_module).get_symbol(
        "divide", adapt_backend_module(backend_module).get_symbol("true_divide", None)
    )
    return func(*args, **kwargs) if func else None


@global_eager_registry.register("Fmod")
def _fmod(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager fmod operation on the active backend array.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            BackendArg: Computed `BackendArg` output.
    """
    func = adapt_backend_module(backend_module).get_symbol(
        "fmod",
        adapt_backend_module(backend_module).get_symbol(
            "remainder", adapt_backend_module(backend_module).get_symbol("mod", None)
        ),
    )
    return func(*args, **kwargs) if func else None


@global_eager_registry.register("AccumulateN")
def _accumulate_n(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager accumulate n operation on the active backend array.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            BackendArg: Computed `BackendArg` output.
    """
    inputs = coerce_operand_list(args[0] if len(args) > 0 else kwargs.get("inputs", []))
    if not inputs:
        return _resolve_eager_mod(backend_module).zeros(())
    res = coerce_numeric(inputs[0])
    for i in range(1, len(inputs)):
        res = res + coerce_numeric(inputs[i])
    return res


@global_eager_registry.register("AssignAdd")
def _assign_add(
    backend_module: BackendModuleProtocol | types.ModuleType | None,
    ref: BackendArg,
    value: BackendArg,
    **kwargs: BackendArg,
) -> BackendArg:
    """Execute the eager assign add operation on the active backend array.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        ref: Input `ref` argument (ref).
        value: Input `value` argument (value).
        **kwargs: Keyword args.

    Returns:
            BackendArg: Computed `BackendArg` output.
    """
    return coerce_numeric(ref) + coerce_numeric(value)


@global_eager_registry.register("AssignSub")
def _assign_sub(
    backend_module: BackendModuleProtocol | types.ModuleType | None,
    ref: BackendArg,
    value: BackendArg,
    **kwargs: BackendArg,
) -> BackendArg:
    """Execute the eager assign sub operation on the active backend array.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        ref: Input `ref` argument (ref).
        value: Input `value` argument (value).
        **kwargs: Keyword args.

    Returns:
            BackendArg: Computed `BackendArg` output.
    """
    return coerce_numeric(ref) - coerce_numeric(value)


@global_eager_registry.register("AddN")
def _add_n(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager add n operation on the active backend array.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            BackendArg: Computed `BackendArg` output.
    """
    inputs = coerce_operand_list(args[0] if len(args) > 0 else kwargs.get("inputs", []))
    if not inputs:
        return _resolve_eager_mod(backend_module).zeros(())
    res = coerce_numeric(inputs[0])
    for i in range(1, len(inputs)):
        res = res + coerce_numeric(inputs[i])
    return res


@global_eager_registry.register("Modf")
def _modf(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager modf operation on the active backend array.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            BackendArg: Computed `BackendArg` output.
    """
    mod = backend_module
    return adapt_backend_module(mod).require_symbol("modf")(*args, **kwargs)


@global_eager_registry.register("RavelMultiIndex")
def _ravelmultiindex(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager ravelmultiindex operation on the active backend array.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            BackendArg: Computed `BackendArg` output.
    """
    mod = backend_module
    return adapt_backend_module(mod).require_symbol("ravel_multi_index")(*args, **kwargs)


@global_eager_registry.register("ScatterMul")
def _np_scattermul(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager np scattermul operation on the active backend array.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            BackendArg: Computed `BackendArg` output.
    """
    func = adapt_backend_module(backend_module).get_symbol(
        "scattermul", adapt_backend_module(backend_module).get_symbol("scattermul", None)
    )
    if func is not None:
        return func(*args, **kwargs)
    return args[0]


@global_eager_registry.register("StringSubstr")
def _np_stringsubstr(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager np stringsubstr operation on the active backend array.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            BackendArg: Computed `BackendArg` output.
    """
    func = adapt_backend_module(backend_module).get_symbol(
        "stringsubstr", adapt_backend_module(backend_module).get_symbol("stringsubstr", None)
    )
    if func is not None:
        return func(*args, **kwargs)
    return _resolve_eager_mod(backend_module).asarray(args[0], dtype=str)


@global_eager_registry.register("TensorScatterSub")
def _np_tensorscattersub(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager np tensorscattersub operation on the active backend array.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            BackendArg: Computed `BackendArg` output.
    """
    func = adapt_backend_module(backend_module).get_symbol(
        "tensorscattersub", adapt_backend_module(backend_module).get_symbol("tensorscattersub", None)
    )
    if func is not None:
        return func(*args, **kwargs)
    return args[0]


@global_eager_registry.register("TruncateDiv")
def _np_truncatediv(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager np truncatediv operation on the active backend array.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            BackendArg: Computed `BackendArg` output.
    """
    func = adapt_backend_module(backend_module).get_symbol(
        "truncatediv", adapt_backend_module(backend_module).get_symbol("truncatediv", None)
    )
    if func is not None:
        return func(*args, **kwargs)
    mod = _resolve_eager_mod(backend_module)
    return mod.trunc(mod.divide(args[0], args[1]))


@global_eager_registry.register("TruncateMod")
def _np_truncatemod(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager np truncatemod operation on the active backend array.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            BackendArg: Computed `BackendArg` output.
    """
    func = adapt_backend_module(backend_module).get_symbol(
        "truncatemod", adapt_backend_module(backend_module).get_symbol("truncatemod", None)
    )
    if func is not None:
        return func(*args, **kwargs)
    return _resolve_eager_mod(backend_module).mod(args[0], args[1])


@global_eager_registry.register("Xdivy")
def _np_xdivy(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager np xdivy operation on the active backend array.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            BackendArg: Computed `BackendArg` output.
    """
    func = adapt_backend_module(backend_module).get_symbol(
        "xdivy", adapt_backend_module(backend_module).get_symbol("xdivy", None)
    )
    if func is not None:
        return func(*args, **kwargs)
    mod = _resolve_eager_mod(backend_module)
    safe_y = mod.where(args[0] == 0, 1, args[1])
    return mod.where(args[0] == 0, 0, mod.divide(args[0], safe_y))
