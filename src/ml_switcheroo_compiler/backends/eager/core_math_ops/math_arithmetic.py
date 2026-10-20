"""Implementation of math arithmetic operations and utilities for the backends subsystem (`ml_switcheroo_compiler.backends.eager.core_math_ops.math_arithmetic`)."""

from __future__ import annotations

import builtins  # noqa: F401
import typing
from collections.abc import Sequence

from ml_switcheroo_compiler.backends.eager.utils import _resolve_eager_mod
from ml_switcheroo_compiler.backends.eager_registry import (
    EagerTensorProtocol,
    EagerValue,
    global_eager_registry,
)
from ml_switcheroo_compiler.backends.protocols import BackendModuleProtocol


@global_eager_registry.register("TrueDivide")
def _true_divide(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager true divide operation on the active backend array.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Result.
    """
    func = getattr(backend_module, "divide", getattr(backend_module, "true_divide", None))
    return func(*args, **kwargs) if func else None


@global_eager_registry.register("Fmod")
def _fmod(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager fmod operation on the active backend array.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Result.
    """
    func = getattr(backend_module, "fmod", getattr(backend_module, "remainder", getattr(backend_module, "mod", None)))
    return func(*args, **kwargs) if func else None


@global_eager_registry.register("AccumulateN")
def _accumulate_n(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager accumulate n operation on the active backend array.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Result.
    """
    inputs = typing.cast(Sequence[EagerValue], args[0] if len(args) > 0 else kwargs.get("inputs", []))
    if not inputs:
        return _resolve_eager_mod(backend_module).zeros(())
    res = typing.cast(EagerTensorProtocol, inputs[0])
    for i in range(1, len(inputs)):
        res = res + typing.cast(EagerTensorProtocol, inputs[i])
    return res


@global_eager_registry.register("AssignAdd")
def _assign_add(backend_module: BackendModuleProtocol, ref: EagerValue, value: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager assign add operation on the active backend array.

    Args:
        backend_module: The backend_module parameter.
        ref: The ref parameter.
        value: The value parameter.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Result.
    """
    return typing.cast(EagerTensorProtocol, ref) + typing.cast(EagerTensorProtocol, value)


@global_eager_registry.register("AssignSub")
def _assign_sub(backend_module: BackendModuleProtocol, ref: EagerValue, value: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager assign sub operation on the active backend array.

    Args:
        backend_module: The backend_module parameter.
        ref: The ref parameter.
        value: The value parameter.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Result.
    """
    return typing.cast(EagerTensorProtocol, ref) - typing.cast(EagerTensorProtocol, value)


@global_eager_registry.register("AddN")
def _add_n(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager add n operation on the active backend array.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Result.
    """
    inputs = typing.cast(Sequence[EagerValue], args[0] if len(args) > 0 else kwargs.get("inputs", []))
    if not inputs:
        return _resolve_eager_mod(backend_module).zeros(())
    res = typing.cast(EagerTensorProtocol, inputs[0])
    for i in range(1, len(inputs)):
        res = res + typing.cast(EagerTensorProtocol, inputs[i])
    return res


@global_eager_registry.register("Modf")
def _modf(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager modf operation on the active backend array.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Result.
    """
    mod = backend_module
    return mod.modf(*args, **kwargs)


@global_eager_registry.register("RavelMultiIndex")
def _ravelmultiindex(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager ravelmultiindex operation on the active backend array.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Result.
    """
    mod = backend_module
    return mod.ravel_multi_index(*args, **kwargs)


@global_eager_registry.register("ScatterMul")
def _np_scattermul(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager np scattermul operation on the active backend array.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Result.
    """
    func = getattr(backend_module, "scattermul", getattr(backend_module, "scattermul", None))
    if func is not None:
        return typing.cast("EagerValue", func(*args, **kwargs))
    return args[0]


@global_eager_registry.register("StringSubstr")
def _np_stringsubstr(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager np stringsubstr operation on the active backend array.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Result.
    """
    func = getattr(backend_module, "stringsubstr", getattr(backend_module, "stringsubstr", None))
    if func is not None:
        return typing.cast("EagerValue", func(*args, **kwargs))
    return _resolve_eager_mod(backend_module).asarray(args[0], dtype=str)


@global_eager_registry.register("TensorScatterSub")
def _np_tensorscattersub(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager np tensorscattersub operation on the active backend array.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Result.
    """
    func = getattr(backend_module, "tensorscattersub", getattr(backend_module, "tensorscattersub", None))
    if func is not None:
        return typing.cast("EagerValue", func(*args, **kwargs))
    return args[0]


@global_eager_registry.register("TruncateDiv")
def _np_truncatediv(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager np truncatediv operation on the active backend array.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Result.
    """
    func = getattr(backend_module, "truncatediv", getattr(backend_module, "truncatediv", None))
    if func is not None:
        return typing.cast("EagerValue", func(*args, **kwargs))
    mod = _resolve_eager_mod(backend_module)
    return mod.trunc(mod.divide(args[0], args[1]))


@global_eager_registry.register("TruncateMod")
def _np_truncatemod(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager np truncatemod operation on the active backend array.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Result.
    """
    func = getattr(backend_module, "truncatemod", getattr(backend_module, "truncatemod", None))
    if func is not None:
        return typing.cast("EagerValue", func(*args, **kwargs))
    return _resolve_eager_mod(backend_module).mod(args[0], args[1])


@global_eager_registry.register("Xdivy")
def _np_xdivy(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager np xdivy operation on the active backend array.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Result.
    """
    func = getattr(backend_module, "xdivy", getattr(backend_module, "xdivy", None))
    if func is not None:
        return typing.cast("EagerValue", func(*args, **kwargs))
    mod = _resolve_eager_mod(backend_module)
    return mod.where(args[0] == 0, 0, mod.divide(args[0], args[1]))
