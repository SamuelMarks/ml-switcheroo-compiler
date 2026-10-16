"""Core utilities."""

from __future__ import annotations

import builtins  # noqa: F401
from collections.abc import Sequence
from types import ModuleType
from typing import cast

from ml_switcheroo_compiler.backends.eager_registry import (
    EagerTensorProtocol,
    EagerValue,
    global_eager_registry,
)
from ml_switcheroo_compiler.distributed.host_collectives import np


@global_eager_registry.register("TrueDivide")
def _true_divide(backend_module: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Evaluate _true_divide operation.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            object: Result.
    """
    func = getattr(backend_module, "divide", getattr(backend_module, "true_divide", None))
    return func(*args, **kwargs) if func else None


@global_eager_registry.register("Fmod")
def _fmod(backend_module: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Evaluate _fmod operation.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            object: Result.
    """
    func = getattr(backend_module, "fmod", getattr(backend_module, "remainder", getattr(backend_module, "mod", None)))
    return func(*args, **kwargs) if func else None


@global_eager_registry.register("AccumulateN")
def _accumulate_n(backend_module: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Evaluate _accumulate_n operation.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            object: Result.
    """
    inputs = cast(Sequence[EagerValue], args[0] if len(args) > 0 else kwargs.get("inputs", []))
    if not inputs:
        return cast(EagerValue, np.zeros(()))
    res = cast(EagerTensorProtocol, inputs[0])
    for i in range(1, len(inputs)):
        res = res + cast(EagerTensorProtocol, inputs[i])
    return res


@global_eager_registry.register("AssignAdd")
def _assign_add(backend_module: EagerValue, ref: EagerValue, value: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Evaluate _assign_add operation.

    Args:
        backend_module: The backend_module parameter.
        ref: The ref parameter.
        value: The value parameter.
        **kwargs: Keyword args.

    Returns:
            object: Result.
    """
    return cast(EagerTensorProtocol, ref) + cast(EagerTensorProtocol, value)


@global_eager_registry.register("AssignSub")
def _assign_sub(backend_module: EagerValue, ref: EagerValue, value: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Evaluate _assign_sub operation.

    Args:
        backend_module: The backend_module parameter.
        ref: The ref parameter.
        value: The value parameter.
        **kwargs: Keyword args.

    Returns:
            object: Result.
    """
    return cast(EagerTensorProtocol, ref) - cast(EagerTensorProtocol, value)


@global_eager_registry.register("AddN")
def _add_n(backend_module: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Evaluate _add_n operation.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            object: Result.
    """
    inputs = cast(Sequence[EagerValue], args[0] if len(args) > 0 else kwargs.get("inputs", []))
    if not inputs:
        return cast(EagerValue, np.zeros(()))
    res = cast(EagerTensorProtocol, inputs[0])
    for i in range(1, len(inputs)):
        res = res + cast(EagerTensorProtocol, inputs[i])
    return res


@global_eager_registry.register("Modf")
def _modf(backend_module: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Evaluate _modf operation.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            object: Result.
    """
    mod = cast(ModuleType, backend_module)
    return cast(EagerValue, mod.modf(*args, **kwargs))


@global_eager_registry.register("RavelMultiIndex")
def _ravelmultiindex(backend_module: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Evaluate _ravelmultiindex operation.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            object: Result.
    """
    mod = cast(ModuleType, backend_module)
    return cast(EagerValue, mod.ravel_multi_index(*args, **kwargs))


@global_eager_registry.register("ScatterMul")
def _np_scattermul(backend_module: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Evaluate _np_scattermul operation.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            object: Result.
    """
    func = getattr(backend_module, "scattermul", getattr(backend_module, "scattermul", None))
    if func is not None:
        return cast(EagerValue, func(*args, **kwargs))
    return args[0]


@global_eager_registry.register("StringSubstr")
def _np_stringsubstr(backend_module: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Evaluate _np_stringsubstr operation.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            object: Result.
    """
    func = getattr(backend_module, "stringsubstr", getattr(backend_module, "stringsubstr", None))
    if func is not None:
        return cast(EagerValue, func(*args, **kwargs))
    return cast(EagerValue, np.asarray(args[0], dtype=str))


@global_eager_registry.register("TensorScatterSub")
def _np_tensorscattersub(backend_module: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Evaluate _np_tensorscattersub operation.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            object: Result.
    """
    func = getattr(backend_module, "tensorscattersub", getattr(backend_module, "tensorscattersub", None))
    if func is not None:
        return cast(EagerValue, func(*args, **kwargs))
    return args[0]


@global_eager_registry.register("TruncateDiv")
def _np_truncatediv(backend_module: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Evaluate _np_truncatediv operation.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            object: Result.
    """
    func = getattr(backend_module, "truncatediv", getattr(backend_module, "truncatediv", None))
    if func is not None:
        return cast(EagerValue, func(*args, **kwargs))
    return cast(EagerValue, np.trunc(np.divide(args[0], args[1])))


@global_eager_registry.register("TruncateMod")
def _np_truncatemod(backend_module: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Evaluate _np_truncatemod operation.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            object: Result.
    """
    func = getattr(backend_module, "truncatemod", getattr(backend_module, "truncatemod", None))
    if func is not None:
        return cast(EagerValue, func(*args, **kwargs))
    return cast(EagerValue, np.mod(args[0], args[1]))


@global_eager_registry.register("Xdivy")
def _np_xdivy(backend_module: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Evaluate _np_xdivy operation.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            object: Result.
    """
    func = getattr(backend_module, "xdivy", getattr(backend_module, "xdivy", None))
    if func is not None:
        return cast(EagerValue, func(*args, **kwargs))
    return cast(EagerValue, np.where(args[0] == 0, 0, np.divide(args[0], args[1])))
