"""math_manipulation module."""

from __future__ import annotations

import builtins  # noqa: F401
import typing  # noqa: F401
from typing import cast

from ml_switcheroo_compiler.backends.eager.utils import host_backend
from ml_switcheroo_compiler.backends.eager_registry import (
    EagerTensorProtocol,
    EagerValue,
    global_eager_registry,
)


@global_eager_registry.register("AllGather")
def _all_gather(backend_module: EagerValue, tensor: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager all gather operation on the active backend array.

    Args:
        backend_module: The backend_module parameter.
        tensor: The tensor parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Result.
    """
    if hasattr(backend_module, "stack"):
        return cast(EagerValue, getattr(backend_module, "stack")([tensor]))
    if hasattr(backend_module, "array"):
        return cast(EagerValue, getattr(backend_module, "array")([tensor]))
    return tensor


@global_eager_registry.register("Argwhere")
def _argwhere(backend_module: EagerValue, a: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager argwhere operation on the active backend array.

    Args:
        backend_module: The backend_module parameter.
        a: The a parameter.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Result.
    """
    return cast(EagerValue, getattr(backend_module, "argwhere")(a))


@global_eager_registry.register("Extract")
def _extract(backend_module: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager extract operation on the active backend array.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Result.
    """
    func = getattr(backend_module, "extract", None)
    if func:
        return cast(EagerValue, func(*args, **kwargs))
    (condition, arr) = (args[0], args[1])
    return cast(EagerValue, host_backend.extract(host_backend.asarray(condition), host_backend.asarray(arr)))


@global_eager_registry.register("Pswapaxes")
def _pswapaxes(backend_module: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager pswapaxes operation on the active backend array.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Result.
    """
    if hasattr(backend_module, "lax") and hasattr(getattr(backend_module, "lax"), "pswapaxes"):
        return cast(EagerValue, getattr(backend_module, "lax").pswapaxes(*args, **kwargs))
    return args[0] if args else None


@global_eager_registry.register("Rot90")
def _rot90(backend_module: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager rot90 operation on the active backend array.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Result.
    """
    return cast(EagerValue, getattr(backend_module, "rot90")(*args, **kwargs))


@global_eager_registry.register("Nonzero")
def _nonzero(backend_module: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager nonzero operation on the active backend array.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Result.
    """
    return cast(EagerValue, getattr(backend_module, "nonzero")(*args, **kwargs))


@global_eager_registry.register("Repeat")
def _repeat(backend_module: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager repeat operation on the active backend array.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Result.
    """
    return cast(EagerValue, getattr(backend_module, "repeat")(*args, **kwargs))


@global_eager_registry.register("Tile")
def _tile(backend_module: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager tile operation on the active backend array.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Result.
    """
    return cast(EagerValue, getattr(backend_module, "tile")(*args, **kwargs))


@global_eager_registry.register("UpdateSlice")
def _updateslice(backend_module: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager updateslice operation on the active backend array.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Result.

    Raises:
        RuntimeError: An exception.
    """
    if not hasattr(backend_module, "array"):
        raise RuntimeError("Expected numpy-like backend")
    x = args[0]
    update = args[1] if len(args) > 1 else kwargs.get("update")
    start_indices = args[2] if len(args) > 2 else kwargs.get("start_indices")
    out = getattr(backend_module, "array")(x).copy()
    slices = tuple(slice(s, s + getattr(update, "shape", ())[i]) for (i, s) in enumerate(getattr(start_indices, "__iter__")()))
    out[slices] = update
    return cast(EagerValue, out)


@global_eager_registry.register("Flatnonzero")
def _flatnonzero(backend_module: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager flatnonzero operation on the active backend array.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Result.
    """
    func = getattr(backend_module, "flatnonzero", None)
    if func:
        return cast(EagerValue, func(*args, **kwargs))
    return cast(EagerValue, getattr(backend_module, "flatnonzero")(getattr(backend_module, "asarray")(args[0])))


@global_eager_registry.register("Fliplr")
def _fliplr(backend_module: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager fliplr operation on the active backend array.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Result.
    """
    func = getattr(backend_module, "fliplr", None)
    if func:
        return cast(EagerValue, func(*args, **kwargs))
    return cast(EagerValue, getattr(backend_module, "fliplr")(getattr(backend_module, "asarray")(args[0])))


@global_eager_registry.register("Flipud")
def _flipud(backend_module: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager flipud operation on the active backend array.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Result.
    """
    func = getattr(backend_module, "flipud", None)
    if func:
        return cast(EagerValue, func(*args, **kwargs))
    return cast(EagerValue, getattr(backend_module, "flipud")(getattr(backend_module, "asarray")(args[0])))


@global_eager_registry.register("Hsplit")
def _np_hsplit(backend_module: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager np hsplit operation on the active backend array.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Result.
    """
    func = getattr(backend_module, "hsplit", getattr(backend_module, "hsplit", None))
    if func is not None:
        return cast(EagerValue, func(*args, **kwargs))
    return cast(EagerValue, host_backend.hsplit(args[0], args[1]))


@global_eager_registry.register("ScatterApply")
def _np_scatterapply(backend_module: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager np scatterapply operation on the active backend array.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Result.

    Raises:
        RuntimeError: An exception.
    """
    func = getattr(backend_module, "scatterapply", getattr(backend_module, "scatterapply", None))
    if func is not None:
        return cast(EagerValue, func(*args, **kwargs))
    return args[0]


@global_eager_registry.register("ScatterNd")
def _np_scatternd(backend_module: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager np scatternd operation on the active backend array.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Result.
    """
    func = getattr(backend_module, "scatternd", getattr(backend_module, "scatternd", None))
    if func is not None:
        return cast(EagerValue, func(*args, **kwargs))
    return args[0]


@global_eager_registry.register("TensorArrayStack")
def _np_tensorarraystack(backend_module: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager np tensorarraystack operation on the active backend array.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Result.
    """
    func = getattr(backend_module, "tensorarraystack", getattr(backend_module, "tensorarraystack", None))
    if func is not None:
        return cast(EagerValue, func(*args, **kwargs))
    return cast(EagerValue, host_backend.stack(args[0]))


@global_eager_registry.register("TensorScatterUpdate")
def _np_tensorscatterupdate(backend_module: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager np tensorscatterupdate operation on the active backend array.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Result.
    """
    func = getattr(backend_module, "tensorscatterupdate", getattr(backend_module, "tensorscatterupdate", None))
    if func is not None:
        return cast(EagerValue, func(*args, **kwargs))
    return args[0]


@global_eager_registry.register("Unfold")
def _np_unfold(backend_module: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager np unfold operation on the active backend array.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Result.
    """
    func = getattr(backend_module, "unfold", getattr(backend_module, "unfold", None))
    if func is not None:
        return cast(EagerValue, func(*args, **kwargs))
    return args[0]


@global_eager_registry.register("Unstack")
def _np_unstack(backend_module: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager np unstack operation on the active backend array.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Result.
    """
    func = getattr(backend_module, "unstack", getattr(backend_module, "unstack", None))
    if func is not None:
        return cast(EagerValue, func(*args, **kwargs))
    axis: int = int(cast(int, kwargs.get("axis", 0)))
    return cast(EagerValue, host_backend.split(args[0], typing.cast(EagerTensorProtocol, args[0]).shape[axis], axis=axis))


@global_eager_registry.register("UpdateSlice")
def _np_updateslice(backend_module: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager np updateslice operation on the active backend array.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Result.

    Raises:
        RuntimeError: An exception.
    """
    func = getattr(backend_module, "updateslice", getattr(backend_module, "updateslice", None))
    if func is not None:
        return cast(EagerValue, func(*args, **kwargs))
    typing.cast(EagerTensorProtocol, args[0])[args[1]] = args[2]
    return args[0]


@global_eager_registry.register("Vsplit")
def _np_vsplit(backend_module: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager np vsplit operation on the active backend array.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Result.
    """
    func = getattr(backend_module, "vsplit", getattr(backend_module, "vsplit", None))
    if func is not None:
        return cast(EagerValue, func(*args, **kwargs))
    return cast(EagerValue, host_backend.vsplit(args[0], args[1]))


@global_eager_registry.register("Transpose")
def _np_transpose(backend_module: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager np transpose operation on the active backend array.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
        EagerValue: Transposed array result.
    """
    perm = kwargs.get("permutation", kwargs.get("axes", None))
    if hasattr(backend_module, "transpose"):
        if perm is not None:
            return cast(EagerValue, getattr(backend_module, "transpose")(args[0], axes=perm))
        return cast(EagerValue, getattr(backend_module, "transpose")(args[0]))
    return cast(EagerValue, host_backend.transpose(args[0], axes=perm))


@global_eager_registry.register("BroadcastTo")
@global_eager_registry.register("Expand")
def _np_broadcast_to(backend_module: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Evaluate BroadcastTo / Expand operation.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
        EagerValue: Broadcasted array result.
    """
    target_shape = kwargs.get("shape", args[1] if len(args) > 1 else ())
    if hasattr(backend_module, "broadcast_to"):
        return cast(EagerValue, getattr(backend_module, "broadcast_to")(args[0], target_shape))
    return cast(EagerValue, host_backend.broadcast_to(args[0], target_shape))
