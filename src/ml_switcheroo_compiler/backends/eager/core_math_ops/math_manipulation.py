"""Implementation of math manipulation operations and utilities for the backends subsystem (`ml_switcheroo_compiler.backends.eager.core_math_ops.math_manipulation`)."""

from __future__ import annotations

import builtins  # noqa: F401
import typing  # noqa: F401

from ml_switcheroo_compiler.backends.eager.utils import _resolve_eager_mod
from ml_switcheroo_compiler.backends.eager_registry import (
    EagerValue,
    ensure_eager_tensor,
    ensure_int,
    ensure_int_list,
    ensure_shape,
    global_eager_registry,
)
from ml_switcheroo_compiler.backends.protocols import BackendModuleProtocol, get_backend_member, has_backend_member


@global_eager_registry.register("AllGather")
def _all_gather(backend_module: BackendModuleProtocol, tensor: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager all gather operation on the active backend array.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        tensor: Input `tensor` argument (tensor).
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Computed `EagerValue` output.
    """
    if has_backend_member(backend_module, "stack"):
        return ensure_eager_tensor(get_backend_member(backend_module, "stack")([tensor]))
    if has_backend_member(backend_module, "array"):
        return ensure_eager_tensor(get_backend_member(backend_module, "array")([tensor]))
    return tensor


@global_eager_registry.register("Argwhere")
def _argwhere(backend_module: BackendModuleProtocol, a: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager argwhere operation on the active backend array.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        a: Input `a` argument (a).
        **kwargs: Keyword args.

    Returns:
            EagerValue: Computed `EagerValue` output.
    """
    return ensure_eager_tensor(get_backend_member(backend_module, "argwhere")(a))


@global_eager_registry.register("Extract")
def _extract(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager extract operation on the active backend array.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Computed `EagerValue` output.
    """
    func = get_backend_member(backend_module, "extract", None)
    if func:
        return ensure_eager_tensor(func(*args, **kwargs))
    (condition, arr) = (args[0], args[1])
    mod = _resolve_eager_mod(backend_module)
    return mod.extract(mod.asarray(condition), mod.asarray(arr))


@global_eager_registry.register("Pswapaxes")
def _pswapaxes(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager pswapaxes operation on the active backend array.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Computed `EagerValue` output.
    """
    if has_backend_member(backend_module, "lax") and has_backend_member(get_backend_member(backend_module, "lax"), "pswapaxes"):
        return ensure_eager_tensor(get_backend_member(backend_module, "lax").pswapaxes(*args, **kwargs))
    return args[0] if args else None


@global_eager_registry.register("Rot90")
def _rot90(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager rot90 operation on the active backend array.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Computed `EagerValue` output.
    """
    return ensure_eager_tensor(get_backend_member(backend_module, "rot90")(*args, **kwargs))


@global_eager_registry.register("Nonzero")
def _nonzero(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager nonzero operation on the active backend array.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Computed `EagerValue` output.
    """
    return ensure_eager_tensor(get_backend_member(backend_module, "nonzero")(*args, **kwargs))


@global_eager_registry.register("Repeat")
def _repeat(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager repeat operation on the active backend array.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Computed `EagerValue` output.
    """
    return ensure_eager_tensor(get_backend_member(backend_module, "repeat")(*args, **kwargs))


@global_eager_registry.register("Tile")
def _tile(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager tile operation on the active backend array.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Computed `EagerValue` output.
    """
    return ensure_eager_tensor(get_backend_member(backend_module, "tile")(*args, **kwargs))


@global_eager_registry.register("UpdateSlice")
def _updateslice(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager updateslice operation on the active backend array.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Computed `EagerValue` output.

    Raises:
        RuntimeError: If invalid inputs or unsupported runtime conditions trigger `RuntimeError`.
    """
    if not has_backend_member(backend_module, "array"):
        raise RuntimeError("Expected numpy-like backend")
    x = args[0]
    update = args[1] if len(args) > 1 else kwargs.get("update")
    start_indices = args[2] if len(args) > 2 else kwargs.get("start_indices")
    out = get_backend_member(backend_module, "array")(x).copy()
    upd_shape = ensure_shape(get_backend_member(update, "shape", ()))
    slices = tuple(slice(s, s + int(ensure_int(upd_shape[i]))) for (i, s) in enumerate(ensure_int_list(start_indices)))
    out[slices] = update
    return ensure_eager_tensor(out)


@global_eager_registry.register("Flatnonzero")
def _flatnonzero(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager flatnonzero operation on the active backend array.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Computed `EagerValue` output.
    """
    func = get_backend_member(backend_module, "flatnonzero", None)
    if func:
        return ensure_eager_tensor(func(*args, **kwargs))
    return ensure_eager_tensor(get_backend_member(backend_module, "flatnonzero")(get_backend_member(backend_module, "asarray")(args[0])))


@global_eager_registry.register("Fliplr")
def _fliplr(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager fliplr operation on the active backend array.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Computed `EagerValue` output.
    """
    func = get_backend_member(backend_module, "fliplr", None)
    if func:
        return ensure_eager_tensor(func(*args, **kwargs))
    return ensure_eager_tensor(get_backend_member(backend_module, "fliplr")(get_backend_member(backend_module, "asarray")(args[0])))


@global_eager_registry.register("Flipud")
def _flipud(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager flipud operation on the active backend array.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Computed `EagerValue` output.
    """
    func = get_backend_member(backend_module, "flipud", None)
    if func:
        return ensure_eager_tensor(func(*args, **kwargs))
    return ensure_eager_tensor(get_backend_member(backend_module, "flipud")(get_backend_member(backend_module, "asarray")(args[0])))


@global_eager_registry.register("Hsplit")
def _np_hsplit(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager np hsplit operation on the active backend array.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Computed `EagerValue` output.
    """
    func = get_backend_member(backend_module, "hsplit", None)
    if func is not None:
        return ensure_eager_tensor(func(*args, **kwargs))
    return _resolve_eager_mod(backend_module).hsplit(args[0], args[1])


@global_eager_registry.register("ScatterApply")
def _np_scatterapply(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager np scatterapply operation on the active backend array.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Computed `EagerValue` output.

    Raises:
        RuntimeError: If invalid inputs or unsupported runtime conditions trigger `RuntimeError`.
    """
    func = get_backend_member(backend_module, "scatterapply", None)
    if func is not None:
        return ensure_eager_tensor(func(*args, **kwargs))
    return args[0]


@global_eager_registry.register("ScatterNd")
def _np_scatternd(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager np scatternd operation on the active backend array.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Computed `EagerValue` output.
    """
    func = get_backend_member(backend_module, "scatternd", None)
    if func is not None:
        return ensure_eager_tensor(func(*args, **kwargs))
    return args[0]


@global_eager_registry.register("TensorArrayStack")
def _np_tensorarraystack(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager np tensorarraystack operation on the active backend array.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Computed `EagerValue` output.
    """
    func = get_backend_member(backend_module, "tensorarraystack", None)
    if func is not None:
        return ensure_eager_tensor(func(*args, **kwargs))
    return _resolve_eager_mod(backend_module).stack(args[0])


@global_eager_registry.register("TensorScatterUpdate")
def _np_tensorscatterupdate(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager np tensorscatterupdate operation on the active backend array.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Computed `EagerValue` output.
    """
    func = get_backend_member(backend_module, "tensorscatterupdate", None)
    if func is not None:
        return ensure_eager_tensor(func(*args, **kwargs))
    return args[0]


@global_eager_registry.register("Unfold")
def _np_unfold(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager np unfold operation on the active backend array.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Computed `EagerValue` output.
    """
    func = get_backend_member(backend_module, "unfold", None)
    if func is not None:
        return ensure_eager_tensor(func(*args, **kwargs))
    return args[0]


@global_eager_registry.register("Unstack")
def _np_unstack(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager np unstack operation on the active backend array.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Computed `EagerValue` output.
    """
    func = get_backend_member(backend_module, "unstack", None)
    if func is not None:
        return ensure_eager_tensor(func(*args, **kwargs))
    axis: int = int(ensure_int(kwargs.get("axis", 0)))
    return _resolve_eager_mod(backend_module).split(args[0], ensure_eager_tensor(args[0]).shape[axis], axis=axis)


@global_eager_registry.register("UpdateSlice")
def _np_updateslice(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager np updateslice operation on the active backend array.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Computed `EagerValue` output.

    Raises:
        RuntimeError: If invalid inputs or unsupported runtime conditions trigger `RuntimeError`.
    """
    func = get_backend_member(backend_module, "updateslice", None)
    if func is not None:
        return ensure_eager_tensor(func(*args, **kwargs))
    ensure_eager_tensor(args[0])[args[1]] = args[2]
    return args[0]


@global_eager_registry.register("Vsplit")
def _np_vsplit(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager np vsplit operation on the active backend array.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Computed `EagerValue` output.
    """
    func = get_backend_member(backend_module, "vsplit", None)
    if func is not None:
        return ensure_eager_tensor(func(*args, **kwargs))
    return _resolve_eager_mod(backend_module).vsplit(args[0], args[1])


@global_eager_registry.register("Transpose")
def _np_transpose(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager np transpose operation on the active backend array.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
        EagerValue: Transposed array result.
    """
    perm = kwargs.get("permutation", kwargs.get("axes", None))
    if has_backend_member(backend_module, "transpose"):
        if perm is not None:
            return ensure_eager_tensor(get_backend_member(backend_module, "transpose")(args[0], axes=perm))
        return ensure_eager_tensor(get_backend_member(backend_module, "transpose")(args[0]))
    return _resolve_eager_mod(backend_module).transpose(args[0], axes=perm)


@global_eager_registry.register("BroadcastTo")
@global_eager_registry.register("Expand")
def _np_broadcast_to(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Evaluate BroadcastTo / Expand operation.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
        EagerValue: Broadcasted array result.
    """
    target_shape = kwargs.get("shape", args[1] if len(args) > 1 else ())
    if has_backend_member(backend_module, "broadcast_to"):
        return ensure_eager_tensor(get_backend_member(backend_module, "broadcast_to")(args[0], target_shape))
    return _resolve_eager_mod(backend_module).broadcast_to(args[0], target_shape)
