"""Operations and utilities for `ml_switcheroo_compiler.backends.eager.core_math_ops.math_manipulation`."""

from __future__ import annotations

import builtins  # noqa: F401
import types
import typing  # noqa: F401

from ml_switcheroo_compiler.backends.eager.utils import _resolve_eager_mod
from ml_switcheroo_compiler.backends.eager_registry import (
    BackendArg,
    coerce_int,
    coerce_int_list,
    coerce_shape,
    global_eager_registry,
)
from ml_switcheroo_compiler.backends.protocols import BackendModuleProtocol, adapt_backend_module


@global_eager_registry.register("AllGather")
def _all_gather(
    backend_module: BackendModuleProtocol | types.ModuleType | None,
    tensor: BackendArg,
    *args: BackendArg,
    **kwargs: BackendArg,
) -> BackendArg:
    """Execute the eager all gather operation on the active backend array.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        tensor: Input `tensor` argument (tensor).
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            BackendArg: Computed `BackendArg` output.
    """
    if adapt_backend_module(backend_module).has_symbol("stack"):
        return adapt_backend_module(backend_module).require_symbol("stack")([tensor])
    if adapt_backend_module(backend_module).has_symbol("array"):
        return adapt_backend_module(backend_module).require_symbol("array")([tensor])
    return tensor


@global_eager_registry.register("Argwhere")
def _argwhere(
    backend_module: BackendModuleProtocol | types.ModuleType | None, a: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager argwhere operation on the active backend array.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        a: Input `a` argument (a).
        **kwargs: Keyword args.

    Returns:
            BackendArg: Computed `BackendArg` output.
    """
    return adapt_backend_module(backend_module).require_symbol("argwhere")(a)


@global_eager_registry.register("Extract")
def _extract(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager extract operation on the active backend array.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            BackendArg: Computed `BackendArg` output.
    """
    func = adapt_backend_module(backend_module).get_symbol("extract", None)
    if func:
        return func(*args, **kwargs)
    (condition, arr) = (args[0], args[1])
    mod = _resolve_eager_mod(backend_module)
    return mod.extract(mod.asarray(condition), mod.asarray(arr))


@global_eager_registry.register("Pswapaxes")
def _pswapaxes(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager pswapaxes operation on the active backend array.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            BackendArg: Computed `BackendArg` output.
    """
    if adapt_backend_module(backend_module).has_symbol("lax") and adapt_backend_module(
        adapt_backend_module(backend_module).require_symbol("lax")
    ).has_symbol("pswapaxes"):
        return adapt_backend_module(backend_module).require_symbol("lax").pswapaxes(*args, **kwargs)
    return args[0] if args else None


@global_eager_registry.register("Rot90")
def _rot90(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager rot90 operation on the active backend array.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            BackendArg: Computed `BackendArg` output.
    """
    return adapt_backend_module(backend_module).require_symbol("rot90")(*args, **kwargs)


@global_eager_registry.register("Nonzero")
def _nonzero(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager nonzero operation on the active backend array.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            BackendArg: Computed `BackendArg` output.
    """
    mod = adapt_backend_module(backend_module)
    norm_args: tuple[BackendArg, ...] = (
        (mod.require_symbol("atleast_1d")(args[0]), *args[1:])
        if args and mod.has_symbol("atleast_1d") and adapt_backend_module(args[0]).get_symbol("ndim", 0) == 0
        else args
    )
    return mod.require_symbol("nonzero")(*norm_args, **kwargs)


@global_eager_registry.register("Repeat")
def _repeat(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager repeat operation on the active backend array.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            BackendArg: Computed `BackendArg` output.
    """
    return adapt_backend_module(backend_module).require_symbol("repeat")(*args, **kwargs)


@global_eager_registry.register("Tile")
def _tile(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager tile operation on the active backend array.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            BackendArg: Computed `BackendArg` output.
    """
    return adapt_backend_module(backend_module).require_symbol("tile")(*args, **kwargs)


@global_eager_registry.register("UpdateSlice")
def _updateslice(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager updateslice operation on the active backend array.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            BackendArg: Computed `BackendArg` output.

    Raises:
        RuntimeError: If invalid inputs or unsupported runtime conditions trigger `RuntimeError`.
    """
    if not adapt_backend_module(backend_module).has_symbol("array"):
        raise RuntimeError("Expected numpy-like backend")
    x = args[0]
    update = args[1] if len(args) > 1 else kwargs.get("update")
    start_indices = args[2] if len(args) > 2 else kwargs.get("start_indices")
    out = adapt_backend_module(backend_module).require_symbol("array")(x).copy()
    upd_shape = coerce_shape(adapt_backend_module(update).get_symbol("shape", ()))
    slices = tuple(slice(s, s + int(coerce_int(upd_shape[i]))) for (i, s) in enumerate(coerce_int_list(start_indices)))
    out[slices] = update
    return out


@global_eager_registry.register("Flatnonzero")
def _flatnonzero(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager flatnonzero operation on the active backend array.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            BackendArg: Computed `BackendArg` output.
    """
    func = adapt_backend_module(backend_module).get_symbol("flatnonzero", None)
    if func:
        return func(*args, **kwargs)
    return adapt_backend_module(backend_module).require_symbol("flatnonzero")(
        adapt_backend_module(backend_module).require_symbol("asarray")(args[0])
    )


@global_eager_registry.register("Fliplr")
def _fliplr(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager fliplr operation on the active backend array.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            BackendArg: Computed `BackendArg` output.
    """
    func = adapt_backend_module(backend_module).get_symbol("fliplr", None)
    if func:
        return func(*args, **kwargs)
    return adapt_backend_module(backend_module).require_symbol("fliplr")(
        adapt_backend_module(backend_module).require_symbol("asarray")(args[0])
    )


@global_eager_registry.register("Flipud")
def _flipud(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager flipud operation on the active backend array.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            BackendArg: Computed `BackendArg` output.
    """
    func = adapt_backend_module(backend_module).get_symbol("flipud", None)
    if func:
        return func(*args, **kwargs)
    return adapt_backend_module(backend_module).require_symbol("flipud")(
        adapt_backend_module(backend_module).require_symbol("asarray")(args[0])
    )


@global_eager_registry.register("Hsplit")
def _np_hsplit(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager np hsplit operation on the active backend array.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            BackendArg: Computed `BackendArg` output.
    """
    func = adapt_backend_module(backend_module).get_symbol("hsplit", None)
    if func is not None:
        return func(*args, **kwargs)
    return _resolve_eager_mod(backend_module).hsplit(args[0], args[1])


@global_eager_registry.register("ScatterApply")
def _np_scatterapply(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager np scatterapply operation on the active backend array.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            BackendArg: Computed `BackendArg` output.

    Raises:
        RuntimeError: If invalid inputs or unsupported runtime conditions trigger `RuntimeError`.
    """
    func = adapt_backend_module(backend_module).get_symbol("scatterapply", None)
    if func is not None:
        return func(*args, **kwargs)
    return args[0]


@global_eager_registry.register("ScatterNd")
def _np_scatternd(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager np scatternd operation on the active backend array.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            BackendArg: Computed `BackendArg` output.
    """
    func = adapt_backend_module(backend_module).get_symbol("scatternd", None)
    if func is not None:
        return func(*args, **kwargs)
    return args[0]


@global_eager_registry.register("TensorArrayStack")
def _np_tensorarraystack(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager np tensorarraystack operation on the active backend array.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            BackendArg: Computed `BackendArg` output.
    """
    func = adapt_backend_module(backend_module).get_symbol("tensorarraystack", None)
    if func is not None:
        return func(*args, **kwargs)
    return _resolve_eager_mod(backend_module).stack(args[0])


@global_eager_registry.register("TensorScatterUpdate")
def _np_tensorscatterupdate(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager np tensorscatterupdate operation on the active backend array.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            BackendArg: Computed `BackendArg` output.
    """
    func = adapt_backend_module(backend_module).get_symbol("tensorscatterupdate", None)
    if func is not None:
        return func(*args, **kwargs)
    return args[0]


@global_eager_registry.register("Unfold")
def _np_unfold(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager np unfold operation on the active backend array.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            BackendArg: Computed `BackendArg` output.
    """
    func = adapt_backend_module(backend_module).get_symbol("unfold", None)
    if func is not None:
        return func(*args, **kwargs)
    return args[0]


@global_eager_registry.register("Unstack")
def _np_unstack(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager np unstack operation on the active backend array.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            BackendArg: Computed `BackendArg` output.
    """
    func = adapt_backend_module(backend_module).get_symbol("unstack", None)
    if func is not None:
        return func(*args, **kwargs)
    from ml_switcheroo_compiler.backends.eager_registry import coerce_shape

    axis: int = int(coerce_int(kwargs.get("axis", 0)))
    in_sh = coerce_shape(adapt_backend_module(args[0]).require_symbol("shape"))
    return _resolve_eager_mod(backend_module).split(args[0], in_sh[axis], axis=axis)


@global_eager_registry.register("UpdateSlice")
def _np_updateslice(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager np updateslice operation on the active backend array.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            BackendArg: Computed `BackendArg` output.

    Raises:
        RuntimeError: If invalid inputs or unsupported runtime conditions trigger `RuntimeError`.
    """
    func = adapt_backend_module(backend_module).get_symbol("updateslice", None)
    if func is not None:
        return func(*args, **kwargs)
    from ml_switcheroo_compiler.backends.eager_registry import coerce_array

    target_arr = coerce_array(backend_module, args[0])
    target_arr[args[1]] = args[2]
    return target_arr


@global_eager_registry.register("Vsplit")
def _np_vsplit(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager np vsplit operation on the active backend array.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            BackendArg: Computed `BackendArg` output.
    """
    func = adapt_backend_module(backend_module).get_symbol("vsplit", None)
    if func is not None:
        return func(*args, **kwargs)
    return _resolve_eager_mod(backend_module).vsplit(args[0], args[1])


@global_eager_registry.register("Transpose")
def _np_transpose(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager np transpose operation on the active backend array.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
        BackendArg: Transposed array result.
    """
    perm = kwargs.get("permutation", kwargs.get("axes", None))
    if adapt_backend_module(backend_module).has_symbol("transpose"):
        if perm is not None:
            return adapt_backend_module(backend_module).require_symbol("transpose")(args[0], axes=perm)
        return adapt_backend_module(backend_module).require_symbol("transpose")(args[0])
    return _resolve_eager_mod(backend_module).transpose(args[0], axes=perm)


@global_eager_registry.register("BroadcastTo")
@global_eager_registry.register("Expand")
def _np_broadcast_to(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Evaluate BroadcastTo / Expand operation.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
        BackendArg: Broadcasted array result.
    """
    target_shape = kwargs.get("shape", args[1] if len(args) > 1 else ())
    if adapt_backend_module(backend_module).has_symbol("broadcast_to"):
        return adapt_backend_module(backend_module).require_symbol("broadcast_to")(args[0], target_shape)
    return _resolve_eager_mod(backend_module).broadcast_to(args[0], target_shape)
