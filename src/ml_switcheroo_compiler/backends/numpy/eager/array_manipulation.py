"""Numpy Shape Ops Extra."""

from __future__ import annotations

import threading
from types import ModuleType
from typing import Callable, cast

import numpy as np

from ml_switcheroo_compiler.backends.eager_registry import EagerTensorProtocol, EagerValue, numpy_eager_registry
from ml_switcheroo_compiler.backends.numpy.eager.math_scatter import _band_part


@numpy_eager_registry.register("Resize")
def _np_resize(backend_module: ModuleType, x: EagerValue, shape: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Eager fallback for _np_resize.

    Args:
        backend_module (ModuleType): The backend_module parameter.
        x (EagerValue): The x parameter.
        shape (EagerValue): The shape parameter.
        *args (EagerValue): Positional args.
        **kwargs (EagerValue): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    from ml_switcheroo_compiler.backends.numpy.eager.vision import resize_bilinear

    x = np.asarray(x)
    return resize_bilinear(backend_module, x, tuple(np.asarray(shape).tolist()))


@numpy_eager_registry.register("BandPart")
def _np_band_part(backend_module: ModuleType, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager np band part operation on the active backend array.

    Args:
        backend_module (ModuleType): The backend_module parameter.
        *args (EagerValue): Positional args.
        **kwargs (EagerValue): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    return cast(EagerValue, cast("Callable[..., EagerTensorProtocol]", _band_part)(*args, **kwargs))


@numpy_eager_registry.register("Diag")
def _np_diag(backend_module: ModuleType, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Eager fallback for _np_diag.

    Args:
        backend_module (ModuleType): The backend_module parameter.
        *args (EagerValue): Positional args.
        **kwargs (EagerValue): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    kwargs["k"] = kwargs.pop("diagonal", kwargs.pop("k", 0))
    return cast(EagerValue, backend_module.diag(*args, **kwargs))


@numpy_eager_registry.register("Unstack")
def _np_unstack(backend_module: ModuleType, x: EagerValue, axis: int = 0, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager np unstack operation on the active backend array.

    Args:
        backend_module (ModuleType): The backend_module parameter.
        x (EagerValue): The x parameter.
        axis (int): The axis parameter.
        *args (EagerValue): Positional args.
        **kwargs (EagerValue): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    return [backend_module.squeeze(a, axis=axis) for a in backend_module.split(x, getattr(x, "shape")[axis], axis=axis)]


@numpy_eager_registry.register("Reshape")
def _np_reshape(backend_module: ModuleType, x: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager np reshape operation on the active backend array.

    Args:
        backend_module (ModuleType): The backend_module parameter.
        x (EagerValue): The x parameter.
        *args (EagerValue): Positional args.
        **kwargs (EagerValue): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    shape: list[EagerValue] | None = cast("list[EagerValue] | None", args[0] if len(args) > 0 else kwargs.get("shape", kwargs.get("newshape")))
    return cast(EagerValue, backend_module.reshape(x, shape))


@numpy_eager_registry.register("Squeeze")
def _np_squeeze(backend_module: ModuleType, x: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager np squeeze operation on the active backend array.

    Args:
        backend_module (ModuleType): The backend_module parameter.
        x (EagerValue): The x parameter.
        *args (EagerValue): Positional args.
        **kwargs (EagerValue): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    axis: int | None = cast("int | None", kwargs.get("axis", args[0] if len(args) > 0 else None))
    return cast(EagerValue, backend_module.squeeze(x, axis=axis))


@numpy_eager_registry.register("Transpose")
def _np_transpose(backend_module: ModuleType, x: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager np transpose operation on the active backend array.

    Args:
        backend_module (ModuleType): The backend_module parameter.
        x (EagerValue): The x parameter.
        *args (EagerValue): Positional args.
        **kwargs (EagerValue): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    axes: list[EagerValue] | None = cast("list[EagerValue] | None", kwargs.get("axes", args[0] if len(args) > 0 else None))
    return cast(EagerValue, backend_module.transpose(x, axes=axes))


@numpy_eager_registry.register("Rot90")
def _np_rot90(backend_module: ModuleType, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager np rot90 operation on the active backend array.

    Args:
        backend_module (ModuleType): The backend_module parameter.
        *args (EagerValue): Positional args.
        **kwargs (EagerValue): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    return cast(EagerValue, backend_module.rot90(*args, **kwargs))


@numpy_eager_registry.register("Gather")
def gather_eager(np_mod: ModuleType, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Eager gather implementation.

    Args:
        np_mod (ModuleType): The np_mod parameter.
        *args (EagerValue): Positional args.
        **kwargs (EagerValue): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    t: threading.Thread = cast(threading.Thread, args[0])
    axis: int | None = cast("int | None", args[1] if len(args) > 1 else kwargs.get("axis"))
    index: np.ndarray[tuple[int, ...], np.dtype[np.generic]] | None = cast("np.ndarray[tuple[int, ...], np.dtype[np.generic]] | None", args[2] if len(args) > 2 else kwargs.get("index"))
    if hasattr(t, "numpy"):
        t = t.numpy()
    if hasattr(index, "numpy"):
        index = getattr(index, "numpy")()
    return cast(EagerValue, np_mod.take_along_axis(t, index, axis=axis))


@numpy_eager_registry.register("Stack")
def stack_eager(np_mod: ModuleType, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Eager stack implementation.

    Args:
        np_mod (ModuleType): The np_mod parameter.
        *args (EagerValue): Positional args.
        **kwargs (EagerValue): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    tensors: list[EagerValue] = cast(list[EagerValue], args[0] if len(args) > 0 else (kwargs.get("tensors") or []))
    axis: int = cast(int, args[1] if len(args) > 1 else kwargs.get("axis", 0))
    arrays: list[EagerValue] = [getattr(t, "numpy")() if hasattr(t, "numpy") else t for t in tensors]
    return cast(EagerValue, np_mod.stack(arrays, axis=axis))


@numpy_eager_registry.register("Tile")
def _np_tile(backend_module: ModuleType, x: EagerValue, reps: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager np tile operation on the active backend array.

    Args:
        backend_module (ModuleType): The backend_module parameter.
        x (EagerValue): The x parameter.
        reps (EagerValue): The reps parameter.
        *args (EagerValue): Positional args.
        **kwargs (EagerValue): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    return cast(EagerValue, backend_module.tile(x, reps))


@numpy_eager_registry.register("Permute")
def _np_permute(backend_module: ModuleType, x: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager np permute operation on the active backend array.

    Args:
        backend_module (ModuleType): The backend_module parameter.
        x (EagerValue): The x parameter.
        *args (EagerValue): Positional args.
        **kwargs (EagerValue): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    dims = kwargs.get("dims", args[0] if len(args) > 0 else None)
    return cast(EagerValue, backend_module.transpose(x, axes=dims))


@numpy_eager_registry.register("Triu")
def _np_triu(backend_module: ModuleType, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Eager fallback for _np_triu.

    Args:
        backend_module (ModuleType): The backend_module parameter.
        *args (EagerValue): Positional args.
        **kwargs (EagerValue): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    kwargs["k"] = kwargs.pop("diagonal", 0)
    return cast(EagerValue, backend_module.triu(*args, **kwargs))


@numpy_eager_registry.register("Tril")
def _np_tril(backend_module: ModuleType, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Eager fallback for _np_tril.

    Args:
        backend_module (ModuleType): The backend_module parameter.
        *args (EagerValue): Positional args.
        **kwargs (EagerValue): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    kwargs["k"] = kwargs.pop("diagonal", 0)
    return cast(EagerValue, backend_module.tril(*args, **kwargs))


@numpy_eager_registry.register("ExpandDims")
def _np_expand_dims(backend_module: ModuleType, x: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager np expand dims operation on the active backend array.

    Args:
        backend_module (ModuleType): The backend_module parameter.
        x (EagerValue): The x parameter.
        *args (EagerValue): Positional args.
        **kwargs (EagerValue): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    axis: int | None = cast("int | None", args[0] if len(args) > 0 else kwargs.get("axis"))
    return cast(EagerValue, backend_module.expand_dims(x, axis=axis))


@numpy_eager_registry.register("Atleast1d")
def _np_atleast_1d(backend_module: ModuleType, x: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager np atleast 1d operation on the active backend array.

    Args:
        backend_module (ModuleType): The backend_module parameter.
        x (EagerValue): The x parameter.
        *args (EagerValue): Positional args.
        **kwargs (EagerValue): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    return cast(EagerValue, backend_module.atleast_1d(x))


@numpy_eager_registry.register("Atleast2d")
def _np_atleast_2d(backend_module: ModuleType, x: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager np atleast 2d operation on the active backend array.

    Args:
        backend_module (ModuleType): The backend_module parameter.
        x (EagerValue): The x parameter.
        *args (EagerValue): Positional args.
        **kwargs (EagerValue): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    return cast(EagerValue, backend_module.atleast_2d(x))


@numpy_eager_registry.register("Atleast3d")
def _np_atleast_3d(backend_module: ModuleType, x: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager np atleast 3d operation on the active backend array.

    Args:
        backend_module (ModuleType): The backend_module parameter.
        x (EagerValue): The x parameter.
        *args (EagerValue): Positional args.
        **kwargs (EagerValue): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    return cast(EagerValue, backend_module.atleast_3d(x))


@numpy_eager_registry.register("Append")
def _np_append(backend_module: ModuleType, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Eager fallback for _np_append.

    Args:
        backend_module (ModuleType): The backend_module parameter.
        *args (EagerValue): Positional args.
        **kwargs (EagerValue): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    return cast(EagerValue, backend_module.append(*args, **kwargs))


@numpy_eager_registry.register("ColumnStack")
def _np_column_stack(backend_module: ModuleType, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Eager fallback for _np_column_stack.

    Args:
        backend_module (ModuleType): The backend_module parameter.
        *args (EagerValue): Positional args.
        **kwargs (EagerValue): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    tup: tuple[EagerValue, ...] | None = cast("tuple[EagerValue, ...] | None", args[0] if len(args) > 0 else kwargs.get("tup"))
    return cast(EagerValue, backend_module.column_stack(tup))


@numpy_eager_registry.register("Dsplit")
def _np_dsplit(backend_module: ModuleType, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Eager fallback for _np_dsplit.

    Args:
        backend_module (ModuleType): The backend_module parameter.
        *args (EagerValue): Positional args.
        **kwargs (EagerValue): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    return cast(EagerValue, backend_module.dsplit(*args, **kwargs))


@numpy_eager_registry.register("Dstack")
def _np_dstack(backend_module: ModuleType, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Eager fallback for _np_dstack.

    Args:
        backend_module (ModuleType): The backend_module parameter.
        *args (EagerValue): Positional args.
        **kwargs (EagerValue): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    tup: tuple[EagerValue, ...] | None = cast("tuple[EagerValue, ...] | None", args[0] if len(args) > 0 else kwargs.get("tup"))
    return cast(EagerValue, backend_module.dstack(tup))


@numpy_eager_registry.register("Hsplit")
def _np_hsplit(backend_module: ModuleType, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Eager fallback for _np_hsplit.

    Args:
        backend_module (ModuleType): The backend_module parameter.
        *args (EagerValue): Positional args.
        **kwargs (EagerValue): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    return cast(EagerValue, backend_module.hsplit(*args, **kwargs))


@numpy_eager_registry.register("Hstack")
def _np_hstack(backend_module: ModuleType, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Eager fallback for _np_hstack.

    Args:
        backend_module (ModuleType): The backend_module parameter.
        *args (EagerValue): Positional args.
        **kwargs (EagerValue): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    tup: tuple[EagerValue, ...] | None = cast("tuple[EagerValue, ...] | None", args[0] if len(args) > 0 else kwargs.get("tup"))
    return cast(EagerValue, backend_module.hstack(tup))


@numpy_eager_registry.register("Vsplit")
def _np_vsplit(backend_module: ModuleType, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Eager fallback for _np_vsplit.

    Args:
        backend_module (ModuleType): The backend_module parameter.
        *args (EagerValue): Positional args.
        **kwargs (EagerValue): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    return cast(EagerValue, backend_module.vsplit(*args, **kwargs))


@numpy_eager_registry.register("Vstack")
def _np_vstack(backend_module: ModuleType, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Eager fallback for _np_vstack.

    Args:
        backend_module (ModuleType): The backend_module parameter.
        *args (EagerValue): Positional args.
        **kwargs (EagerValue): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    tup: tuple[EagerValue, ...] | None = cast("tuple[EagerValue, ...] | None", args[0] if len(args) > 0 else kwargs.get("tup"))
    return cast(EagerValue, backend_module.vstack(tup))


@numpy_eager_registry.register("Moveaxis")
def _np_moveaxis(backend_module: ModuleType, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Eager fallback for _np_moveaxis.

    Args:
        backend_module (ModuleType): The backend_module parameter.
        *args (EagerValue): Positional args.
        **kwargs (EagerValue): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    return cast(EagerValue, backend_module.moveaxis(*args, **kwargs))


@numpy_eager_registry.register("Swapaxes")
def _np_swapaxes(backend_module: ModuleType, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Eager fallback for _np_swapaxes.

    Args:
        backend_module (ModuleType): The backend_module parameter.
        *args (EagerValue): Positional args.
        **kwargs (EagerValue): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    return cast(EagerValue, backend_module.swapaxes(*args, **kwargs))


@numpy_eager_registry.register("Roll")
def _np_roll(backend_module: ModuleType, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Eager fallback for _np_roll.

    Args:
        backend_module (ModuleType): The backend_module parameter.
        *args (EagerValue): Positional args.
        **kwargs (EagerValue): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    return cast(EagerValue, backend_module.roll(*args, **kwargs))


@numpy_eager_registry.register("Block")
def _np_block(backend_module: ModuleType, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Eager fallback for _np_block.

    Args:
        backend_module (ModuleType): The backend_module parameter.
        *args (EagerValue): Positional args.
        **kwargs (EagerValue): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    return cast(EagerValue, backend_module.block(*args, **kwargs))


@numpy_eager_registry.register("Delete")
def _np_delete(backend_module: ModuleType, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Eager fallback for _np_delete.

    Args:
        backend_module (ModuleType): The backend_module parameter.
        *args (EagerValue): Positional args.
        **kwargs (EagerValue): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    return cast(EagerValue, backend_module.delete(*args, **kwargs))


@numpy_eager_registry.register("DiagIndices")
def _np_diag_indices(backend_module: ModuleType, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Eager fallback for _np_diag_indices.

    Args:
        backend_module (ModuleType): The backend_module parameter.
        *args (EagerValue): Positional args.
        **kwargs (EagerValue): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    return cast(EagerValue, backend_module.diag_indices(*args, **kwargs))


@numpy_eager_registry.register("DiagIndicesFrom")
def _np_diag_indices_from(backend_module: ModuleType, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Eager fallback for _np_diag_indices_from.

    Args:
        backend_module (ModuleType): The backend_module parameter.
        *args (EagerValue): Positional args.
        **kwargs (EagerValue): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    return cast(EagerValue, backend_module.diag_indices_from(*args, **kwargs))


@numpy_eager_registry.register("Diagflat")
def _np_diagflat(backend_module: ModuleType, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Eager fallback for _np_diagflat.

    Args:
        backend_module (ModuleType): The backend_module parameter.
        *args (EagerValue): Positional args.
        **kwargs (EagerValue): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    return cast(EagerValue, backend_module.diagflat(*args, **kwargs))


@numpy_eager_registry.register("FillDiagonal")
def _np_fill_diagonal(backend_module: ModuleType, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Eager fallback for _np_fill_diagonal.

    Args:
        backend_module (ModuleType): The backend_module parameter.
        *args (EagerValue): Positional args.
        **kwargs (EagerValue): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    a: np.ndarray[tuple[int, ...], np.dtype[np.generic]] | None = cast("np.ndarray[tuple[int, ...], np.dtype[np.generic]] | None", args[0] if len(args) > 0 else kwargs.get("a"))
    val: np.ndarray[tuple[int, ...], np.dtype[np.generic]] | None = cast("np.ndarray[tuple[int, ...], np.dtype[np.generic]] | None", args[1] if len(args) > 1 else kwargs.get("val"))
    wrap: bool = cast(bool, kwargs.get("wrap", False))
    backend_module.fill_diagonal(a, val, wrap=wrap)
    return a


@numpy_eager_registry.register("Insert")
def _np_insert(backend_module: ModuleType, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Eager fallback for _np_insert.

    Args:
        backend_module (ModuleType): The backend_module parameter.
        *args (EagerValue): Positional args.
        **kwargs (EagerValue): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    return cast(EagerValue, backend_module.insert(*args, **kwargs))
