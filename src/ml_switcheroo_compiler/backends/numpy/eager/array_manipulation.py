"""Numpy Shape Ops Extra."""

from typing import Any

# ruff: noqa: E402, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, E701, E722, F403, E711, E712, PLR0913, PLR0915
from ml_switcheroo_compiler.backends.eager_registry import numpy_eager_registry
from ml_switcheroo_compiler.backends.numpy.eager.math_scatter import _band_part


@numpy_eager_registry.register("Resize")
def _np_resize(backend_module: Any, x: Any, shape: Any, *args: Any, **kwargs: Any) -> Any:
    """Eager fallback for _np_resize.

    Args:
        backend_module (object): The backend_module parameter.
        x (object): The x parameter.
        shape (object): The shape parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    import numpy as np

    from ml_switcheroo_compiler.backends.numpy.eager.vision import resize_bilinear

    x = np.asarray(x)
    return resize_bilinear(backend_module, x, tuple(np.asarray(shape).tolist()))


@numpy_eager_registry.register("BandPart")
def _np_band_part(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _np_band_part operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    return _band_part(*args, **kwargs)


@numpy_eager_registry.register("Diag")
def _np_diag(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Eager fallback for _np_diag.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    kwargs["k"] = kwargs.pop("diagonal", kwargs.pop("k", 0))
    return backend_module.diag(*args, **kwargs)


@numpy_eager_registry.register("Unstack")
def _np_unstack(backend_module: Any, x: Any, axis: Any = 0, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _np_unstack operation.

    Args:
        backend_module (object): The backend_module parameter.
        x (object): The x parameter.
        axis (object): The axis parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    return [backend_module.squeeze(a, axis=axis) for a in backend_module.split(x, x.shape[axis], axis=axis)]


@numpy_eager_registry.register("Reshape")
def _np_reshape(backend_module: Any, x: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _np_reshape operation.

    Args:
        backend_module (object): The backend_module parameter.
        x (object): The x parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    shape = args[0] if len(args) > 0 else kwargs.get("shape", kwargs.get("newshape"))
    return backend_module.reshape(x, shape)


@numpy_eager_registry.register("Squeeze")
def _np_squeeze(backend_module: Any, x: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _np_squeeze operation.

    Args:
        backend_module (object): The backend_module parameter.
        x (object): The x parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    axis = kwargs.get("dim", args[0] if len(args) > 0 else None)
    return backend_module.squeeze(x, axis=axis)


@numpy_eager_registry.register("Transpose")
def _np_transpose(backend_module: Any, x: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _np_transpose operation.

    Args:
        backend_module (object): The backend_module parameter.
        x (object): The x parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    axes = kwargs.get("axes", kwargs.get("dims", args[0] if len(args) > 0 else None))
    return backend_module.transpose(x, axes=axes)


@numpy_eager_registry.register("Rot90")
def _np_rot90(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _np_rot90 operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    return backend_module.rot90(*args, **kwargs)


@numpy_eager_registry.register("Gather")
def gather_eager(np_mod: Any, *args: Any, **kwargs: Any) -> Any:
    """Eager gather implementation.

    Args:
        np_mod (object): The np_mod parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    t = args[0]
    dim = args[1] if len(args) > 1 else kwargs.get("dim")
    index = args[2] if len(args) > 2 else kwargs.get("index")
    if hasattr(t, "numpy"):
        t = t.numpy()
    if hasattr(index, "numpy"):
        index = index.numpy()  # type: ignore  # Justification: Polymorphic / Duck Typing for Framework Agnosticism
    return np_mod.take_along_axis(t, index, axis=dim)


@numpy_eager_registry.register("Stack")
def stack_eager(np_mod: Any, *args: Any, **kwargs: Any) -> Any:
    """Eager stack implementation.

    Args:
        np_mod (object): The np_mod parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    tensors = args[0] if len(args) > 0 else kwargs.get("tensors")
    dim = args[1] if len(args) > 1 else kwargs.get("dim", 0)
    if "axis" in kwargs:
        dim = kwargs["axis"]
    arrays = [t.numpy() if hasattr(t, "numpy") else t for t in tensors]  # type: ignore  # Justification: Polymorphic / Duck Typing for Framework Agnosticism
    return np_mod.stack(arrays, axis=dim)


@numpy_eager_registry.register("Tile")
def _np_tile(backend_module: Any, x: Any, reps: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _np_tile operation.

    Args:
        backend_module (object): The backend_module parameter.
        x (object): The x parameter.
        reps (object): The reps parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    return backend_module.tile(x, reps)


@numpy_eager_registry.register("Permute")
def _np_permute(backend_module: Any, x: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _np_permute operation.

    Args:
        backend_module (object): The backend_module parameter.
        x (object): The x parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    dims = kwargs.get("dims", args[0] if len(args) > 0 else None)
    return backend_module.transpose(x, axes=dims)


@numpy_eager_registry.register("Triu")
def _np_triu(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Eager fallback for _np_triu.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    kwargs["k"] = kwargs.pop("diagonal", 0)
    return backend_module.triu(*args, **kwargs)


@numpy_eager_registry.register("Tril")
def _np_tril(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Eager fallback for _np_tril.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    kwargs["k"] = kwargs.pop("diagonal", 0)
    return backend_module.tril(*args, **kwargs)


@numpy_eager_registry.register("ExpandDims")
def _np_expand_dims(backend_module: Any, x: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _np_expand_dims operation.

    Args:
        backend_module (object): The backend_module parameter.
        x (object): The x parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    axis = args[0] if len(args) > 0 else kwargs.get("axis")
    return backend_module.expand_dims(x, axis=axis)


@numpy_eager_registry.register("Atleast1d")
def _np_atleast_1d(backend_module: Any, x: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _np_atleast_1d operation.

    Args:
        backend_module (object): The backend_module parameter.
        x (object): The x parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    return backend_module.atleast_1d(x)


@numpy_eager_registry.register("Atleast2d")
def _np_atleast_2d(backend_module: Any, x: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _np_atleast_2d operation.

    Args:
        backend_module (object): The backend_module parameter.
        x (object): The x parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    return backend_module.atleast_2d(x)


@numpy_eager_registry.register("Atleast3d")
def _np_atleast_3d(backend_module: Any, x: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _np_atleast_3d operation.

    Args:
        backend_module (object): The backend_module parameter.
        x (object): The x parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    return backend_module.atleast_3d(x)


@numpy_eager_registry.register("Append")
def _np_append(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Eager fallback for _np_append.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    return backend_module.append(*args, **kwargs)


@numpy_eager_registry.register("ColumnStack")
def _np_column_stack(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Eager fallback for _np_column_stack.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    tup = args[0] if len(args) > 0 else kwargs.get("tup")
    return backend_module.column_stack(tup)


@numpy_eager_registry.register("Dsplit")
def _np_dsplit(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Eager fallback for _np_dsplit.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    return backend_module.dsplit(*args, **kwargs)


@numpy_eager_registry.register("Dstack")
def _np_dstack(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Eager fallback for _np_dstack.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    tup = args[0] if len(args) > 0 else kwargs.get("tup")
    return backend_module.dstack(tup)


@numpy_eager_registry.register("Hsplit")
def _np_hsplit(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Eager fallback for _np_hsplit.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    return backend_module.hsplit(*args, **kwargs)


@numpy_eager_registry.register("Hstack")
def _np_hstack(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Eager fallback for _np_hstack.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    tup = args[0] if len(args) > 0 else kwargs.get("tup")
    return backend_module.hstack(tup)


@numpy_eager_registry.register("Vsplit")
def _np_vsplit(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Eager fallback for _np_vsplit.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    return backend_module.vsplit(*args, **kwargs)


@numpy_eager_registry.register("Vstack")
def _np_vstack(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Eager fallback for _np_vstack.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    tup = args[0] if len(args) > 0 else kwargs.get("tup")
    return backend_module.vstack(tup)


@numpy_eager_registry.register("Moveaxis")
def _np_moveaxis(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Eager fallback for _np_moveaxis.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    return backend_module.moveaxis(*args, **kwargs)


@numpy_eager_registry.register("Swapaxes")
def _np_swapaxes(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Eager fallback for _np_swapaxes.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    return backend_module.swapaxes(*args, **kwargs)


@numpy_eager_registry.register("Roll")
def _np_roll(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Eager fallback for _np_roll.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    return backend_module.roll(*args, **kwargs)


@numpy_eager_registry.register("Block")
def _np_block(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Eager fallback for _np_block.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    return backend_module.block(*args, **kwargs)


@numpy_eager_registry.register("Delete")
def _np_delete(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Eager fallback for _np_delete.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    return backend_module.delete(*args, **kwargs)


@numpy_eager_registry.register("DiagIndices")
def _np_diag_indices(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Eager fallback for _np_diag_indices.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    return backend_module.diag_indices(*args, **kwargs)


@numpy_eager_registry.register("DiagIndicesFrom")
def _np_diag_indices_from(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Eager fallback for _np_diag_indices_from.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    return backend_module.diag_indices_from(*args, **kwargs)


@numpy_eager_registry.register("Diagflat")
def _np_diagflat(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Eager fallback for _np_diagflat.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    return backend_module.diagflat(*args, **kwargs)


@numpy_eager_registry.register("FillDiagonal")
def _np_fill_diagonal(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Eager fallback for _np_fill_diagonal.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    a = args[0] if len(args) > 0 else kwargs.get("a")
    val = args[1] if len(args) > 1 else kwargs.get("val")
    wrap = kwargs.get("wrap", False)
    backend_module.fill_diagonal(a, val, wrap=wrap)
    return a


@numpy_eager_registry.register("Insert")
def _np_insert(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Eager fallback for _np_insert.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    return backend_module.insert(*args, **kwargs)
