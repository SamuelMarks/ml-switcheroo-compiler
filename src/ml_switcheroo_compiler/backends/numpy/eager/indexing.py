"""Module indexing.py."""

# ruff: noqa: E402, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, E701, E722, F403, E711, E712, PLR0913, PLR0915
# ruff: noqa: E402, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, E701, E722, F403, E711, E712, PLR0913, PLR0915
"""Core abstractions and logic definitions for indexing.py."""

from dataclasses import dataclass


@dataclass
class IndexTarget:
    """Index target container."""

    operand: object
    update: object
    index: object


import numpy as np

from ml_switcheroo_compiler.backends.eager_registry import numpy_eager_registry

from ._indexing_parsing_utils import _safe_parse_key

"""Core abstractions and logic definitions for indexing.py."""


@dataclass
class IndexingContext:
    """Configuration class for indexing context."""

    axis: int = 0
    start_index: object = None
    limit_index: object = None
    slice_size: object = None
    stride: int = 1
    keepdims: bool = True


def _dynamic_update_slice(x: object, update: object, start_indices: object) -> object:
    """Evaluate _dynamic_update_slice operation.

    Args:
        x (object): The x parameter.
        update (object): The update parameter.
        start_indices (object): The start_indices parameter.

    Returns:
            tuple[int, ...]: Result.
    """
    out: object = np.copy(x)

    def _to_int(v: object) -> int:
        """Evaluate _to_int operation.

        Args:
            v (object): The v parameter.

        Returns:
            int: Result.
        """
        if hasattr(v, "data"):
            v: object = v.data
        if hasattr(v, "item"):
            return int(v.item())
        return int(v)

    slices: object = tuple(slice(_to_int(start), _to_int(start) + size) for (start, size) in zip(start_indices, update.shape))
    out[slices] = update
    return out


@numpy_eager_registry.register("DynamicUpdateSlice")
def _np_dynamic_update_slice(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate _np_dynamic_update_slice operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    return _dynamic_update_slice(*args, **kwargs)


@numpy_eager_registry.register("Unstack")
def _np_unstack(backend_module: object, x: object, axis: object = 0, *args: object, **kwargs: object) -> object:
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


@numpy_eager_registry.register("DynamicSlice")
def _np_dynamic_slice(backend_module: object, x: object, start_indices: object, slice_sizes: object) -> object:
    """Evaluate _np_dynamic_slice operation.

    Args:
        backend_module (object): The backend_module parameter.
        x (object): The x parameter.
        start_indices (object): The start_indices parameter.
        slice_sizes (object): The slice_sizes parameter.

    Returns:
            tuple[int, ...]: Result.
    """
    slices: object = tuple(slice(start, start + size) for (start, size) in zip(start_indices, slice_sizes))
    return x[slices]


@numpy_eager_registry.register("DynamicSliceInDim")
def _np_dynamic_slice_in_dim(backend_module: object, operand: object, context: IndexingContext, *args: object, **kwargs: object) -> object:
    """Evaluate _np_dynamic_slice_in_dim operation.

    Args:
        backend_module (object): The backend_module parameter.
        operand (object): The operand parameter.
        context (IndexingContext): The context parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    operand: object = np.asarray(operand)
    start_index: object = np.asarray(context.start_index).item()
    sl: object = [slice(None)] * operand.ndim
    sl[context.axis] = slice(start_index, start_index + context.slice_size)
    return operand[tuple(sl)]


@numpy_eager_registry.register("DynamicUpdateSliceInDim")
def _np_dynamic_update_slice_in_dim(backend_module: object, operand: object, update: object, context: IndexingContext, *args: object, **kwargs: object) -> object:
    """Evaluate _np_dynamic_update_slice_in_dim operation.

    Args:
        backend_module (object): The backend_module parameter.
        operand (object): The operand parameter.
        update (object): The update parameter.
        context (IndexingContext): The context parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    operand: object = np.copy(np.asarray(operand))
    start_index: object = np.asarray(context.start_index).item()
    slice_size: object = np.asarray(update).shape[context.axis]
    sl: object = [slice(None)] * operand.ndim
    sl[context.axis] = slice(start_index, start_index + slice_size)
    operand[tuple(sl)] = update
    return operand


@numpy_eager_registry.register("DynamicIndexInDim")
def _np_dynamic_index_in_dim(backend_module: object, operand: object, index: object, context: IndexingContext, *args: object, **kwargs: object) -> object:
    """Evaluate _np_dynamic_index_in_dim operation.

    Args:
        backend_module (object): The backend_module parameter.
        operand (object): The operand parameter.
        index (object): The index parameter.
        context (IndexingContext): The context parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    operand: object = np.asarray(operand)
    idx: object = np.asarray(index).item()
    if context.keepdims:
        sl: object = [slice(None)] * operand.ndim
        sl[context.axis] = slice(idx, idx + 1)
        return operand[tuple(sl)]
    else:
        sl: object = [slice(None)] * operand.ndim
        sl[context.axis] = idx
        return operand[tuple(sl)]


@numpy_eager_registry.register("DynamicUpdateIndexInDim")
def _np_dynamic_update_index_in_dim(backend_module: object, target: IndexTarget, context: IndexingContext, *args: object, **kwargs: object) -> object:
    """Evaluate _np_dynamic_update_index_in_dim operation.

    Args:
        backend_module (object): The backend_module parameter.
        target (IndexTarget): The target parameter.
        context (IndexingContext): The context parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    operand, update, index = target.operand, target.update, target.index
    operand: object = np.copy(np.asarray(operand))
    idx: object = np.asarray(index).item()
    sl: object = [slice(None)] * operand.ndim
    sl[context.axis] = idx
    operand[tuple(sl)] = update
    return operand


@numpy_eager_registry.register("SliceInDim")
def _np_slice_in_dim(backend_module: object, operand: object, context: IndexingContext, *args: object, **kwargs: object) -> object:
    """Evaluate _np_slice_in_dim operation.

    Args:
        backend_module (object): The backend_module parameter.
        operand (object): The operand parameter.
        context (IndexingContext): The context parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    operand: object = np.asarray(operand)
    sl: object = [slice(None)] * operand.ndim
    sl[context.axis] = slice(context.start_index, context.limit_index, context.stride)
    return operand[tuple(sl)]


@numpy_eager_registry.register("Slice")
def _np_slice(backend_module: object, x: object, context: IndexingContext) -> object:
    """Evaluate _np_slice operation.

    Args:
        backend_module (object): The backend_module parameter.
        x (object): The x parameter.
        context (IndexingContext): The context parameter.

    Returns:
            tuple[int, ...]: Result.
    """
    sl: object = [slice(None)] * x.ndim
    sl[context.axis] = slice(context.start_index, context.limit_index, context.stride)
    return x[tuple(sl)]


@numpy_eager_registry.register("GetItem")
def _np_getitem(backend_module: object, x: object, key: str) -> object:
    """Evaluate _np_getitem operation.

    Args:
        backend_module (object): The backend_module parameter.
        x (object): The x parameter.
        key (str): The key parameter.

    Returns:
            tuple[int, ...]: Result.
    """
    parsed_key: object = _safe_parse_key(key)
    return x[parsed_key]


@numpy_eager_registry.register("SetItem")
def _np_setitem(backend_module: object, x: object, value: object, key: str) -> object:
    """Evaluate _np_setitem operation.

    Args:
        backend_module (object): The backend_module parameter.
        x (object): The x parameter.
        value (object): The value parameter.
        key (str): The key parameter.

    Returns:
            tuple[int, ...]: Result.
    """
    parsed_key: object = _safe_parse_key(key)
    out: object = np.copy(np.asarray(x))
    out[parsed_key] = np.asarray(value)
    return out


@numpy_eager_registry.register("IndexInDim")
def _eager_indexindim(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate _eager_indexindim operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    import numpy as np

    x, idx, dim = args[0], args[1], args[2]
    return np.take(x, idx, axis=dim)


@numpy_eager_registry.register("Gather")
def gather_eager(np_mod: object, *args: object, **kwargs: object) -> object:
    """gather_eager function.

    Args:
        np_mod (object): The np_mod parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    t: object = args[0]
    dim: object = args[1] if len(args) > 1 else kwargs.get("dim")
    index: object = args[2] if len(args) > 2 else kwargs.get("index")
    if hasattr(t, "numpy"):
        t: object = t.numpy()
    if hasattr(index, "numpy"):
        index: object = index.numpy()
    return np_mod.take_along_axis(t, index, axis=dim)


@numpy_eager_registry.register("Stack")
def stack_eager(np_mod: object, *args: object, **kwargs: object) -> object:
    """stack_eager function.

    Args:
        np_mod (object): The np_mod parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    tensors: object = args[0] if len(args) > 0 else kwargs.get("tensors")
    dim: object = args[1] if len(args) > 1 else kwargs.get("dim", 0)
    if "axis" in kwargs:
        dim: object = kwargs["axis"]
    arrays: object = [t.numpy() if hasattr(t, "numpy") else t for t in tensors]
    return np_mod.stack(arrays, axis=dim)
