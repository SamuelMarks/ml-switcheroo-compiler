"""Module indexing.py."""

# ruff: noqa: E402, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, E701, E722, F403, E711, E712, PLR0913, PLR0915
# ruff: noqa: E402, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, E701, E722, F403, E711, E712, PLR0913, PLR0915
"""Core abstractions and logic definitions for indexing.py."""

from dataclasses import dataclass


@dataclass
class IndexTarget:
    """Index target container."""


import threading

import numpy as np

from ml_switcheroo_compiler.backends.eager_registry import numpy_eager_registry

from ._indexing_parsing_utils import _safe_parse_key

"""Core abstractions and logic definitions for indexing.py."""


@dataclass
class IndexingContext:
    """Configuration class for indexing context."""

    axis: int = 0
    start_index: np.ndarray = None
    limit_index: np.ndarray = None
    slice_size: int = None
    stride: int = 1
    keepdims: bool = True


def _dynamic_update_slice(x, update, start_indices):
    """Evaluate _dynamic_update_slice operation.

    Args:
        x (object): The x parameter.
        update (object): The update parameter.
        start_indices (object): The start_indices parameter.

    Returns:
            tuple[int, ...]: Result.
    """
    out: np.ndarray = np.copy(x)

    def _to_int(v) -> int:
        """Evaluate _to_int operation.

        Args:
            v (object): The v parameter.

        Returns:
            int: Result.
        """
        if hasattr(v, "data"):
            v: np.ndarray = v.data
        if hasattr(v, "item"):
            return int(v.item())
        return int(v)

    slices: tuple = tuple(slice(_to_int(start), _to_int(start) + size) for (start, size) in zip(start_indices, update.shape))
    out[slices] = update
    return out


@numpy_eager_registry.register("DynamicUpdateSlice")
def _np_dynamic_update_slice(backend_module, *args, **kwargs):
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
def _np_unstack(backend_module, x, axis: int = 0, *args, **kwargs):
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
def _np_dynamic_slice(backend_module, x, start_indices, slice_sizes):
    """Evaluate _np_dynamic_slice operation.

    Args:
        backend_module (object): The backend_module parameter.
        x (object): The x parameter.
        start_indices (object): The start_indices parameter.
        slice_sizes (object): The slice_sizes parameter.

    Returns:
            tuple[int, ...]: Result.
    """
    slices: tuple = tuple(slice(start, start + size) for (start, size) in zip(start_indices, slice_sizes))
    return x[slices]


@numpy_eager_registry.register("DynamicSliceInDim")
def _np_dynamic_slice_in_dim(backend_module, operand, context: IndexingContext, *args, **kwargs):
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
    operand: np.ndarray = np.asarray(operand)
    start_index: np.ndarray = np.asarray(context.start_index).item()
    sl: list = [slice(None)] * operand.ndim
    sl[context.axis] = slice(start_index, start_index + context.slice_size)
    return operand[tuple(sl)]


@numpy_eager_registry.register("DynamicUpdateSliceInDim")
def _np_dynamic_update_slice_in_dim(backend_module, operand, update, context: IndexingContext, *args, **kwargs):
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
    operand: np.ndarray = np.copy(np.asarray(operand))
    start_index: np.ndarray = np.asarray(context.start_index).item()
    slice_size: np.ndarray = np.asarray(update).shape[context.axis]
    sl: list = [slice(None)] * operand.ndim
    sl[context.axis] = slice(start_index, start_index + slice_size)
    operand[tuple(sl)] = update
    return operand


@numpy_eager_registry.register("DynamicIndexInDim")
def _np_dynamic_index_in_dim(backend_module, operand, index, context: IndexingContext, *args, **kwargs):
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
    operand: np.ndarray = np.asarray(operand)
    idx: np.ndarray = np.asarray(index).item()
    if context.keepdims:
        sl: list = [slice(None)] * operand.ndim
        sl[context.axis] = slice(idx, idx + 1)
        return operand[tuple(sl)]
    else:
        sl: list = [slice(None)] * operand.ndim
        sl[context.axis] = idx
        return operand[tuple(sl)]


@numpy_eager_registry.register("DynamicUpdateIndexInDim")
def _np_dynamic_update_index_in_dim(backend_module, target: IndexTarget, context: IndexingContext, *args, **kwargs):
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
    operand: np.ndarray = np.copy(np.asarray(operand))
    idx: np.ndarray = np.asarray(index).item()
    sl: list = [slice(None)] * operand.ndim
    sl[context.axis] = idx
    operand[tuple(sl)] = update
    return operand


@numpy_eager_registry.register("SliceInDim")
def _np_slice_in_dim(backend_module, operand, context: IndexingContext, *args, **kwargs):
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
    operand: np.ndarray = np.asarray(operand)
    sl: list = [slice(None)] * operand.ndim
    sl[context.axis] = slice(context.start_index, context.limit_index, context.stride)
    return operand[tuple(sl)]


@numpy_eager_registry.register("Slice")
def _np_slice(backend_module, x, context: IndexingContext):
    """Evaluate _np_slice operation.

    Args:
        backend_module (object): The backend_module parameter.
        x (object): The x parameter.
        context (IndexingContext): The context parameter.

    Returns:
            tuple[int, ...]: Result.
    """
    sl: list = [slice(None)] * x.ndim
    sl[context.axis] = slice(context.start_index, context.limit_index, context.stride)
    return x[tuple(sl)]


@numpy_eager_registry.register("GetItem")
def _np_getitem(backend_module, x, key: str):
    """Evaluate _np_getitem operation.

    Args:
        backend_module (object): The backend_module parameter.
        x (object): The x parameter.
        key (str): The key parameter.

    Returns:
            tuple[int, ...]: Result.
    """
    parsed_key: np.ndarray = _safe_parse_key(key)
    return x[parsed_key]


@numpy_eager_registry.register("SetItem")
def _np_setitem(backend_module, x, value, key: str):
    """Evaluate _np_setitem operation.

    Args:
        backend_module (object): The backend_module parameter.
        x (object): The x parameter.
        value (object): The value parameter.
        key (str): The key parameter.

    Returns:
            tuple[int, ...]: Result.
    """
    parsed_key: np.ndarray = _safe_parse_key(key)
    out: np.ndarray = np.copy(np.asarray(x))
    out[parsed_key] = np.asarray(value)
    return out


@numpy_eager_registry.register("IndexInDim")
def _eager_indexindim(backend_module, *args, **kwargs):
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
def gather_eager(np_mod, *args, **kwargs):
    """gather_eager function.

    Args:
        np_mod (object): The np_mod parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    t: threading.Thread = args[0]
    dim: int = args[1] if len(args) > 1 else kwargs.get("dim")
    index: np.ndarray = args[2] if len(args) > 2 else kwargs.get("index")
    if hasattr(t, "numpy"):
        t: threading.Thread = t.numpy()
    if hasattr(index, "numpy"):
        index: np.ndarray = index.numpy()
    return np_mod.take_along_axis(t, index, axis=dim)


@numpy_eager_registry.register("Stack")
def stack_eager(np_mod, *args, **kwargs):
    """stack_eager function.

    Args:
        np_mod (object): The np_mod parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    tensors: list = args[0] if len(args) > 0 else kwargs.get("tensors")
    dim: int = args[1] if len(args) > 1 else kwargs.get("dim", 0)
    if "axis" in kwargs:
        dim: int = kwargs["axis"]
    arrays: list = [t.numpy() if hasattr(t, "numpy") else t for t in tensors]
    return np_mod.stack(arrays, axis=dim)
