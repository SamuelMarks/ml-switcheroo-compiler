"""Module indexing.py."""

from typing import Any

# ruff: noqa: E402, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, E701, E722, F403, E711, E712, PLR0913, PLR0915
# ruff: noqa: E402, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, E701, E722, F403, E711, E712, PLR0913, PLR0915
"""Core abstractions and logic definitions for indexing.py."""

from dataclasses import dataclass


@dataclass
class IndexTarget:
    """Index target container."""

    operand: Any
    update: Any
    index: Any


import numpy as np

from ml_switcheroo_compiler.backends.eager_registry import numpy_eager_registry

from ._indexing_parsing_utils import _safe_parse_key

"""Core abstractions and logic definitions for indexing.py."""
from typing import Any


@dataclass
class IndexingContext:
    """Configuration class for indexing context."""

    axis: int = 0
    start_index: Any = None
    limit_index: Any = None
    slice_size: Any = None
    stride: int = 1
    keepdims: bool = True


def _dynamic_update_slice(x: Any, update: Any, start_indices: Any) -> Any:
    """Evaluate _dynamic_update_slice operation.

    Args:
        x (object): The x parameter.
        update (object): The update parameter.
        start_indices (object): The start_indices parameter.

    Returns: Any: Result.
    """
    out = np.copy(x)

    def _to_int(v: Any) -> int:
        """Evaluate _to_int operation.

        Args:
            v (object): The v parameter.

        Returns:
            int: Result.
        """
        if hasattr(v, "data"):
            v = v.data
        if hasattr(v, "item"):
            return int(v.item())
        return int(v)

    slices = tuple(slice(_to_int(start), _to_int(start) + size) for (start, size) in zip(start_indices, update.shape))
    out[slices] = update
    return out


@numpy_eager_registry.register("DynamicUpdateSlice")
def _np_dynamic_update_slice(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _np_dynamic_update_slice operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    return _dynamic_update_slice(*args, **kwargs)


@numpy_eager_registry.register("Unstack")
def _np_unstack(backend_module: Any, x: Any, axis: Any = 0, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _np_unstack operation.

    Args:
        backend_module (object): The backend_module parameter.
        x (object): The x parameter.
        axis (object): The axis parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    return [backend_module.squeeze(a, axis=axis) for a in backend_module.split(x, x.shape[axis], axis=axis)]


@numpy_eager_registry.register("DynamicSlice")
def _np_dynamic_slice(backend_module: Any, x: Any, start_indices: Any, slice_sizes: Any) -> Any:
    """Evaluate _np_dynamic_slice operation.

    Args:
        backend_module (object): The backend_module parameter.
        x (object): The x parameter.
        start_indices (object): The start_indices parameter.
        slice_sizes (object): The slice_sizes parameter.

    Returns: Any: Result.
    """
    slices = tuple(slice(start, start + size) for (start, size) in zip(start_indices, slice_sizes))
    return x[slices]


@numpy_eager_registry.register("DynamicSliceInDim")
def _np_dynamic_slice_in_dim(backend_module: Any, operand: Any, context: IndexingContext, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _np_dynamic_slice_in_dim operation.

    Args:
        backend_module (object): The backend_module parameter.
        operand (object): The operand parameter.
        context (IndexingContext): The context parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    operand = np.asarray(operand)
    start_index = np.asarray(context.start_index).item()
    sl = [slice(None)] * operand.ndim
    sl[context.axis] = slice(start_index, start_index + context.slice_size)
    return operand[tuple(sl)]


@numpy_eager_registry.register("DynamicUpdateSliceInDim")
def _np_dynamic_update_slice_in_dim(backend_module: Any, operand: Any, update: Any, context: IndexingContext, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _np_dynamic_update_slice_in_dim operation.

    Args:
        backend_module (object): The backend_module parameter.
        operand (object): The operand parameter.
        update (object): The update parameter.
        context (IndexingContext): The context parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    operand = np.copy(np.asarray(operand))
    start_index = np.asarray(context.start_index).item()
    slice_size = np.asarray(update).shape[context.axis]
    sl = [slice(None)] * operand.ndim
    sl[context.axis] = slice(start_index, start_index + slice_size)
    operand[tuple(sl)] = update
    return operand


@numpy_eager_registry.register("DynamicIndexInDim")
def _np_dynamic_index_in_dim(backend_module: Any, operand: Any, index: Any, context: IndexingContext, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _np_dynamic_index_in_dim operation.

    Args:
        backend_module (object): The backend_module parameter.
        operand (object): The operand parameter.
        index (object): The index parameter.
        context (IndexingContext): The context parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    operand = np.asarray(operand)
    idx = np.asarray(index).item()
    if context.keepdims:
        sl = [slice(None)] * operand.ndim
        sl[context.axis] = slice(idx, idx + 1)
        return operand[tuple(sl)]
    else:
        sl = [slice(None)] * operand.ndim
        sl[context.axis] = idx
        return operand[tuple(sl)]


@numpy_eager_registry.register("DynamicUpdateIndexInDim")
def _np_dynamic_update_index_in_dim(backend_module: Any, target: IndexTarget, context: IndexingContext, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _np_dynamic_update_index_in_dim operation.

    Args:
        backend_module (object): The backend_module parameter.
        target (IndexTarget): The target parameter.
        context (IndexingContext): The context parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    operand, update, index = target.operand, target.update, target.index
    operand = np.copy(np.asarray(operand))
    idx = np.asarray(index).item()
    sl = [slice(None)] * operand.ndim
    sl[context.axis] = idx
    operand[tuple(sl)] = update
    return operand


@numpy_eager_registry.register("SliceInDim")
def _np_slice_in_dim(backend_module: Any, operand: Any, context: IndexingContext, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _np_slice_in_dim operation.

    Args:
        backend_module (object): The backend_module parameter.
        operand (object): The operand parameter.
        context (IndexingContext): The context parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    operand = np.asarray(operand)
    sl = [slice(None)] * operand.ndim
    sl[context.axis] = slice(context.start_index, context.limit_index, context.stride)
    return operand[tuple(sl)]


@numpy_eager_registry.register("Slice")
def _np_slice(backend_module: Any, x: Any, context: IndexingContext) -> Any:
    """Evaluate _np_slice operation.

    Args:
        backend_module (object): The backend_module parameter.
        x (object): The x parameter.
        context (IndexingContext): The context parameter.

    Returns: Any: Result.
    """
    sl = [slice(None)] * x.ndim
    sl[context.axis] = slice(context.start_index, context.limit_index, context.stride)
    return x[tuple(sl)]


@numpy_eager_registry.register("GetItem")
def _np_getitem(backend_module: Any, x: Any, key: str) -> Any:
    """Evaluate _np_getitem operation.

    Args:
        backend_module (object): The backend_module parameter.
        x (object): The x parameter.
        key (str): The key parameter.

    Returns: Any: Result.
    """
    parsed_key = _safe_parse_key(key)
    return x[parsed_key]


@numpy_eager_registry.register("SetItem")
def _np_setitem(backend_module: Any, x: Any, value: Any, key: str) -> Any:
    """Evaluate _np_setitem operation.

    Args:
        backend_module (object): The backend_module parameter.
        x (object): The x parameter.
        value (object): The value parameter.
        key (str): The key parameter.

    Returns: Any: Result.
    """
    parsed_key = _safe_parse_key(key)
    out = np.copy(np.asarray(x))
    out[parsed_key] = np.asarray(value)
    return out


@numpy_eager_registry.register("IndexInDim")
def _eager_indexindim(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _eager_indexindim operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    import numpy as np

    x, idx, dim = args[0], args[1], args[2]
    return np.take(x, idx, axis=dim)


@numpy_eager_registry.register("Gather")
def gather_eager(np_mod: Any, *args: Any, **kwargs: Any) -> Any:
    """gather_eager function.

    Args:
        np_mod (object): The np_mod parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
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
    """stack_eager function.

    Args:
        np_mod (object): The np_mod parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    tensors = args[0] if len(args) > 0 else kwargs.get("tensors")
    dim = args[1] if len(args) > 1 else kwargs.get("dim", 0)
    if "axis" in kwargs:
        dim = kwargs["axis"]
    arrays = [t.numpy() if hasattr(t, "numpy") else t for t in tensors]  # type: ignore  # Justification: Polymorphic / Duck Typing for Framework Agnosticism
    return np_mod.stack(arrays, axis=dim)
