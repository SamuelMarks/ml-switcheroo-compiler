# ruff: noqa
# ruff: noqa: E501
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
    start_index: int = None
    limit_index: int = None
    slice_size: int = None
    stride: int = 1
    keepdims: bool = True


def _dynamic_update_slice(x: object, update: object, start_indices: object) -> object:
    """Evaluate and process the dynamic update slice operation.

    Args:
        x (object): Required parameter for x.
        update (object): Required parameter for update.
        start_indices (object): Required parameter for start_indices.

    Returns:
        object: The evaluated or processed output.
    """
    out = np.copy(x)

    def _to_int(v: object) -> int:
        """Evaluate and process the to int operation.

        Args:
            v (object): Required parameter for v.

        Returns:
            int: The evaluated or processed output.
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
def _np_dynamic_update_slice(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate the dynamic update slice logic eagerly backed by NumPy.

    Args:
        backend_module (object): Required parameter for backend_module.
        *args (Any): Variable positional arguments.
        **kwargs (Any): Arbitrary keyword arguments.

    Returns:
        object: The evaluated or processed output.
    """
    return _dynamic_update_slice(*args, **kwargs)


@numpy_eager_registry.register("Unstack")
def _np_unstack(backend_module: object, x: object, axis: object = 0, *args: object, **kwargs: object) -> object:
    """Evaluate the unstack logic eagerly backed by NumPy.

    Args:
        backend_module (object): Required parameter for backend_module.
        x (object): Required parameter for x.
        axis (object): Required parameter for axis.
        *args (Any): Variable positional arguments.
        **kwargs (Any): Arbitrary keyword arguments.

    Returns:
        object: The evaluated or processed output.
    """
    return [backend_module.squeeze(a, axis=axis) for a in backend_module.split(x, x.shape[axis], axis=axis)]


@numpy_eager_registry.register("DynamicSlice")
def _np_dynamic_slice(backend_module: object, x: object, start_indices: object, slice_sizes: object) -> object:
    """Evaluate the dynamic slice logic eagerly backed by NumPy.

    Args:
        backend_module (object): Required parameter for backend_module.
        x (object): Required parameter for x.
        start_indices (object): Required parameter for start_indices.
        slice_sizes (object): Required parameter for slice_sizes.

    Returns:
        object: The evaluated or processed output.
    """
    slices = tuple(slice(start, start + size) for (start, size) in zip(start_indices, slice_sizes))
    return x[slices]


@numpy_eager_registry.register("DynamicSliceInDim")
def _np_dynamic_slice_in_dim(backend_module: object, operand: object, context: IndexingContext, *args: object, **kwargs: object) -> object:
    """Evaluate the dynamic slice in dim logic eagerly backed by NumPy.

    Args:
        backend_module (object): Required parameter for backend_module.
        operand (object): Required parameter for operand.
        context (IndexingContext): Required parameter for context.
        *args (Any): Variable positional arguments.
        **kwargs (Any): Arbitrary keyword arguments.

    Returns:
        object: The evaluated or processed output.
    """
    operand = np.asarray(operand)
    start_index = np.asarray(context.start_index).item()
    sl = [slice(None)] * operand.ndim
    sl[context.axis] = slice(start_index, start_index + context.slice_size)
    return operand[tuple(sl)]


@numpy_eager_registry.register("DynamicUpdateSliceInDim")
def _np_dynamic_update_slice_in_dim(backend_module: object, operand: object, update: object, context: IndexingContext, *args: object, **kwargs: object) -> object:
    """Evaluate the dynamic update slice in dim logic eagerly backed by NumPy.

    Args:
        backend_module (object): Required parameter for backend_module.
        operand (object): Required parameter for operand.
        update (object): Required parameter for update.
        context (IndexingContext): Required parameter for context.
        *args (Any): Variable positional arguments.
        **kwargs (Any): Arbitrary keyword arguments.

    Returns:
        object: The evaluated or processed output.
    """
    operand = np.copy(np.asarray(operand))
    start_index = np.asarray(context.start_index).item()
    slice_size = np.asarray(update).shape[context.axis]
    sl = [slice(None)] * operand.ndim
    sl[context.axis] = slice(start_index, start_index + slice_size)
    operand[tuple(sl)] = update
    return operand


@numpy_eager_registry.register("DynamicIndexInDim")
def _np_dynamic_index_in_dim(backend_module: object, operand: object, index: object, context: IndexingContext, *args: object, **kwargs: object) -> object:
    """Evaluate the dynamic index in dim logic eagerly backed by NumPy.

    Args:
        backend_module (object): Required parameter for backend_module.
        operand (object): Required parameter for operand.
        index (object): Required parameter for index.
        context (IndexingContext): Required parameter for context.
        *args (Any): Variable positional arguments.
        **kwargs (Any): Arbitrary keyword arguments.

    Returns:
        object: The evaluated or processed output.
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
def _np_dynamic_update_index_in_dim(backend_module: object, target: IndexTarget, context: IndexingContext, *args: object, **kwargs: object) -> object:
    operand, update, index = target.operand, target.update, target.index

    """Evaluate the dynamic update index in dim logic eagerly backed by NumPy.

    Args:
        backend_module (object): Required parameter for backend_module.
        operand (object): Required parameter for operand.
        update (object): Required parameter for update.
        index (object): Required parameter for index.
        context (IndexingContext): Required parameter for context.
        *args (Any): Variable positional arguments.
        **kwargs (Any): Arbitrary keyword arguments.

    Returns:
        object: The evaluated or processed output.
    """
    operand = np.copy(np.asarray(operand))
    idx = np.asarray(index).item()
    sl = [slice(None)] * operand.ndim
    sl[context.axis] = idx
    operand[tuple(sl)] = update
    return operand


@numpy_eager_registry.register("SliceInDim")
def _np_slice_in_dim(backend_module: object, operand: object, context: IndexingContext, *args: object, **kwargs: object) -> object:
    """Evaluate the slice in dim logic eagerly backed by NumPy.

    Args:
        backend_module (object): Required parameter for backend_module.
        operand (object): Required parameter for operand.
        context (IndexingContext): Required parameter for context.
        *args (Any): Variable positional arguments.
        **kwargs (Any): Arbitrary keyword arguments.

    Returns:
        object: The evaluated or processed output.
    """
    operand = np.asarray(operand)
    sl = [slice(None)] * operand.ndim
    sl[context.axis] = slice(context.start_index, context.limit_index, context.stride)
    return operand[tuple(sl)]


@numpy_eager_registry.register("Slice")
def _np_slice(backend_module: object, x: object, context: IndexingContext) -> object:
    """Evaluate the slice logic eagerly backed by NumPy.

    Args:
        backend_module (object): Required parameter for backend_module.
        x (object): Required parameter for x.
        context (IndexingContext): Required parameter for context.

    Returns:
        object: The evaluated or processed output.
    """
    sl = [slice(None)] * x.ndim
    sl[context.axis] = slice(context.start_index, context.limit_index, context.stride)
    return x[tuple(sl)]


@numpy_eager_registry.register("GetItem")
def _np_getitem(backend_module: object, x: object, key: str) -> object:
    """Evaluate the getitem logic eagerly backed by NumPy.

    Args:
        backend_module (object): Required parameter for backend_module.
        x (object): Required parameter for x.
        key (str): Required parameter for key.

    Returns:
        object: The evaluated or processed output.
    """
    parsed_key = _safe_parse_key(key)
    return x[parsed_key]


@numpy_eager_registry.register("SetItem")
def _np_setitem(backend_module: object, x: object, value: object, key: str) -> object:
    """Evaluate the setitem logic eagerly backed by NumPy.

    Args:
        backend_module (object): Required parameter for backend_module.
        x (object): Required parameter for x.
        value (object): Required parameter for value.
        key (str): Required parameter for key.

    Returns:
        object: The evaluated or processed output.
    """
    parsed_key = _safe_parse_key(key)
    out = np.copy(np.asarray(x))
    out[parsed_key] = np.asarray(value)
    return out


@numpy_eager_registry.register("IndexInDim")
def _eager_indexindim(backend_module: object, *args: object, **kwargs: object) -> object:
    import numpy as np

    x, idx, dim = args[0], args[1], args[2]
    return np.take(x, idx, axis=dim)


@numpy_eager_registry.register("Gather")
def gather_eager(np_mod: object, *args: object, **kwargs: object) -> object:
    """gather_eager function."""
    t = args[0]
    dim = args[1] if len(args) > 1 else kwargs.get("dim")
    index = args[2] if len(args) > 2 else kwargs.get("index")
    if hasattr(t, "numpy"):
        t = t.numpy()
    if hasattr(index, "numpy"):
        index = index.numpy()
    return np_mod.take_along_axis(t, index, axis=dim)


@numpy_eager_registry.register("Stack")
def stack_eager(np_mod: object, *args: object, **kwargs: object) -> object:
    """stack_eager function."""
    tensors = args[0] if len(args) > 0 else kwargs.get("tensors")
    dim = args[1] if len(args) > 1 else kwargs.get("dim", 0)
    if "axis" in kwargs:
        dim = kwargs["axis"]
    arrays = [t.numpy() if hasattr(t, "numpy") else t for t in tensors]
    return np_mod.stack(arrays, axis=dim)
