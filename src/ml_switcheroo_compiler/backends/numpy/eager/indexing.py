"""Implementation of indexing operations and utilities for the backends subsystem (`ml_switcheroo_compiler.backends.numpy.eager.indexing`)."""

from __future__ import annotations

import typing  # noqa: F401

import numpy as np
import pydantic

from ml_switcheroo_compiler.backends.eager_registry import EagerTensorProtocol, EagerValue, numpy_eager_registry
from ml_switcheroo_compiler.backends.protocols import BackendModuleProtocol

from ._indexing_parsing_utils import _safe_parse_key


class IndexTarget(pydantic.BaseModel):
    """Index target container."""

    model_config = pydantic.ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    operand: EagerValue | None = None
    update: EagerValue | None = None
    index: EagerValue | None = None


class IndexingContext(pydantic.BaseModel):
    """Configuration class for indexing context."""

    model_config = pydantic.ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    axis: int = 0
    start_index: np.ndarray[tuple[int, ...], np.dtype[np.generic]] | int | None = None
    limit_index: np.ndarray[tuple[int, ...], np.dtype[np.generic]] | int | None = None
    slice_size: int | None = None
    stride: int = 1
    keepdims: bool = True


def _dynamic_update_slice(
    x: EagerValue,
    update: EagerValue,
    start_indices: EagerValue,
) -> np.ndarray[tuple[int, ...], np.dtype[np.generic]]:
    """Execute the eager dynamic update slice operation on the active backend array.

    Args:
        x (EagerValue): The x parameter.
        update (EagerValue): The update parameter.
        start_indices (EagerValue): The start_indices parameter.

    Returns:
            tuple[int, ...]: Result.
    """
    out: np.ndarray[tuple[int, ...], np.dtype[np.generic]] = getattr(np, "copy")(x)

    def _to_int(v: EagerValue) -> int:
        """Execute the eager to int operation on the active backend array.

        Args:
            v (EagerValue): The v parameter.

        Returns:
            int: Result.
        """
        if hasattr(v, "data"):
            v = getattr(v, "data")
        if hasattr(v, "item"):
            return int(getattr(v, "item")())
        return int(typing.cast(int, v))

    slices: tuple[EagerValue, ...] = tuple(slice(_to_int(start), _to_int(start) + int(size)) for (start, size) in zip(getattr(start_indices, "__iter__")(), getattr(update, "shape")))
    out[slices] = update
    return out


@numpy_eager_registry.register("DynamicUpdateSlice")
def _np_dynamic_update_slice(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager np dynamic update slice operation on the active backend array.

    Args:
        backend_module (ModuleType): The backend_module parameter.
        *args (EagerValue): Positional args.
        **kwargs (EagerValue): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    return _dynamic_update_slice(*args, **kwargs)


@numpy_eager_registry.register("Unstack")
def _np_unstack(backend_module: BackendModuleProtocol, x: EagerValue, axis: int = 0, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
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


@numpy_eager_registry.register("DynamicSlice")
def _np_dynamic_slice(backend_module: BackendModuleProtocol, x: EagerValue, start_indices: EagerValue, slice_sizes: EagerValue) -> EagerValue:
    """Execute the eager np dynamic slice operation on the active backend array.

    Args:
        backend_module (ModuleType): The backend_module parameter.
        x (EagerValue): The x parameter.
        start_indices (EagerValue): The start_indices parameter.
        slice_sizes (EagerValue): The slice_sizes parameter.

    Returns:
            tuple[int, ...]: Result.
    """
    slices: tuple[EagerValue, ...] = tuple(slice(start, int(typing.cast(EagerTensorProtocol, start) + typing.cast(EagerTensorProtocol, size))) for (start, size) in zip(getattr(start_indices, "__iter__")(), getattr(slice_sizes, "__iter__")()))
    return typing.cast("EagerValue", getattr(x, "__getitem__")(slices))


@numpy_eager_registry.register("DynamicSliceInDim")
def _np_dynamic_slice_in_dim(backend_module: BackendModuleProtocol, operand: EagerValue, context: IndexingContext, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager np dynamic slice in dim operation on the active backend array.

    Args:
        backend_module (ModuleType): The backend_module parameter.
        operand (EagerValue): The operand parameter.
        context (IndexingContext): The context parameter.
        *args (EagerValue): Positional args.
        **kwargs (EagerValue): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    operand_arr = typing.cast(EagerTensorProtocol, np.asarray(operand))
    start_index: int = int(np.asarray(context.start_index).item())
    sl: list[EagerValue] = [slice(None)] * operand_arr.ndim
    sl[context.axis] = slice(start_index, start_index + int(context.slice_size or 0))
    return typing.cast(EagerTensorProtocol, getattr(operand_arr, "__getitem__")(tuple(getattr(sl, "__iter__")())))


@numpy_eager_registry.register("DynamicUpdateSliceInDim")
def _np_dynamic_update_slice_in_dim(backend_module: BackendModuleProtocol, operand: EagerValue, update: EagerValue, context: IndexingContext, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager np dynamic update slice in dim operation on the active backend array.

    Args:
        backend_module (ModuleType): The backend_module parameter.
        operand (EagerValue): The operand parameter.
        update (EagerValue): The update parameter.
        context (IndexingContext): The context parameter.
        *args (EagerValue): Positional args.
        **kwargs (EagerValue): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    operand_arr = typing.cast(EagerTensorProtocol, np.copy(np.asarray(operand)))
    start_index: int = int(np.asarray(context.start_index).item())
    slice_size: int = np.asarray(update).shape[context.axis]
    sl: list[EagerValue] = [slice(None)] * operand_arr.ndim
    sl[context.axis] = slice(start_index, start_index + slice_size)
    getattr(operand_arr, "__setitem__")(tuple(sl), update)
    return operand_arr


@numpy_eager_registry.register("DynamicIndexInDim")
def _np_dynamic_index_in_dim(backend_module: BackendModuleProtocol, operand: EagerValue, index: EagerValue, context: IndexingContext, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager np dynamic index in dim operation on the active backend array.

    Args:
        backend_module (ModuleType): The backend_module parameter.
        operand (EagerValue): The operand parameter.
        index (EagerValue): The index parameter.
        context (IndexingContext): The context parameter.
        *args (EagerValue): Positional args.
        **kwargs (EagerValue): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    operand_arr = typing.cast(EagerTensorProtocol, np.asarray(operand))
    idx: int = int(np.asarray(index).item())
    if context.keepdims:
        sl: list[EagerValue] = [slice(None)] * operand_arr.ndim
        sl[context.axis] = slice(idx, idx + 1)
        return typing.cast(EagerTensorProtocol, getattr(operand_arr, "__getitem__")(tuple(getattr(sl, "__iter__")())))
    else:
        sl = [slice(None)] * operand_arr.ndim
        sl[context.axis] = idx
        return typing.cast(EagerTensorProtocol, getattr(operand_arr, "__getitem__")(tuple(getattr(sl, "__iter__")())))


@numpy_eager_registry.register("DynamicUpdateIndexInDim")
def _np_dynamic_update_index_in_dim(backend_module: BackendModuleProtocol, target: IndexTarget, context: IndexingContext, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager np dynamic update index in dim operation on the active backend array.

    Args:
        backend_module (ModuleType): The backend_module parameter.
        target (IndexTarget): The target parameter.
        context (IndexingContext): The context parameter.
        *args (EagerValue): Positional args.
        **kwargs (EagerValue): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    operand, update, index = target.operand, target.update, target.index
    operand_arr = typing.cast(EagerTensorProtocol, np.copy(np.asarray(operand)))
    idx: np.ndarray[tuple[int, ...], np.dtype[np.generic]] = np.asarray(index).item()
    sl: list[EagerValue] = [slice(None)] * operand_arr.ndim
    sl[context.axis] = idx
    getattr(operand_arr, "__setitem__")(tuple(sl), update)
    return operand_arr


@numpy_eager_registry.register("SliceInDim")
def _np_slice_in_dim(backend_module: BackendModuleProtocol, operand: EagerValue, context: IndexingContext, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager np slice in dim operation on the active backend array.

    Args:
        backend_module (ModuleType): The backend_module parameter.
        operand (EagerValue): The operand parameter.
        context (IndexingContext): The context parameter.
        *args (EagerValue): Positional args.
        **kwargs (EagerValue): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    operand_arr = typing.cast(EagerTensorProtocol, np.asarray(operand))
    sl: list[EagerValue] = [slice(None)] * operand_arr.ndim
    sl[context.axis] = slice(typing.cast("int | None", context.start_index), typing.cast("int | None", context.limit_index), context.stride)
    return typing.cast(EagerTensorProtocol, getattr(operand_arr, "__getitem__")(tuple(getattr(sl, "__iter__")())))


@numpy_eager_registry.register("Slice")
def _np_slice(backend_module: BackendModuleProtocol, x: EagerValue, context: IndexingContext) -> EagerValue:
    """Execute the eager np slice operation on the active backend array.

    Args:
        backend_module (ModuleType): The backend_module parameter.
        x (EagerValue): The x parameter.
        context (IndexingContext): The context parameter.

    Returns:
            tuple[int, ...]: Result.
    """
    sl: list[EagerValue] = [slice(None)] * getattr(x, "ndim")
    sl[context.axis] = slice(typing.cast("int | None", context.start_index), typing.cast("int | None", context.limit_index), context.stride)
    return typing.cast("EagerValue", getattr(x, "__getitem__")(tuple(sl)))


@numpy_eager_registry.register("GetItem")
def _np_getitem(backend_module: BackendModuleProtocol, x: EagerValue, key: str) -> EagerValue:
    """Execute the eager np getitem operation on the active backend array.

    Args:
        backend_module (ModuleType): The backend_module parameter.
        x (EagerValue): The x parameter.
        key (str): The key parameter.

    Returns:
            tuple[int, ...]: Result.
    """
    parsed_key: np.ndarray[tuple[int, ...], np.dtype[np.generic]] = typing.cast(np.ndarray[tuple[int, ...], np.dtype[np.generic]], _safe_parse_key(key))
    return typing.cast("EagerValue", getattr(x, "__getitem__")(parsed_key))


@numpy_eager_registry.register("SetItem")
def _np_setitem(backend_module: BackendModuleProtocol, x: EagerValue, value: EagerValue, key: str) -> EagerValue:
    """Execute the eager np setitem operation on the active backend array.

    Args:
        backend_module (ModuleType): The backend_module parameter.
        x (EagerValue): The x parameter.
        value (EagerValue): The value parameter.
        key (str): The key parameter.

    Returns:
            tuple[int, ...]: Result.
    """
    parsed_key: np.ndarray[tuple[int, ...], np.dtype[np.generic]] = typing.cast(np.ndarray[tuple[int, ...], np.dtype[np.generic]], _safe_parse_key(key))
    out: np.ndarray[tuple[int, ...], np.dtype[np.generic]] = np.copy(np.asarray(x))
    out[parsed_key] = np.asarray(value)
    return out


@numpy_eager_registry.register("IndexInDim")
def _eager_indexindim(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager eager indexindim operation on the active backend array.

    Args:
        backend_module (ModuleType): The backend_module parameter.
        *args (EagerValue): Positional args.
        **kwargs (EagerValue): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    import numpy as np

    x, idx, dim = args[0], args[1], args[2]
    return typing.cast("EagerValue", getattr(np, "take")(x, idx, axis=dim))


@numpy_eager_registry.register("Gather")
def gather_eager(np_mod: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager gather eager operation on the active backend array.

    Args:
        np_mod (ModuleType): The np_mod parameter.
        *args (EagerValue): Positional args.
        **kwargs (EagerValue): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    t = args[0] if len(args) > 0 else kwargs.get("tensor")
    dim: int | None = typing.cast("int | None", args[1] if len(args) > 1 else kwargs.get("dim"))
    index: np.ndarray[tuple[int, ...], np.dtype[np.generic]] | None = typing.cast("np.ndarray[tuple[int, ...], np.dtype[np.generic]] | None", args[2] if len(args) > 2 else kwargs.get("index"))
    if hasattr(t, "numpy"):
        t = getattr(t, "numpy")()
    if hasattr(index, "numpy"):
        index = getattr(index, "numpy")()
    return np_mod.take_along_axis(t, index, axis=dim)


@numpy_eager_registry.register("Stack")
def stack_eager(np_mod: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager stack eager operation on the active backend array.

    Args:
        np_mod (ModuleType): The np_mod parameter.
        *args (EagerValue): Positional args.
        **kwargs (EagerValue): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    tensors: list[EagerValue] = typing.cast(list[EagerValue], args[0] if len(args) > 0 else (kwargs.get("tensors") or []))
    dim: int = typing.cast(int, args[1] if len(args) > 1 else kwargs.get("dim", 0))
    if "axis" in kwargs:
        dim = typing.cast(int, kwargs["axis"])
    arrays: list[EagerValue] = [getattr(t, "numpy")() if hasattr(t, "numpy") else t for t in tensors]
    return np_mod.stack(arrays, axis=dim)
