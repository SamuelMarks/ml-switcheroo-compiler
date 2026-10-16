"""Core abstractions and logic definitions for indexing.py."""

from __future__ import annotations

import typing  # noqa: F401
from dataclasses import dataclass
from types import ModuleType
from typing import cast

import numpy as np

from ml_switcheroo_compiler.backends.eager_registry import EagerTensorProtocol, EagerValue, numpy_eager_registry

from ._indexing_parsing_utils import _safe_parse_key


@dataclass
class IndexTarget:
    """Index target container."""

    operand: np.ndarray[tuple[int, ...], np.dtype[np.generic]] | None = None
    update: np.ndarray[tuple[int, ...], np.dtype[np.generic]] | None = None
    index: tuple[int | slice | np.ndarray[tuple[int, ...], np.dtype[np.generic]], ...] | np.ndarray[tuple[int, ...], np.dtype[np.generic]] | int | slice | None = None


@dataclass
class IndexingContext:
    """Configuration class for indexing context."""

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
    """Evaluate _dynamic_update_slice operation.

    Args:
        x (object): The x parameter.
        update (object): The update parameter.
        start_indices (object): The start_indices parameter.

    Returns:
            tuple[int, ...]: Result.
    """
    out: np.ndarray[tuple[int, ...], np.dtype[np.generic]] = getattr(np, "copy")(x)

    def _to_int(v: EagerValue) -> int:
        """Evaluate _to_int operation.

        Args:
            v (object): The v parameter.

        Returns:
            int: Result.
        """
        if hasattr(v, "data"):
            v = getattr(v, "data")
        if hasattr(v, "item"):
            return int(getattr(v, "item")())
        return int(cast(int, v))

    slices: tuple[EagerValue, ...] = tuple(slice(_to_int(start), _to_int(start) + int(size)) for (start, size) in zip(getattr(start_indices, "__iter__")(), getattr(update, "shape")))
    out[slices] = update
    return out


@numpy_eager_registry.register("DynamicUpdateSlice")
def _np_dynamic_update_slice(backend_module: ModuleType, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
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
def _np_unstack(backend_module: ModuleType, x: EagerValue, axis: int = 0, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
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
    return [backend_module.squeeze(a, axis=axis) for a in backend_module.split(x, getattr(x, "shape")[axis], axis=axis)]


@numpy_eager_registry.register("DynamicSlice")
def _np_dynamic_slice(backend_module: ModuleType, x: EagerValue, start_indices: EagerValue, slice_sizes: EagerValue) -> EagerValue:
    """Evaluate _np_dynamic_slice operation.

    Args:
        backend_module (object): The backend_module parameter.
        x (object): The x parameter.
        start_indices (object): The start_indices parameter.
        slice_sizes (object): The slice_sizes parameter.

    Returns:
            tuple[int, ...]: Result.
    """
    slices: tuple[EagerValue, ...] = tuple(slice(start, cast(EagerTensorProtocol, start) + cast(EagerTensorProtocol, size)) for (start, size) in zip(getattr(start_indices, "__iter__")(), getattr(slice_sizes, "__iter__")()))
    return cast(EagerValue, getattr(x, "__getitem__")(slices))


@numpy_eager_registry.register("DynamicSliceInDim")
def _np_dynamic_slice_in_dim(backend_module: ModuleType, operand: EagerValue, context: IndexingContext, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
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
    operand_arr = cast(EagerTensorProtocol, np.asarray(operand))
    start_index: int = int(np.asarray(context.start_index).item())
    sl: list[EagerValue] = [slice(None)] * operand_arr.ndim
    sl[context.axis] = slice(start_index, start_index + int(context.slice_size or 0))
    return cast(EagerTensorProtocol, getattr(operand_arr, "__getitem__")(tuple(getattr(sl, "__iter__")())))


@numpy_eager_registry.register("DynamicUpdateSliceInDim")
def _np_dynamic_update_slice_in_dim(backend_module: ModuleType, operand: EagerValue, update: EagerValue, context: IndexingContext, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
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
    operand_arr = cast(EagerTensorProtocol, np.copy(np.asarray(operand)))
    start_index: int = int(np.asarray(context.start_index).item())
    slice_size: int = np.asarray(update).shape[context.axis]
    sl: list[EagerValue] = [slice(None)] * operand_arr.ndim
    sl[context.axis] = slice(start_index, start_index + slice_size)
    getattr(operand_arr, "__setitem__")(tuple(sl), update)
    return operand_arr


@numpy_eager_registry.register("DynamicIndexInDim")
def _np_dynamic_index_in_dim(backend_module: ModuleType, operand: EagerValue, index: EagerValue, context: IndexingContext, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
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
    operand_arr = cast(EagerTensorProtocol, np.asarray(operand))
    idx: int = int(np.asarray(index).item())
    if context.keepdims:
        sl: list[EagerValue] = [slice(None)] * operand_arr.ndim
        sl[context.axis] = slice(idx, idx + 1)
        return cast(EagerTensorProtocol, getattr(operand_arr, "__getitem__")(tuple(getattr(sl, "__iter__")())))
    else:
        sl = [slice(None)] * operand_arr.ndim
        sl[context.axis] = idx
        return cast(EagerTensorProtocol, getattr(operand_arr, "__getitem__")(tuple(getattr(sl, "__iter__")())))


@numpy_eager_registry.register("DynamicUpdateIndexInDim")
def _np_dynamic_update_index_in_dim(backend_module: ModuleType, target: IndexTarget, context: IndexingContext, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
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
    operand_arr = cast(EagerTensorProtocol, np.copy(np.asarray(operand)))
    idx: np.ndarray[tuple[int, ...], np.dtype[np.generic]] = np.asarray(index).item()
    sl: list[EagerValue] = [slice(None)] * operand_arr.ndim
    sl[context.axis] = idx
    getattr(operand_arr, "__setitem__")(tuple(sl), update)
    return operand_arr


@numpy_eager_registry.register("SliceInDim")
def _np_slice_in_dim(backend_module: ModuleType, operand: EagerValue, context: IndexingContext, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
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
    operand_arr = cast(EagerTensorProtocol, np.asarray(operand))
    sl: list[EagerValue] = [slice(None)] * operand_arr.ndim
    sl[context.axis] = slice(context.start_index, context.limit_index, context.stride)
    return cast(EagerTensorProtocol, getattr(operand_arr, "__getitem__")(tuple(getattr(sl, "__iter__")())))


@numpy_eager_registry.register("Slice")
def _np_slice(backend_module: ModuleType, x: EagerValue, context: IndexingContext) -> EagerValue:
    """Evaluate _np_slice operation.

    Args:
        backend_module (object): The backend_module parameter.
        x (object): The x parameter.
        context (IndexingContext): The context parameter.

    Returns:
            tuple[int, ...]: Result.
    """
    sl: list[EagerValue] = [slice(None)] * getattr(x, "ndim")
    sl[context.axis] = slice(context.start_index, context.limit_index, context.stride)
    return cast(EagerValue, getattr(x, "__getitem__")(tuple(sl)))


@numpy_eager_registry.register("GetItem")
def _np_getitem(backend_module: ModuleType, x: EagerValue, key: str) -> EagerValue:
    """Evaluate _np_getitem operation.

    Args:
        backend_module (object): The backend_module parameter.
        x (object): The x parameter.
        key (str): The key parameter.

    Returns:
            tuple[int, ...]: Result.
    """
    parsed_key: np.ndarray[tuple[int, ...], np.dtype[np.generic]] = cast(np.ndarray[tuple[int, ...], np.dtype[np.generic]], _safe_parse_key(key))
    return cast(EagerValue, getattr(x, "__getitem__")(parsed_key))


@numpy_eager_registry.register("SetItem")
def _np_setitem(backend_module: ModuleType, x: EagerValue, value: EagerValue, key: str) -> EagerValue:
    """Evaluate _np_setitem operation.

    Args:
        backend_module (object): The backend_module parameter.
        x (object): The x parameter.
        value (object): The value parameter.
        key (str): The key parameter.

    Returns:
            tuple[int, ...]: Result.
    """
    parsed_key: np.ndarray[tuple[int, ...], np.dtype[np.generic]] = cast(np.ndarray[tuple[int, ...], np.dtype[np.generic]], _safe_parse_key(key))
    out: np.ndarray[tuple[int, ...], np.dtype[np.generic]] = np.copy(np.asarray(x))
    out[parsed_key] = np.asarray(value)
    return out


@numpy_eager_registry.register("IndexInDim")
def _eager_indexindim(backend_module: ModuleType, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
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
    return cast(EagerValue, getattr(np, "take")(x, idx, axis=dim))


@numpy_eager_registry.register("Gather")
def gather_eager(np_mod: ModuleType, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """gather_eager function.

    Args:
        np_mod (object): The np_mod parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    t = args[0] if len(args) > 0 else kwargs.get("tensor")
    dim: int | None = cast("int | None", args[1] if len(args) > 1 else kwargs.get("dim"))
    index: np.ndarray[tuple[int, ...], np.dtype[np.generic]] | None = cast("np.ndarray[tuple[int, ...], np.dtype[np.generic]] | None", args[2] if len(args) > 2 else kwargs.get("index"))
    if hasattr(t, "numpy"):
        t = getattr(t, "numpy")()
    if hasattr(index, "numpy"):
        index = getattr(index, "numpy")()
    return cast(EagerValue, np_mod.take_along_axis(t, index, axis=dim))


@numpy_eager_registry.register("Stack")
def stack_eager(np_mod: ModuleType, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """stack_eager function.

    Args:
        np_mod (object): The np_mod parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    tensors: list[EagerValue] = cast(list[EagerValue], args[0] if len(args) > 0 else (kwargs.get("tensors") or []))
    dim: int = cast(int, args[1] if len(args) > 1 else kwargs.get("dim", 0))
    if "axis" in kwargs:
        dim = cast(int, kwargs["axis"])
    arrays: list[EagerValue] = [getattr(t, "numpy")() if hasattr(t, "numpy") else t for t in tensors]
    return cast(EagerValue, np_mod.stack(arrays, axis=dim))
