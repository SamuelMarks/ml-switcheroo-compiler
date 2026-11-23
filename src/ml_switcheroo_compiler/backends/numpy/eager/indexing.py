"""Operations and utilities for `ml_switcheroo_compiler.backends.numpy.eager.indexing`."""

from __future__ import annotations

import types
import typing  # noqa: F401

import numpy as np
import pydantic

from ml_switcheroo_compiler.backends.eager_registry import (
    BackendArg,
    coerce_int,
    coerce_operand_list,
    coerce_opt_int,
    numpy_eager_registry,
)
from ml_switcheroo_compiler.backends.numpy.eager import ensure_ndarray
from ml_switcheroo_compiler.backends.protocols import BackendModuleProtocol, adapt_backend_module
from ml_switcheroo_compiler.core.shape import OpValue as OpValue

from ._indexing_parsing_utils import _safe_parse_key


class IndexTarget(pydantic.BaseModel):
    """Index target container."""

    model_config = pydantic.ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    operand: BackendArg | None = None
    update: BackendArg | None = None
    index: BackendArg | None = None


class IndexingContext(pydantic.BaseModel):
    """Configuration class for indexing context."""

    model_config = pydantic.ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    axis: int = 0
    start_index: np.ndarray[tuple[int, ...], np.dtype[np.float64]] | int | None = None
    limit_index: np.ndarray[tuple[int, ...], np.dtype[np.float64]] | int | None = None
    slice_size: int | None = None
    stride: int = 1
    keepdims: bool = True


def _dynamic_update_slice(
    x: BackendArg,
    update: BackendArg,
    start_indices: BackendArg,
) -> np.ndarray[tuple[int, ...], np.dtype[np.float64]]:
    """Execute the eager dynamic update slice operation on the active backend array.

    Args:
        x (BackendArg): Input `x` argument (x).
        update (BackendArg): Input `update` argument (update).
        start_indices (BackendArg): Input `start_indices` argument (start indices).

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    out: np.ndarray[tuple[int, ...], np.dtype[np.float64]] = ensure_ndarray(
        adapt_backend_module(np).require_op("copy")(x)
    )

    def _to_int(v: BackendArg) -> int:
        """Execute the eager to int operation on the active backend array.

        Args:
            v (BackendArg): Input `v` argument (v).

        Returns:
            int: Computed `int` output.
        """
        if adapt_backend_module(v).has_symbol("data"):
            v = adapt_backend_module(v).require_symbol("data")
        if adapt_backend_module(v).has_symbol("item"):
            return int(adapt_backend_module(v).require_symbol("item")())
        return int(coerce_int(v))

    slices: tuple[slice[int | None, int | None, int | None], ...] = tuple(
        slice(_to_int(start), _to_int(start) + int(coerce_int(size)))
        for (start, size) in zip(
            coerce_operand_list(start_indices), adapt_backend_module(update).require_symbol("shape")
        )
    )
    out[slices] = ensure_ndarray(update)
    return out


def _np_dynamic_update_slice(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager np dynamic update slice operation on the active backend array.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None):
            Input `backend_module` argument (backend module).
        *args (BackendArg): Positional args.
        **kwargs (BackendArg): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    return _dynamic_update_slice(*args, **kwargs)


@numpy_eager_registry.register("Unstack")
def _np_unstack(
    backend_module: BackendModuleProtocol | types.ModuleType | None,
    x: BackendArg,
    axis: int = 0,
    *args: BackendArg,
    **kwargs: BackendArg,
) -> BackendArg:
    """Execute the eager np unstack operation on the active backend array.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None):
            Input `backend_module` argument (backend module).
        x (BackendArg): Input `x` argument (x).
        axis (int): Input `axis` argument (axis).
        *args (BackendArg): Positional args.
        **kwargs (BackendArg): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    return [
        adapt_backend_module(backend_module).require_symbol("squeeze")(a, axis=axis)
        for a in adapt_backend_module(backend_module).require_symbol("split")(
            x, adapt_backend_module(x).require_symbol("shape")[axis], axis=axis
        )
    ]


def _np_dynamic_slice(
    backend_module: BackendModuleProtocol | types.ModuleType | None,
    x: BackendArg,
    start_indices: BackendArg,
    slice_sizes: BackendArg,
) -> BackendArg:
    """Execute the eager np dynamic slice operation on the active backend array.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None):
            Input `backend_module` argument (backend module).
        x (BackendArg): Input `x` argument (x).
        start_indices (BackendArg): Input `start_indices` argument (start indices).
        slice_sizes (BackendArg): Input `slice_sizes` argument (slice sizes).

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    slices: tuple[slice[int | None, int | None, int | None], ...] = tuple(
        slice(coerce_int(start), coerce_int(start) + coerce_int(size))
        for (start, size) in zip(coerce_operand_list(start_indices), coerce_operand_list(slice_sizes))
    )
    return ensure_ndarray(x)[slices]


@numpy_eager_registry.register("DynamicSliceInDim")
def _np_dynamic_slice_in_dim(
    backend_module: BackendModuleProtocol | types.ModuleType | None,
    operand: BackendArg,
    context: IndexingContext,
    *args: BackendArg,
    **kwargs: BackendArg,
) -> BackendArg:
    """Execute the eager np dynamic slice in dim operation on the active backend array.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None):
            Input `backend_module` argument (backend module).
        operand (BackendArg): Input `operand` argument (operand).
        context (IndexingContext): Input `context` argument (context).
        *args (BackendArg): Positional args.
        **kwargs (BackendArg): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    operand_arr = np.asarray(operand)
    start_index: int = int(np.asarray(context.start_index).item())
    sl: list[slice[int | None, int | None, int | None]] = [slice(None)] * operand_arr.ndim
    sl[context.axis] = slice(start_index, start_index + int(context.slice_size or 0))
    return operand_arr[tuple(sl)]


@numpy_eager_registry.register("DynamicUpdateSliceInDim")
def _np_dynamic_update_slice_in_dim(
    backend_module: BackendModuleProtocol | types.ModuleType | None,
    operand: BackendArg,
    update: BackendArg,
    context: IndexingContext,
    *args: BackendArg,
    **kwargs: BackendArg,
) -> BackendArg:
    """Execute the eager np dynamic update slice in dim operation on the active backend array.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None):
            Input `backend_module` argument (backend module).
        operand (BackendArg): Input `operand` argument (operand).
        update (BackendArg): Input `update` argument (update).
        context (IndexingContext): Input `context` argument (context).
        *args (BackendArg): Positional args.
        **kwargs (BackendArg): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    operand_arr = adapt_backend_module(np).require_op("copy")(np.asarray(operand))
    start_index: int = int(np.asarray(context.start_index).item())
    slice_size: int = np.asarray(update).shape[context.axis]
    sl: list[slice[int | None, int | None, int | None]] = [slice(None)] * operand_arr.ndim
    sl[context.axis] = slice(start_index, start_index + slice_size)
    operand_arr[tuple(sl)] = update
    return operand_arr


@numpy_eager_registry.register("DynamicIndexInDim")
def _np_dynamic_index_in_dim(
    backend_module: BackendModuleProtocol | types.ModuleType | None,
    operand: BackendArg,
    index: BackendArg,
    context: IndexingContext,
    *args: BackendArg,
    **kwargs: BackendArg,
) -> BackendArg:
    """Execute the eager np dynamic index in dim operation on the active backend array.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None):
            Input `backend_module` argument (backend module).
        operand (BackendArg): Input `operand` argument (operand).
        index (BackendArg): Input `index` argument (index).
        context (IndexingContext): Input `context` argument (context).
        *args (BackendArg): Positional args.
        **kwargs (BackendArg): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    operand_arr = np.asarray(operand)
    idx: int = int(np.asarray(index).item())
    if context.keepdims:
        sl_keep: list[slice[int | None, int | None, int | None]] = [slice(None)] * operand_arr.ndim
        sl_keep[context.axis] = slice(idx, idx + 1)
        return operand_arr[tuple(sl_keep)]
    else:
        sl_drop: list[slice[int | None, int | None, int | None] | int] = [slice(None)] * operand_arr.ndim
        sl_drop[context.axis] = idx
        return operand_arr[tuple(sl_drop)]


@numpy_eager_registry.register("DynamicUpdateIndexInDim")
def _np_dynamic_update_index_in_dim(
    backend_module: BackendModuleProtocol | types.ModuleType | None,
    target: IndexTarget,
    context: IndexingContext,
    *args: BackendArg,
    **kwargs: BackendArg,
) -> BackendArg:
    """Execute the eager np dynamic update index in dim operation on the active backend array.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None):
            Input `backend_module` argument (backend module).
        target (IndexTarget): Input `target` argument (target).
        context (IndexingContext): Input `context` argument (context).
        *args (BackendArg): Positional args.
        **kwargs (BackendArg): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    operand, update, index = target.operand, target.update, target.index
    operand_arr = adapt_backend_module(np).require_op("copy")(np.asarray(operand))
    idx: int = int(np.asarray(index).item())
    sl: list[slice[int | None, int | None, int | None] | int] = [slice(None)] * operand_arr.ndim
    sl[context.axis] = idx
    operand_arr[tuple(sl)] = update
    return operand_arr


@numpy_eager_registry.register("SliceInDim")
def _np_slice_in_dim(
    backend_module: BackendModuleProtocol | types.ModuleType | None,
    operand: BackendArg,
    context: IndexingContext,
    *args: BackendArg,
    **kwargs: BackendArg,
) -> BackendArg:
    """Execute the eager np slice in dim operation on the active backend array.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None):
            Input `backend_module` argument (backend module).
        operand (BackendArg): Input `operand` argument (operand).
        context (IndexingContext): Input `context` argument (context).
        *args (BackendArg): Positional args.
        **kwargs (BackendArg): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    operand_arr = np.asarray(operand)
    sl: list[slice[int | None, int | None, int | None]] = [slice(None)] * operand_arr.ndim
    sl[context.axis] = slice(coerce_opt_int(context.start_index), coerce_opt_int(context.limit_index), context.stride)
    return operand_arr[tuple(sl)]


@numpy_eager_registry.register("Slice")
def _np_slice(
    backend_module: BackendModuleProtocol | types.ModuleType | None, x: BackendArg, context: IndexingContext
) -> BackendArg:
    """Execute the eager np slice operation on the active backend array.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None):
            Input `backend_module` argument (backend module).
        x (BackendArg): Input `x` argument (x).
        context (IndexingContext): Input `context` argument (context).

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    sl: list[slice[int | None, int | None, int | None]] = [slice(None)] * coerce_int(
        adapt_backend_module(x).require_symbol("ndim")
    )
    sl[context.axis] = slice(coerce_opt_int(context.start_index), coerce_opt_int(context.limit_index), context.stride)
    return ensure_ndarray(x)[tuple(sl)]


@numpy_eager_registry.register("GetItem")
def _np_getitem(backend_module: BackendModuleProtocol | types.ModuleType | None, x: BackendArg, key: str) -> BackendArg:
    """Execute the eager np getitem operation on the active backend array.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None):
            Input `backend_module` argument (backend module).
        x (BackendArg): Input `x` argument (x).
        key (str): Input `key` argument (key).

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    parsed_key = _safe_parse_key(key)
    return adapt_backend_module(backend_module).require_symbol("asarray")(x)[parsed_key]


@numpy_eager_registry.register("SetItem")
def _np_setitem(
    backend_module: BackendModuleProtocol | types.ModuleType | None, x: BackendArg, value: BackendArg, key: str
) -> BackendArg:
    """Execute the eager np setitem operation on the active backend array.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None):
            Input `backend_module` argument (backend module).
        x (BackendArg): Input `x` argument (x).
        value (BackendArg): Input `value` argument (value).
        key (str): Input `key` argument (key).

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    parsed_key: np.ndarray[tuple[int, ...], np.dtype[np.float64]] = ensure_ndarray(_safe_parse_key(key))
    out: np.ndarray[tuple[int, ...], np.dtype[np.float64]] = np.copy(np.asarray(x))
    out[parsed_key] = np.asarray(value)
    return out


@numpy_eager_registry.register("IndexInDim")
def _eager_indexindim(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager eager indexindim operation on the active backend array.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None):
            Input `backend_module` argument (backend module).
        *args (BackendArg): Positional args.
        **kwargs (BackendArg): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    import numpy as np

    x, idx, dim = args[0], args[1], args[2]
    return adapt_backend_module(np).require_op("take")(x, idx, axis=dim)


@numpy_eager_registry.register("Gather")
def gather_eager(
    np_mod: BackendModuleProtocol | types.ModuleType,
    *args: BackendArg,
    **kwargs: BackendArg,
) -> BackendArg:
    """Execute the eager gather eager operation on the active backend array.

    Args:
        np_mod (BackendModuleProtocol | types.ModuleType): Input `np_mod` argument (np mod).
        *args (BackendArg): Positional args.
        **kwargs (BackendArg): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    t = args[0] if len(args) > 0 else kwargs.get("tensor")
    dim: int | None = coerce_opt_int(args[1] if len(args) > 1 else kwargs.get("dim"))
    raw_idx = args[2] if len(args) > 2 else kwargs.get("index")
    if adapt_backend_module(t).has_symbol("numpy"):
        t = adapt_backend_module(t).require_symbol("numpy")()
    if adapt_backend_module(raw_idx).has_symbol("numpy"):
        raw_idx = adapt_backend_module(raw_idx).require_symbol("numpy")()
    index: BackendArg = ensure_ndarray(raw_idx) if raw_idx is not None else None
    return adapt_backend_module(np_mod).require_symbol("take_along_axis")(t, index, axis=dim)


@numpy_eager_registry.register("Stack")
def stack_eager(
    np_mod: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager stack eager operation on the active backend array.

    Args:
        np_mod (BackendModuleProtocol | types.ModuleType | None): Input `np_mod` argument (np mod).
        *args (BackendArg): Positional args.
        **kwargs (BackendArg): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    tensors: list[BackendArg] = coerce_operand_list(args[0] if len(args) > 0 else (kwargs.get("tensors") or []))
    dim: int = coerce_int(args[1] if len(args) > 1 else kwargs.get("dim", 0))
    if "axis" in kwargs:
        dim = coerce_int(kwargs["axis"])
    arrays: list[BackendArg] = [
        adapt_backend_module(t).require_symbol("numpy")() if adapt_backend_module(t).has_symbol("numpy") else t
        for t in tensors
    ]
    return adapt_backend_module(np_mod).require_symbol("stack")(arrays, axis=dim)
