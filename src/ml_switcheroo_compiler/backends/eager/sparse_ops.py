"""Generic backend-parametric eager reference operations for `sparse_ops`."""

from __future__ import annotations

import types

from ml_switcheroo_compiler.backends.eager.utils import _resolve_eager_mod
from ml_switcheroo_compiler.backends.eager_registry import (
    BackendArg,
    coerce_int,
    global_eager_registry,
)
from ml_switcheroo_compiler.backends.protocols import BackendModuleProtocol, adapt_backend_module


@global_eager_registry.register("SparseBincount")
def _np_sparse_bincount(
    backend_module: BackendModuleProtocol | types.ModuleType | None, x: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Implement sparse bincount in Numpy.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None):
            Input `backend_module` argument (backend module).
        x (BackendArg): Input `x` argument (x).
        **kwargs (BackendArg): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    arr_mod = _resolve_eager_mod(backend_module)
    return arr_mod.bincount(arr_mod.asarray(x).astype(int).flatten())


@global_eager_registry.register("SparseReduceMax")
def _np_sparse_reduce_max(
    backend_module: BackendModuleProtocol | types.ModuleType | None, x: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Implement sparse reduce max in Numpy.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None):
            Input `backend_module` argument (backend module).
        x (BackendArg): Input `x` argument (x).
        **kwargs (BackendArg): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    arr_mod = _resolve_eager_mod(backend_module)
    return arr_mod.max(arr_mod.asarray(x))


@global_eager_registry.register("SparseReduceSum")
def _np_sparse_reduce_sum(
    backend_module: BackendModuleProtocol | types.ModuleType | None, x: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Implement sparse reduce sum in Numpy.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None):
            Input `backend_module` argument (backend module).
        x (BackendArg): Input `x` argument (x).
        **kwargs (BackendArg): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    arr_mod = _resolve_eager_mod(backend_module)
    return arr_mod.sum(arr_mod.asarray(x))


@global_eager_registry.register("SparseSegmentMean")
def _np_sparse_segment_mean(
    backend_module: BackendModuleProtocol | types.ModuleType | None,
    data: BackendArg,
    indices: BackendArg,
    segment_ids: BackendArg,
    **kwargs: BackendArg,
) -> BackendArg:
    """Implement sparse segment mean in Numpy.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None):
            Input `backend_module` argument (backend module).
        data (BackendArg): Input `data` argument (data).
        indices (BackendArg): Input `indices` argument (indices).
        segment_ids (BackendArg): Input `segment_ids` argument (segment ids).
        **kwargs (BackendArg): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    arr_mod = _resolve_eager_mod(backend_module)
    return arr_mod.mean(arr_mod.asarray(data))


@global_eager_registry.register("SparseSegmentSqrtN")
def _np_sparse_segment_sqrt_n(
    backend_module: BackendModuleProtocol | types.ModuleType | None,
    data: BackendArg,
    indices: BackendArg,
    segment_ids: BackendArg,
    **kwargs: BackendArg,
) -> BackendArg:
    """Implement sparse segment sqrt n in Numpy.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None):
            Input `backend_module` argument (backend module).
        data (BackendArg): Input `data` argument (data).
        indices (BackendArg): Input `indices` argument (indices).
        segment_ids (BackendArg): Input `segment_ids` argument (segment ids).
        **kwargs (BackendArg): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    arr_mod = _resolve_eager_mod(backend_module)
    return arr_mod.sum(arr_mod.asarray(data)) / arr_mod.sqrt(max(1, int(coerce_int(arr_mod.asarray(segment_ids).size))))


@global_eager_registry.register("SparseSegmentSum")
def _np_sparse_segment_sum(
    backend_module: BackendModuleProtocol | types.ModuleType | None,
    data: BackendArg,
    indices: BackendArg,
    segment_ids: BackendArg,
    **kwargs: BackendArg,
) -> BackendArg:
    """Implement sparse segment sum in Numpy.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None):
            Input `backend_module` argument (backend module).
        data (BackendArg): Input `data` argument (data).
        indices (BackendArg): Input `indices` argument (indices).
        segment_ids (BackendArg): Input `segment_ids` argument (segment ids).
        **kwargs (BackendArg): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    arr_mod = _resolve_eager_mod(backend_module)
    return arr_mod.sum(arr_mod.asarray(data))


@global_eager_registry.register("SparseExpandDims")
def _np_sparse_expand_dims(
    backend_module: BackendModuleProtocol | types.ModuleType | None,
    x: BackendArg,
    axis: BackendArg = None,
    **kwargs: BackendArg,
) -> BackendArg:
    """Implement sparse expand dims in Numpy.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None): The backend module.
        x (BackendArg): The input array.
        axis (BackendArg): The axis to expand.
        **kwargs (BackendArg): Keyword arguments.

    Returns: EagerTensorProtocol: The expanded array.
    """
    arr_mod = _resolve_eager_mod(backend_module)
    ax = kwargs.get("axis", axis)
    if ax is None:
        ax = -1
    return adapt_backend_module(arr_mod).require_symbol("expand_dims")(arr_mod.asarray(x), axis=ax)


@global_eager_registry.register("RaggedDot")
def _np_ragged_dot(
    backend_module: BackendModuleProtocol | types.ModuleType | None, a: BackendArg, b: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Implement ragged dot in Numpy.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None):
            Input `backend_module` argument (backend module).
        a (BackendArg): Input `a` argument (a).
        b (BackendArg): Input `b` argument (b).
        **kwargs (BackendArg): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    arr_mod = _resolve_eager_mod(backend_module)
    return arr_mod.dot(arr_mod.asarray(a), arr_mod.asarray(b))
