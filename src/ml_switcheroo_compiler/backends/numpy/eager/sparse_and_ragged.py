"""Numpy implementations for sparse and ragged ops."""

from __future__ import annotations

from types import ModuleType
from typing import cast

import numpy as np

from ml_switcheroo_compiler.backends.eager_registry import EagerValue, numpy_eager_registry


@numpy_eager_registry.register("SparseBincount")
def _np_sparse_bincount(backend_module: ModuleType, x: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Implement sparse bincount in Numpy.

    Args:
        backend_module (ModuleType): The backend_module parameter.
        x (EagerValue): The x parameter.
        **kwargs (EagerValue): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    return np.bincount(np.asarray(x).astype(int).flatten())


@numpy_eager_registry.register("SparseReduceMax")
def _np_sparse_reduce_max(backend_module: ModuleType, x: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Implement sparse reduce max in Numpy.

    Args:
        backend_module (ModuleType): The backend_module parameter.
        x (EagerValue): The x parameter.
        **kwargs (EagerValue): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    return cast(EagerValue, np.max(np.asarray(x)))


@numpy_eager_registry.register("SparseReduceSum")
def _np_sparse_reduce_sum(backend_module: ModuleType, x: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Implement sparse reduce sum in Numpy.

    Args:
        backend_module (ModuleType): The backend_module parameter.
        x (EagerValue): The x parameter.
        **kwargs (EagerValue): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    return cast(EagerValue, np.sum(np.asarray(x)))


@numpy_eager_registry.register("SparseSegmentMean")
def _np_sparse_segment_mean(backend_module: ModuleType, data: EagerValue, indices: EagerValue, segment_ids: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Implement sparse segment mean in Numpy.

    Args:
        backend_module (ModuleType): The backend_module parameter.
        data (EagerValue): The data parameter.
        indices (EagerValue): The indices parameter.
        segment_ids (EagerValue): The segment_ids parameter.
        **kwargs (EagerValue): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    return cast(EagerValue, np.mean(np.asarray(data)))


@numpy_eager_registry.register("SparseSegmentSqrtN")
def _np_sparse_segment_sqrt_n(backend_module: ModuleType, data: EagerValue, indices: EagerValue, segment_ids: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Implement sparse segment sqrt n in Numpy.

    Args:
        backend_module (ModuleType): The backend_module parameter.
        data (EagerValue): The data parameter.
        indices (EagerValue): The indices parameter.
        segment_ids (EagerValue): The segment_ids parameter.
        **kwargs (EagerValue): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    return cast(EagerValue, np.sum(np.asarray(data)) / np.sqrt(max(1, np.asarray(segment_ids).size)))


@numpy_eager_registry.register("SparseSegmentSum")
def _np_sparse_segment_sum(backend_module: ModuleType, data: EagerValue, indices: EagerValue, segment_ids: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Implement sparse segment sum in Numpy.

    Args:
        backend_module (ModuleType): The backend_module parameter.
        data (EagerValue): The data parameter.
        indices (EagerValue): The indices parameter.
        segment_ids (EagerValue): The segment_ids parameter.
        **kwargs (EagerValue): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    return cast(EagerValue, np.sum(np.asarray(data)))


@numpy_eager_registry.register("SparseExpandDims")
def _np_sparse_expand_dims(backend_module: ModuleType, x: EagerValue, axis: EagerValue = None, **kwargs: EagerValue) -> EagerValue:
    """Implement sparse expand dims in Numpy.

    Args:
        backend_module (ModuleType): The backend module.
        x (EagerValue): The input array.
        axis (int | None): The axis to expand.
        **kwargs (EagerValue): Keyword arguments.

    Returns: np.ndarray: The expanded array.
    """
    import numpy as np

    ax = kwargs.get("axis", axis)
    if ax is None:
        ax = -1
    return cast(EagerValue, getattr(np, "expand_dims")(np.asarray(x), axis=ax))


@numpy_eager_registry.register("RaggedDot")
def _np_ragged_dot(backend_module: ModuleType, a: EagerValue, b: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Implement ragged dot in Numpy.

    Args:
        backend_module (ModuleType): The backend_module parameter.
        a (EagerValue): The a parameter.
        b (EagerValue): The b parameter.
        **kwargs (EagerValue): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    return cast(EagerValue, np.dot(np.asarray(a), np.asarray(b)))
