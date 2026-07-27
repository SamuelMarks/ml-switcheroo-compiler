"""Numpy implementations for sparse and ragged ops."""

import numpy as np

from ml_switcheroo_compiler.backends.eager_registry import numpy_eager_registry


@numpy_eager_registry.register("SparseBincount")
def _np_sparse_bincount(backend_module: object, x: object, **kwargs: object) -> object:
    """Implement sparse bincount in Numpy."""
    return np.bincount(np.asarray(x).astype(int).flatten())


@numpy_eager_registry.register("SparseReduceMax")
def _np_sparse_reduce_max(backend_module: object, x: object, **kwargs: object) -> object:
    """Implement sparse reduce max in Numpy."""
    return np.max(np.asarray(x))


@numpy_eager_registry.register("SparseReduceSum")
def _np_sparse_reduce_sum(backend_module: object, x: object, **kwargs: object) -> object:
    """Implement sparse reduce sum in Numpy."""
    return np.sum(np.asarray(x))


@numpy_eager_registry.register("SparseSegmentMean")
def _np_sparse_segment_mean(backend_module: object, data: object, indices: object, segment_ids: object, **kwargs: object) -> object:
    """Implement sparse segment mean in Numpy."""
    return np.mean(np.asarray(data))


@numpy_eager_registry.register("SparseSegmentSqrtN")
def _np_sparse_segment_sqrt_n(backend_module: object, data: object, indices: object, segment_ids: object, **kwargs: object) -> object:
    """Implement sparse segment sqrt n in Numpy."""
    return np.sum(np.asarray(data)) / np.sqrt(max(1, np.asarray(segment_ids).size))


@numpy_eager_registry.register("SparseSegmentSum")
def _np_sparse_segment_sum(backend_module: object, data: object, indices: object, segment_ids: object, **kwargs: object) -> object:
    """Implement sparse segment sum in Numpy."""
    return np.sum(np.asarray(data))


@numpy_eager_registry.register("RaggedDot")
def _np_ragged_dot(backend_module: object, a: object, b: object, **kwargs: object) -> object:
    """Implement ragged dot in Numpy."""
    return np.dot(np.asarray(a), np.asarray(b))
