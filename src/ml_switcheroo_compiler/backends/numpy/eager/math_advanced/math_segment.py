"""Math Ops Segment."""

from __future__ import annotations

import numpy as np

from ml_switcheroo_compiler.backends.eager_registry import EagerValue, numpy_eager_registry
from ml_switcheroo_compiler.backends.numpy.eager import as_ndarray
from ml_switcheroo_compiler.backends.protocols import BackendModuleProtocol


@numpy_eager_registry.register("SegmentSum")
def _np_segment_sum(
    backend_module: BackendModuleProtocol,
    data: np.ndarray[tuple[int, ...], np.dtype[np.float64]],
    segment_ids: np.ndarray[tuple[int, ...], np.dtype[np.int64]],
    num_segments: int | None = None,
    **kwargs: int | float | str | None,
) -> np.ndarray[tuple[int, ...], np.dtype[np.float64]]:
    """Execute the eager np segment sum operation on the active backend array.

    Args:
        backend_module (ModuleType): Active backend module.
        data (np.ndarray): Data array to sum over segments.
        segment_ids (np.ndarray): 1D array indicating segment memberships.
        num_segments (int | None): Number of distinct segments.
        **kwargs (Union[int, float, str, None]): Additional keyword arguments.

    Returns:
        np.ndarray: Segment-summed result array.
    """
    n_segments = num_segments if num_segments is not None else int(np.max(segment_ids)) + 1
    out = np.zeros((n_segments,) + data.shape[1:], dtype=data.dtype)
    np.add.at(out, segment_ids, data)
    return out


@numpy_eager_registry.register("SparseSegmentSum")
def _np_sparsesegmentsum(
    backend_module: BackendModuleProtocol,
    *args: EagerValue,
    **kwargs: int | float | str | None,
) -> np.ndarray[tuple[int, ...], np.dtype[np.float64]]:
    """Implement SparseSegmentSum.

    Args:
        backend_module (ModuleType): Active backend module.
        *args (Union[np.ndarray, list[float], list[int]]): Positional input arguments.
        **kwargs (Union[int, float, str, None]): Additional keyword arguments.

    Returns:
        np.ndarray: Summed tensor result.
    """
    data = backend_module.asarray(args[0])
    return as_ndarray(backend_module.sum(data, axis=0, keepdims=True))
