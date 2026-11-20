"""Generic backend-parametric eager reference operations for `reference_math.math_segment`."""

from __future__ import annotations

import types

from ml_switcheroo_compiler.backends.eager.utils import _resolve_eager_mod
from ml_switcheroo_compiler.backends.eager_registry import (
    BackendArg,
    EagerTensorProtocol,
    global_eager_registry,
)
from ml_switcheroo_compiler.backends.protocols import (
    BackendModuleProtocol,
    adapt_backend_module,
)


def _ensure_arr(val: BackendArg) -> EagerTensorProtocol:
    """Coerce `val` to an eager array via the active backend module.

    Args:
        val (BackendArg): Input array or scalar value.

    Returns:
        EagerTensorProtocol: Coerced backend array.
    """
    return _resolve_eager_mod(None).asarray(val)


@global_eager_registry.register("SegmentSum")
def _np_segment_sum(
    backend_module: BackendModuleProtocol | types.ModuleType | None,
    data: BackendArg,
    segment_ids: BackendArg,
    num_segments: int | None = None,
    **kwargs: int | float | str | None,
) -> EagerTensorProtocol:
    """Execute the eager np segment sum operation on the active backend array.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None): Active backend module.
        data (BackendArg): Data array to sum over segments.
        segment_ids (BackendArg): 1D array indicating segment memberships.
        num_segments (int | None): Number of distinct segments.
        **kwargs (Union[int, float, str, None]): Additional keyword arguments.

    Returns:
        EagerTensorProtocol: Segment-summed result array.
    """
    arr_mod = _resolve_eager_mod(backend_module)
    data_arr = arr_mod.asarray(data)
    seg_arr = arr_mod.asarray(segment_ids)
    n_segments = num_segments if num_segments is not None else int(arr_mod.max(seg_arr)) + 1
    out = arr_mod.zeros((n_segments,) + data_arr.shape[1:], dtype=data_arr.dtype)
    arr_mod.add.at(out, seg_arr, data_arr)
    return _ensure_arr(out)


@global_eager_registry.register("SparseSegmentSum")
def _np_sparsesegmentsum(
    backend_module: BackendModuleProtocol | types.ModuleType | None,
    *args: BackendArg,
    **kwargs: int | float | str | None,
) -> EagerTensorProtocol:
    """Implement SparseSegmentSum.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None): Active backend module.
        *args (BackendArg): Positional input arguments.
        **kwargs (Union[int, float, str, None]): Additional keyword arguments.

    Returns:
        EagerTensorProtocol: Summed tensor result.
    """
    data = adapt_backend_module(backend_module).require_symbol("asarray")(args[0])
    return _ensure_arr(adapt_backend_module(backend_module).require_symbol("sum")(data, axis=0, keepdims=True))
