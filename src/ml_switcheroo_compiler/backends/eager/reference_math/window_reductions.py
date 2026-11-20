"""Generic backend-parametric eager reference operations for `reference_math.window_reductions`."""

from __future__ import annotations

import types

from ml_switcheroo_compiler.backends.eager.reference_math.reductions import _reduce_window, _segment_sum
from ml_switcheroo_compiler.backends.eager.utils import _resolve_eager_mod
from ml_switcheroo_compiler.backends.eager_registry import (
    BackendArg,
    coerce_int,
    global_eager_registry,
)
from ml_switcheroo_compiler.backends.protocols import BackendModuleProtocol, adapt_backend_module
from ml_switcheroo_compiler.core.shape import ShapeProtocol


@global_eager_registry.register("ReduceWindow")
def _np_reduce_window(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager np reduce window operation on the active backend array.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None):
            Input `backend_module` argument (backend module).
        *args (BackendArg): Positional args.
        **kwargs (BackendArg): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    from ml_switcheroo_compiler.backends.eager_registry import require_callable

    return require_callable(_reduce_window)(*args, **kwargs)


@global_eager_registry.register("SegmentSum")
def _np_segment_sum(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager np segment sum operation on the active backend array.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None):
            Input `backend_module` argument (backend module).
        *args (BackendArg): Positional args.
        **kwargs (BackendArg): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    from ml_switcheroo_compiler.backends.eager_registry import require_callable

    return require_callable(_segment_sum)(*args, **kwargs)


@global_eager_registry.register("SegmentMax")
def _np_segment_max(
    backend_module: BackendModuleProtocol | types.ModuleType | None,
    data: BackendArg,
    segment_ids: BackendArg,
    num_segments: BackendArg = None,
) -> BackendArg:
    """Execute the eager np segment max operation on the active backend array.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None):
            Input `backend_module` argument (backend module).
        data (BackendArg): Input `data` argument (data).
        segment_ids (BackendArg): Input `segment_ids` argument (segment ids).
        num_segments (BackendArg): Input `num_segments` argument (num segments).

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    from ml_switcheroo_compiler.backends.eager_registry import coerce_array

    arr_mod = _resolve_eager_mod(backend_module)
    data_arr = coerce_array(arr_mod, data)
    n_seg = int(
        coerce_int(
            adapt_backend_module(arr_mod).require_symbol("max")(segment_ids) + 1
            if num_segments is None
            else num_segments
        )
    )
    out = arr_mod.full(
        (n_seg,) + data_arr.shape[1:],
        -arr_mod.inf,
        dtype=adapt_backend_module(data).require_symbol("dtype"),
    )
    for i in range(n_seg):
        mask = arr_mod.asarray(segment_ids) == i
        if arr_mod.any(mask):
            out[i] = arr_mod.max(data_arr[mask], axis=0)
    return out


@global_eager_registry.register("SegmentMin")
def _np_segment_min(
    backend_module: BackendModuleProtocol | types.ModuleType | None,
    data: BackendArg,
    segment_ids: BackendArg,
    num_segments: BackendArg = None,
) -> BackendArg:
    """Execute the eager np segment min operation on the active backend array.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None):
            Input `backend_module` argument (backend module).
        data (BackendArg): Input `data` argument (data).
        segment_ids (BackendArg): Input `segment_ids` argument (segment ids).
        num_segments (BackendArg): Input `num_segments` argument (num segments).

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    from ml_switcheroo_compiler.backends.eager_registry import coerce_array

    arr_mod = _resolve_eager_mod(backend_module)
    data_arr = coerce_array(arr_mod, data)
    n_seg = int(
        coerce_int(
            adapt_backend_module(arr_mod).require_symbol("max")(segment_ids) + 1
            if num_segments is None
            else num_segments
        )
    )
    out = arr_mod.full(
        (n_seg,) + data_arr.shape[1:],
        arr_mod.inf,
        dtype=adapt_backend_module(data).require_symbol("dtype"),
    )
    for i in range(n_seg):
        mask = arr_mod.asarray(segment_ids) == i
        if arr_mod.any(mask):
            out[i] = arr_mod.min(data_arr[mask], axis=0)
    return out


@global_eager_registry.register("SegmentProd")
def _np_segment_prod(
    backend_module: BackendModuleProtocol | types.ModuleType | None,
    data: BackendArg,
    segment_ids: BackendArg,
    num_segments: BackendArg = None,
) -> BackendArg:
    """Execute the eager np segment prod operation on the active backend array.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None):
            Input `backend_module` argument (backend module).
        data (BackendArg): Input `data` argument (data).
        segment_ids (BackendArg): Input `segment_ids` argument (segment ids).
        num_segments (BackendArg): Input `num_segments` argument (num segments).

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    from ml_switcheroo_compiler.backends.eager_registry import coerce_array

    arr_mod = _resolve_eager_mod(backend_module)
    data_arr = coerce_array(arr_mod, data)
    n_seg = int(
        coerce_int(
            adapt_backend_module(arr_mod).require_symbol("max")(segment_ids) + 1
            if num_segments is None
            else num_segments
        )
    )
    out = arr_mod.ones(
        (n_seg,) + data_arr.shape[1:],
        dtype=adapt_backend_module(data).require_symbol("dtype"),
    )
    for i in range(n_seg):
        mask = arr_mod.asarray(segment_ids) == i
        if arr_mod.any(mask):
            out[i] = arr_mod.prod(data_arr[mask], axis=0)
    return out


def _adaptive_pool_1d_indices(input_dim: int, output_dim: int) -> list[tuple[int, int]]:
    """Execute the eager adaptive pool 1d indices operation on the active backend array.

    Args:
        input_dim (int): Input `input_dim` argument (input dim).
        output_dim (int): Input `output_dim` argument (output dim).

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    arr_mod = _resolve_eager_mod(None)
    indices = []
    for i in range(output_dim):
        start = int(arr_mod.floor(i * input_dim / output_dim))
        end = int(arr_mod.ceil((i + 1) * input_dim / output_dim))
        start = max(0, min(start, input_dim - 1))
        end = max(start + 1, min(end, input_dim))
        indices.append((start, end))
    return indices


@global_eager_registry.register("AdaptiveAvgPool2D")
def _np_adaptive_avg_pool2d(
    backend_module: BackendModuleProtocol | types.ModuleType | None,
    operand: BackendArg,
    output_size: tuple[int, int],
    **kwargs: BackendArg,
) -> BackendArg:
    """Execute the eager np adaptive avg pool2d operation on the active backend array.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None):
            Input `backend_module` argument (backend module).
        operand (BackendArg): Input `operand` argument (operand).
        output_size (tuple[int, int]): Input `output_size` argument (output size).
        **kwargs (BackendArg): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    arr_mod = _resolve_eager_mod(backend_module)
    if isinstance(operand, ShapeProtocol):
        operand = arr_mod.asarray(operand)
        sh = list(operand.shape)
        h_in, w_in = sh[-2], sh[-1]
        h_out, w_out = output_size[0], output_size[1]

        h_indices = _adaptive_pool_1d_indices(h_in, h_out)
        w_indices = _adaptive_pool_1d_indices(w_in, w_out)

        out_shape = sh[:-2] + [h_out, w_out]
        out = arr_mod.zeros(out_shape, dtype=operand.dtype)

        for i, (h_start, h_end) in enumerate(h_indices):
            for j, (w_start, w_end) in enumerate(w_indices):
                slice_val = operand[..., h_start:h_end, w_start:w_end]
                out[..., i, j] = arr_mod.mean(slice_val, axis=(-2, -1))
        return out
    return operand


@global_eager_registry.register("AdaptiveAvgPool3D")
def _np_adaptive_avg_pool3d(
    backend_module: BackendModuleProtocol | types.ModuleType | None,
    operand: BackendArg,
    output_size: tuple[int, int, int],
    **kwargs: BackendArg,
) -> BackendArg:
    """Execute the eager np adaptive avg pool3d operation on the active backend array.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None):
            Input `backend_module` argument (backend module).
        operand (BackendArg): Input `operand` argument (operand).
        output_size (tuple[int, int, int]): Input `output_size` argument (output size).
        **kwargs (BackendArg): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    arr_mod = _resolve_eager_mod(backend_module)
    if isinstance(operand, ShapeProtocol):
        operand = arr_mod.asarray(operand)
        sh = list(operand.shape)
        d_in, h_in, w_in = sh[-3], sh[-2], sh[-1]
        d_out, h_out, w_out = output_size[0], output_size[1], output_size[2]

        d_indices = _adaptive_pool_1d_indices(d_in, d_out)
        h_indices = _adaptive_pool_1d_indices(h_in, h_out)
        w_indices = _adaptive_pool_1d_indices(w_in, w_out)

        out_shape = sh[:-3] + [d_out, h_out, w_out]
        out = arr_mod.zeros(out_shape, dtype=operand.dtype)

        for d_idx, (d_start, d_end) in enumerate(d_indices):
            for h_idx, (h_start, h_end) in enumerate(h_indices):
                for w_idx, (w_start, w_end) in enumerate(w_indices):
                    slice_val = operand[..., d_start:d_end, h_start:h_end, w_start:w_end]
                    out[..., d_idx, h_idx, w_idx] = arr_mod.mean(slice_val, axis=(-3, -2, -1))
        return out
    return operand


@global_eager_registry.register("AdaptiveMaxPool3D")
def _np_adaptive_max_pool3d(
    backend_module: BackendModuleProtocol | types.ModuleType | None,
    operand: BackendArg,
    output_size: tuple[int, int, int],
    **kwargs: BackendArg,
) -> BackendArg:
    """Execute the eager np adaptive max pool3d operation on the active backend array.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None):
            Input `backend_module` argument (backend module).
        operand (BackendArg): Input `operand` argument (operand).
        output_size (tuple[int, int, int]): Input `output_size` argument (output size).
        **kwargs (BackendArg): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    arr_mod = _resolve_eager_mod(backend_module)
    if isinstance(operand, ShapeProtocol):
        operand = arr_mod.asarray(operand)
        sh = list(operand.shape)
        d_in, h_in, w_in = sh[-3], sh[-2], sh[-1]
        d_out, h_out, w_out = output_size[0], output_size[1], output_size[2]

        d_indices = _adaptive_pool_1d_indices(d_in, d_out)
        h_indices = _adaptive_pool_1d_indices(h_in, h_out)
        w_indices = _adaptive_pool_1d_indices(w_in, w_out)

        out_shape = sh[:-3] + [d_out, h_out, w_out]
        out = arr_mod.zeros(out_shape, dtype=operand.dtype)

        for d_idx, (d_start, d_end) in enumerate(d_indices):
            for h_idx, (h_start, h_end) in enumerate(h_indices):
                for w_idx, (w_start, w_end) in enumerate(w_indices):
                    slice_val = operand[..., d_start:d_end, h_start:h_end, w_start:w_end]
                    out[..., d_idx, h_idx, w_idx] = arr_mod.max(slice_val, axis=(-3, -2, -1))
        return out
    return operand


@global_eager_registry.register("AdaptiveMaxPool3D_Indices")
def _np_adaptive_max_pool3d_indices(
    backend_module: BackendModuleProtocol | types.ModuleType | None,
    operand: BackendArg,
    output_size: tuple[int, int, int],
    **kwargs: BackendArg,
) -> BackendArg:
    """Execute the eager np adaptive max pool3d indices operation on the active backend array.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None):
            Input `backend_module` argument (backend module).
        operand (BackendArg): Input `operand` argument (operand).
        output_size (tuple[int, int, int]): Input `output_size` argument (output size).
        **kwargs (BackendArg): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    arr_mod = _resolve_eager_mod(backend_module)
    res = _np_adaptive_max_pool3d(backend_module, operand, output_size, **kwargs)
    return (res, arr_mod.zeros_like(res, dtype=arr_mod.int64))


@global_eager_registry.register("AdaptiveMaxPool2D")
def _np_adaptive_max_pool2d(
    backend_module: BackendModuleProtocol | types.ModuleType | None,
    operand: BackendArg,
    output_size: tuple[int, int],
    **kwargs: BackendArg,
) -> BackendArg:
    """Execute the eager np adaptive max pool2d operation on the active backend array.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None):
            Input `backend_module` argument (backend module).
        operand (BackendArg): Input `operand` argument (operand).
        output_size (tuple[int, int]): Input `output_size` argument (output size).
        **kwargs (BackendArg): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    arr_mod = _resolve_eager_mod(backend_module)
    if isinstance(operand, ShapeProtocol):
        operand = arr_mod.asarray(operand)
        sh = list(operand.shape)
        h_in, w_in = sh[-2], sh[-1]
        h_out, w_out = output_size[0], output_size[1]

        h_indices = _adaptive_pool_1d_indices(h_in, h_out)
        w_indices = _adaptive_pool_1d_indices(w_in, w_out)

        out_shape = sh[:-2] + [h_out, w_out]
        out = arr_mod.zeros(out_shape, dtype=operand.dtype)

        for i, (h_start, h_end) in enumerate(h_indices):
            for j, (w_start, w_end) in enumerate(w_indices):
                slice_val = operand[..., h_start:h_end, w_start:w_end]
                out[..., i, j] = arr_mod.max(slice_val, axis=(-2, -1))
        return out
    return operand


@global_eager_registry.register("FractionalAvgPool")
def _np_fractional_avg_pool(
    backend_module: BackendModuleProtocol | types.ModuleType | None, value: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager np fractional avg pool operation on the active backend array.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None):
            Input `backend_module` argument (backend module).
        value (BackendArg): Input `value` argument (value).
        **kwargs (BackendArg): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    arr_mod = _resolve_eager_mod(backend_module)
    return arr_mod.asarray(value)


@global_eager_registry.register("FractionalMaxPool")
def _np_fractional_max_pool(
    backend_module: BackendModuleProtocol | types.ModuleType | None, value: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager np fractional max pool operation on the active backend array.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None):
            Input `backend_module` argument (backend module).
        value (BackendArg): Input `value` argument (value).
        **kwargs (BackendArg): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    arr_mod = _resolve_eager_mod(backend_module)
    return arr_mod.asarray(value)
