"""Numpy Window Reductions."""

from __future__ import annotations

import numpy as np

from ml_switcheroo_compiler.backends.eager_registry import EagerValue, ensure_callable, ensure_eager_tensor, ensure_int, numpy_eager_registry
from ml_switcheroo_compiler.backends.numpy.eager.reductions import _reduce_window, _segment_sum
from ml_switcheroo_compiler.backends.protocols import BackendModuleProtocol, get_backend_member


@numpy_eager_registry.register("ReduceWindow")
def _np_reduce_window(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager np reduce window operation on the active backend array.

    Args:
        backend_module (ModuleType): Input `backend_module` argument (backend module).
        *args (EagerValue): Positional args.
        **kwargs (EagerValue): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    return ensure_callable(_reduce_window)(*args, **kwargs)


@numpy_eager_registry.register("SegmentSum")
def _np_segment_sum(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager np segment sum operation on the active backend array.

    Args:
        backend_module (ModuleType): Input `backend_module` argument (backend module).
        *args (EagerValue): Positional args.
        **kwargs (EagerValue): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    return ensure_callable(_segment_sum)(*args, **kwargs)


@numpy_eager_registry.register("SegmentMax")
def _np_segment_max(backend_module: BackendModuleProtocol, data: EagerValue, segment_ids: EagerValue, num_segments: EagerValue = None) -> EagerValue:
    """Execute the eager np segment max operation on the active backend array.

    Args:
        backend_module (ModuleType): Input `backend_module` argument (backend module).
        data (EagerValue): Input `data` argument (data).
        segment_ids (EagerValue): Input `segment_ids` argument (segment ids).
        num_segments (EagerValue): Input `num_segments` argument (num segments).

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    if num_segments is None:
        num_segments = get_backend_member(np, "max")(segment_ids) + 1
    out = np.full((num_segments,) + get_backend_member(data, "shape")[1:], -np.inf, dtype=get_backend_member(data, "dtype"))
    for i in range(ensure_int(num_segments)):
        mask = segment_ids == i
        if np.any(mask):
            ensure_eager_tensor(out)[i] = np.max(ensure_eager_tensor(data)[mask], axis=0)
    return out


@numpy_eager_registry.register("SegmentMin")
def _np_segment_min(backend_module: BackendModuleProtocol, data: EagerValue, segment_ids: EagerValue, num_segments: EagerValue = None) -> EagerValue:
    """Execute the eager np segment min operation on the active backend array.

    Args:
        backend_module (ModuleType): Input `backend_module` argument (backend module).
        data (EagerValue): Input `data` argument (data).
        segment_ids (EagerValue): Input `segment_ids` argument (segment ids).
        num_segments (EagerValue): Input `num_segments` argument (num segments).

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    if num_segments is None:
        num_segments = get_backend_member(np, "max")(segment_ids) + 1
    out = np.full((num_segments,) + get_backend_member(data, "shape")[1:], np.inf, dtype=get_backend_member(data, "dtype"))
    for i in range(ensure_int(num_segments)):
        mask = segment_ids == i
        if np.any(mask):
            ensure_eager_tensor(out)[i] = np.min(ensure_eager_tensor(data)[mask], axis=0)
    return out


@numpy_eager_registry.register("SegmentProd")
def _np_segment_prod(backend_module: BackendModuleProtocol, data: EagerValue, segment_ids: EagerValue, num_segments: EagerValue = None) -> EagerValue:
    """Execute the eager np segment prod operation on the active backend array.

    Args:
        backend_module (ModuleType): Input `backend_module` argument (backend module).
        data (EagerValue): Input `data` argument (data).
        segment_ids (EagerValue): Input `segment_ids` argument (segment ids).
        num_segments (EagerValue): Input `num_segments` argument (num segments).

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    if num_segments is None:
        num_segments = get_backend_member(np, "max")(segment_ids) + 1
    out = np.ones((num_segments,) + get_backend_member(data, "shape")[1:], dtype=get_backend_member(data, "dtype"))
    for i in range(ensure_int(num_segments)):
        mask = segment_ids == i
        if np.any(mask):
            ensure_eager_tensor(out)[i] = np.prod(ensure_eager_tensor(data)[mask], axis=0)
    return out


def _adaptive_pool_1d_indices(input_dim: int, output_dim: int) -> list[tuple[int, int]]:
    """Execute the eager adaptive pool 1d indices operation on the active backend array.

    Args:
        input_dim (int): Input `input_dim` argument (input dim).
        output_dim (int): Input `output_dim` argument (output dim).

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    indices = []
    for i in range(output_dim):
        start = int(np.floor(i * input_dim / output_dim))
        end = int(np.ceil((i + 1) * input_dim / output_dim))
        start = max(0, min(start, input_dim - 1))
        end = max(start + 1, min(end, input_dim))
        indices.append((start, end))
    return indices


@numpy_eager_registry.register("AdaptiveAvgPool2D")
def _np_adaptive_avg_pool2d(backend_module: BackendModuleProtocol, operand: EagerValue, output_size: tuple[int, int], **kwargs: EagerValue) -> EagerValue:
    """Execute the eager np adaptive avg pool2d operation on the active backend array.

    Args:
        backend_module (ModuleType): Input `backend_module` argument (backend module).
        operand (EagerValue): Input `operand` argument (operand).
        output_size (tuple[int, int]): Input `output_size` argument (output size).
        **kwargs (EagerValue): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    if isinstance(operand, np.ndarray):
        sh = list(operand.shape)
        h_in, w_in = sh[-2], sh[-1]
        h_out, w_out = output_size[0], output_size[1]

        h_indices = _adaptive_pool_1d_indices(h_in, h_out)
        w_indices = _adaptive_pool_1d_indices(w_in, w_out)

        out_shape = sh[:-2] + [h_out, w_out]
        out = np.zeros(out_shape, dtype=operand.dtype)

        for i, (h_start, h_end) in enumerate(h_indices):
            for j, (w_start, w_end) in enumerate(w_indices):
                slice_val = operand[..., h_start:h_end, w_start:w_end]
                out[..., i, j] = np.mean(slice_val, axis=(-2, -1))
        return out
    return operand


@numpy_eager_registry.register("AdaptiveAvgPool3D")
def _np_adaptive_avg_pool3d(backend_module: BackendModuleProtocol, operand: EagerValue, output_size: tuple[int, int, int], **kwargs: EagerValue) -> EagerValue:
    """Execute the eager np adaptive avg pool3d operation on the active backend array.

    Args:
        backend_module (ModuleType): Input `backend_module` argument (backend module).
        operand (EagerValue): Input `operand` argument (operand).
        output_size (tuple[int, int, int]): Input `output_size` argument (output size).
        **kwargs (EagerValue): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    if isinstance(operand, np.ndarray):
        sh = list(operand.shape)
        d_in, h_in, w_in = sh[-3], sh[-2], sh[-1]
        d_out, h_out, w_out = output_size[0], output_size[1], output_size[2]

        d_indices = _adaptive_pool_1d_indices(d_in, d_out)
        h_indices = _adaptive_pool_1d_indices(h_in, h_out)
        w_indices = _adaptive_pool_1d_indices(w_in, w_out)

        out_shape = sh[:-3] + [d_out, h_out, w_out]
        out = np.zeros(out_shape, dtype=operand.dtype)

        for d_idx, (d_start, d_end) in enumerate(d_indices):
            for h_idx, (h_start, h_end) in enumerate(h_indices):
                for w_idx, (w_start, w_end) in enumerate(w_indices):
                    slice_val = operand[..., d_start:d_end, h_start:h_end, w_start:w_end]
                    out[..., d_idx, h_idx, w_idx] = np.mean(slice_val, axis=(-3, -2, -1))
        return out
    return operand


@numpy_eager_registry.register("AdaptiveMaxPool3D")
def _np_adaptive_max_pool3d(backend_module: BackendModuleProtocol, operand: EagerValue, output_size: tuple[int, int, int], **kwargs: EagerValue) -> EagerValue:
    """Execute the eager np adaptive max pool3d operation on the active backend array.

    Args:
        backend_module (ModuleType): Input `backend_module` argument (backend module).
        operand (EagerValue): Input `operand` argument (operand).
        output_size (tuple[int, int, int]): Input `output_size` argument (output size).
        **kwargs (EagerValue): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    if isinstance(operand, np.ndarray):
        sh = list(operand.shape)
        d_in, h_in, w_in = sh[-3], sh[-2], sh[-1]
        d_out, h_out, w_out = output_size[0], output_size[1], output_size[2]

        d_indices = _adaptive_pool_1d_indices(d_in, d_out)
        h_indices = _adaptive_pool_1d_indices(h_in, h_out)
        w_indices = _adaptive_pool_1d_indices(w_in, w_out)

        out_shape = sh[:-3] + [d_out, h_out, w_out]
        out = np.zeros(out_shape, dtype=operand.dtype)

        for d_idx, (d_start, d_end) in enumerate(d_indices):
            for h_idx, (h_start, h_end) in enumerate(h_indices):
                for w_idx, (w_start, w_end) in enumerate(w_indices):
                    slice_val = operand[..., d_start:d_end, h_start:h_end, w_start:w_end]
                    out[..., d_idx, h_idx, w_idx] = np.max(slice_val, axis=(-3, -2, -1))
        return out
    return operand


@numpy_eager_registry.register("AdaptiveMaxPool3D_Indices")
def _np_adaptive_max_pool3d_indices(backend_module: BackendModuleProtocol, operand: EagerValue, output_size: tuple[int, int, int], **kwargs: EagerValue) -> EagerValue:
    """Execute the eager np adaptive max pool3d indices operation on the active backend array.

    Args:
        backend_module (ModuleType): Input `backend_module` argument (backend module).
        operand (EagerValue): Input `operand` argument (operand).
        output_size (tuple[int, int, int]): Input `output_size` argument (output size).
        **kwargs (EagerValue): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    res = _np_adaptive_max_pool3d(backend_module, operand, output_size, **kwargs)
    return (res, np.zeros_like(res, dtype=np.int64))


@numpy_eager_registry.register("AdaptiveMaxPool2D")
def _np_adaptive_max_pool2d(backend_module: BackendModuleProtocol, operand: EagerValue, output_size: tuple[int, int], **kwargs: EagerValue) -> EagerValue:
    """Execute the eager np adaptive max pool2d operation on the active backend array.

    Args:
        backend_module (ModuleType): Input `backend_module` argument (backend module).
        operand (EagerValue): Input `operand` argument (operand).
        output_size (tuple[int, int]): Input `output_size` argument (output size).
        **kwargs (EagerValue): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    if isinstance(operand, np.ndarray):
        sh = list(operand.shape)
        h_in, w_in = sh[-2], sh[-1]
        h_out, w_out = output_size[0], output_size[1]

        h_indices = _adaptive_pool_1d_indices(h_in, h_out)
        w_indices = _adaptive_pool_1d_indices(w_in, w_out)

        out_shape = sh[:-2] + [h_out, w_out]
        out = np.zeros(out_shape, dtype=operand.dtype)

        for i, (h_start, h_end) in enumerate(h_indices):
            for j, (w_start, w_end) in enumerate(w_indices):
                slice_val = operand[..., h_start:h_end, w_start:w_end]
                out[..., i, j] = np.max(slice_val, axis=(-2, -1))
        return out
    return operand


@numpy_eager_registry.register("FractionalAvgPool")
def _np_fractional_avg_pool(backend_module: BackendModuleProtocol, value: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager np fractional avg pool operation on the active backend array.

    Args:
        backend_module (ModuleType): Input `backend_module` argument (backend module).
        value (EagerValue): Input `value` argument (value).
        **kwargs (EagerValue): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    return np.asarray(value)


@numpy_eager_registry.register("FractionalMaxPool")
def _np_fractional_max_pool(backend_module: BackendModuleProtocol, value: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager np fractional max pool operation on the active backend array.

    Args:
        backend_module (ModuleType): Input `backend_module` argument (backend module).
        value (EagerValue): Input `value` argument (value).
        **kwargs (EagerValue): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    return np.asarray(value)
