"""Numpy Window Reductions."""

from typing import Any

# ruff: noqa: E402, D100, D103, D104, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, D101, D102, D107, E701, E722, F403, E711, E712, PLR0913, PLR0915
import numpy as np

from ml_switcheroo_compiler.backends.eager_registry import numpy_eager_registry
from ml_switcheroo_compiler.backends.numpy.eager.reductions import _reduce_window, _segment_sum


@numpy_eager_registry.register("ReduceWindow")
def _np_reduce_window(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _np_reduce_window operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    return _reduce_window(*args, **kwargs)


@numpy_eager_registry.register("SegmentSum")
def _np_segment_sum(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _np_segment_sum operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    return _segment_sum(*args, **kwargs)


@numpy_eager_registry.register("SegmentMax")
def _np_segment_max(backend_module: Any, data: Any, segment_ids: Any, num_segments: Any = None) -> Any:
    """Evaluate _np_segment_max operation.

    Args:
        backend_module (object): The backend_module parameter.
        data (object): The data parameter.
        segment_ids (object): The segment_ids parameter.
        num_segments (object): The num_segments parameter.

    Returns: Any: Result.
    """
    if num_segments is None:
        num_segments = np.max(segment_ids) + 1
    out = np.full((num_segments,) + data.shape[1:], -np.inf, dtype=data.dtype)
    for i in range(num_segments):
        mask = segment_ids == i
        if np.any(mask):
            out[i] = np.max(data[mask], axis=0)
    return out


@numpy_eager_registry.register("SegmentMin")
def _np_segment_min(backend_module: Any, data: Any, segment_ids: Any, num_segments: Any = None) -> Any:
    """Evaluate _np_segment_min operation.

    Args:
        backend_module (object): The backend_module parameter.
        data (object): The data parameter.
        segment_ids (object): The segment_ids parameter.
        num_segments (object): The num_segments parameter.

    Returns: Any: Result.
    """
    if num_segments is None:
        num_segments = np.max(segment_ids) + 1
    out = np.full((num_segments,) + data.shape[1:], np.inf, dtype=data.dtype)
    for i in range(num_segments):
        mask = segment_ids == i
        if np.any(mask):
            out[i] = np.min(data[mask], axis=0)
    return out


@numpy_eager_registry.register("SegmentProd")
def _np_segment_prod(backend_module: Any, data: Any, segment_ids: Any, num_segments: Any = None) -> Any:
    """Evaluate _np_segment_prod operation.

    Args:
        backend_module (object): The backend_module parameter.
        data (object): The data parameter.
        segment_ids (object): The segment_ids parameter.
        num_segments (object): The num_segments parameter.

    Returns: Any: Result.
    """
    if num_segments is None:
        num_segments = np.max(segment_ids) + 1
    out = np.ones((num_segments,) + data.shape[1:], dtype=data.dtype)
    for i in range(num_segments):
        mask = segment_ids == i
        if np.any(mask):
            out[i] = np.prod(data[mask], axis=0)
    return out


def _adaptive_pool_1d_indices(input_dim: int, output_dim: int) -> list[tuple[int, int]]:
    """Evaluate _adaptive_pool_1d_indices operation.

    Args:
        input_dim (int): The input_dim parameter.
        output_dim (int): The output_dim parameter.

    Returns: Any: Result.
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
def _np_adaptive_avg_pool2d(backend_module: Any, operand: Any, output_size: tuple[int, int], **kwargs: Any) -> Any:
    """Evaluate _np_adaptive_avg_pool2d operation.

    Args:
        backend_module (object): The backend_module parameter.
        operand (object): The operand parameter.
        output_size (object): The output_size parameter.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
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
def _np_adaptive_avg_pool3d(backend_module: Any, operand: Any, output_size: tuple[int, int, int], **kwargs: Any) -> Any:
    """Evaluate _np_adaptive_avg_pool3d operation.

    Args:
        backend_module (object): The backend_module parameter.
        operand (object): The operand parameter.
        output_size (object): The output_size parameter.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
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
def _np_adaptive_max_pool3d(backend_module: Any, operand: Any, output_size: tuple[int, int, int], **kwargs: Any) -> Any:
    """Evaluate _np_adaptive_max_pool3d operation.

    Args:
        backend_module (object): The backend_module parameter.
        operand (object): The operand parameter.
        output_size (object): The output_size parameter.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
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
def _np_adaptive_max_pool3d_indices(backend_module: Any, operand: Any, output_size: tuple[int, int, int], **kwargs: Any) -> Any:
    """Evaluate _np_adaptive_max_pool3d_indices operation.

    Args:
        backend_module (object): The backend_module parameter.
        operand (object): The operand parameter.
        output_size (object): The output_size parameter.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    res = _np_adaptive_max_pool3d(backend_module, operand, output_size, **kwargs)
    return (res, np.zeros_like(res, dtype=np.int64))


@numpy_eager_registry.register("AdaptiveMaxPool2D")
def _np_adaptive_max_pool2d(backend_module: Any, operand: Any, output_size: tuple[int, int], **kwargs: Any) -> Any:
    """Evaluate _np_adaptive_max_pool2d operation.

    Args:
        backend_module (object): The backend_module parameter.
        operand (object): The operand parameter.
        output_size (object): The output_size parameter.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
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
def _np_fractional_avg_pool(backend_module: Any, value: Any, **kwargs: Any) -> Any:
    """Evaluate _np_fractional_avg_pool operation.

    Args:
        backend_module (object): The backend_module parameter.
        value (object): The value parameter.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    return np.asarray(value)


@numpy_eager_registry.register("FractionalMaxPool")
def _np_fractional_max_pool(backend_module: Any, value: Any, **kwargs: Any) -> Any:
    """Evaluate _np_fractional_max_pool operation.

    Args:
        backend_module (object): The backend_module parameter.
        value (object): The value parameter.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    return np.asarray(value)
