"""Numpy Window Reductions."""

# ruff: noqa: E501
import numpy as np

from ml_switcheroo_compiler.backends.eager_registry import numpy_eager_registry
from ml_switcheroo_compiler.backends.numpy.eager.reductions import _reduce_window, _segment_sum


@numpy_eager_registry.register("ReduceWindow")
def _np_reduce_window(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate the reduce window logic eagerly backed by NumPy.

    Args:
        backend_module (object): Required parameter for backend_module.
        *args (Any): Variable positional arguments.
        **kwargs (Any): Arbitrary keyword arguments.

    Returns:
        object: The evaluated or processed output.
    """
    return _reduce_window(*args, **kwargs)


@numpy_eager_registry.register("SegmentSum")
def _np_segment_sum(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate the segment sum logic eagerly backed by NumPy.

    Args:
        backend_module (object): Required parameter for backend_module.
        *args (Any): Variable positional arguments.
        **kwargs (Any): Arbitrary keyword arguments.

    Returns:
        object: The evaluated or processed output.
    """
    return _segment_sum(*args, **kwargs)


@numpy_eager_registry.register("SegmentMax")
def _np_segment_max(backend_module: object, data: object, segment_ids: object, num_segments: object = None) -> object:
    """Evaluate the segment max logic eagerly backed by NumPy.

    Args:
        backend_module (object): Required parameter for backend_module.
        data (object): Required parameter for data.
        segment_ids (object): Required parameter for segment_ids.
        num_segments (object): Required parameter for num_segments.

    Returns:
        object: The evaluated or processed output.
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
def _np_segment_min(backend_module: object, data: object, segment_ids: object, num_segments: object = None) -> object:
    """Evaluate the segment min logic eagerly backed by NumPy.

    Args:
        backend_module (object): Required parameter for backend_module.
        data (object): Required parameter for data.
        segment_ids (object): Required parameter for segment_ids.
        num_segments (object): Required parameter for num_segments.

    Returns:
        object: The evaluated or processed output.
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
def _np_segment_prod(backend_module: object, data: object, segment_ids: object, num_segments: object = None) -> object:
    """Evaluate the segment prod logic eagerly backed by NumPy.

    Args:
        backend_module (object): Required parameter for backend_module.
        data (object): Required parameter for data.
        segment_ids (object): Required parameter for segment_ids.
        num_segments (object): Required parameter for num_segments.

    Returns:
        object: The evaluated or processed output.
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
    """Compute start and end input indices for 1D adaptive pooling.

    Args:
        input_dim (int): Size of input dimension.
        output_dim (int): Size of target output dimension.

    Returns:
        list[tuple[int, int]]: Index ranges.
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
def _np_adaptive_avg_pool2d(backend_module: object, operand: object, output_size: tuple[int, int], **kwargs: object) -> object:
    """Evaluate the adaptive avg pool2d logic eagerly backed by NumPy.

    Args:
        backend_module (object): Required parameter for backend_module.
        operand (object): Required parameter for operand.
        output_size (tuple): Required parameter for output_size.
        **kwargs (Any): Arbitrary keyword arguments.

    Returns:
        object: The evaluated or processed output.
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
def _np_adaptive_avg_pool3d(backend_module: object, operand: object, output_size: tuple[int, int, int], **kwargs: object) -> object:
    """Evaluate np adaptive avg pool3d."""
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
def _np_adaptive_max_pool3d(backend_module: object, operand: object, output_size: tuple[int, int, int], **kwargs: object) -> object:
    """Evaluate np adaptive max pool3d."""
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
def _np_adaptive_max_pool3d_indices(backend_module: object, operand: object, output_size: tuple[int, int, int], **kwargs: object) -> object:
    """Evaluate np adaptive max pool3d indices."""
    res = _np_adaptive_max_pool3d(backend_module, operand, output_size, **kwargs)
    return (res, np.zeros_like(res, dtype=np.int64))


@numpy_eager_registry.register("AdaptiveMaxPool2D")
def _np_adaptive_max_pool2d(backend_module: object, operand: object, output_size: tuple[int, int], **kwargs: object) -> object:
    """Evaluate the adaptive max pool2d logic eagerly backed by NumPy.

    Args:
        backend_module (object): Required parameter for backend_module.
        operand (object): Required parameter for operand.
        output_size (tuple): Required parameter for output_size.
        **kwargs (Any): Arbitrary keyword arguments.

    Returns:
        object: The evaluated or processed output.
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
def _np_fractional_avg_pool(backend_module: object, value: object, **kwargs: object) -> object:
    """Evaluate the fractional avg pool logic eagerly backed by NumPy.

    Args:
        backend_module (object): Required parameter for backend_module.
        value (object): Required parameter for value.
        **kwargs (Any): Arbitrary keyword arguments.

    Returns:
        object: The evaluated or processed output.
    """
    return np.asarray(value)


@numpy_eager_registry.register("FractionalMaxPool")
def _np_fractional_max_pool(backend_module: object, value: object, **kwargs: object) -> object:
    """Evaluate the fractional max pool logic eagerly backed by NumPy.

    Args:
        backend_module (object): Required parameter for backend_module.
        value (object): Required parameter for value.
        **kwargs (Any): Arbitrary keyword arguments.

    Returns:
        object: The evaluated or processed output.
    """
    return np.asarray(value)
