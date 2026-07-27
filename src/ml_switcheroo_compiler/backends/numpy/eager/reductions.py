# ruff: noqa: E501
"""Core abstractions and logic definitions for reductions.py."""

import typing

import numpy as np
from numpy.lib.stride_tricks import as_strided

from ml_switcheroo_compiler.backends.eager import nms_eager
from ml_switcheroo_compiler.backends.eager_registry import numpy_eager_registry
from ml_switcheroo_compiler.ops.configs import WindowConfig


def _calc_same_padding(operand_ndim: int, window_dimensions: list) -> list:
    """Evaluate and process the calc same padding operation.

    Args:
        operand_ndim (int): Required parameter for operand_ndim.
        window_dimensions (list): Required parameter for window_dimensions.

    Returns:
        list: The evaluated or processed output.
    """
    pad_total = [max(0, w - 1) for w in window_dimensions]
    if len(pad_total) < operand_ndim:
        pad_total = [0] * (operand_ndim - len(pad_total)) + pad_total
    return [(p // 2, p - p // 2) for p in pad_total]


def _calculate_padding_for_window(padding: typing.Union[str, list], operand_ndim: int, window_dimensions: list) -> list:
    """Evaluate and process the calculate padding for window operation.

    Args:
        padding (Any): Required parameter for padding.
        operand_ndim (int): Required parameter for operand_ndim.
        window_dimensions (list): Required parameter for window_dimensions.

    Returns:
        list: The evaluated or processed output.
    """
    if isinstance(padding, str):
        if padding == "SAME":
            return _calc_same_padding(operand_ndim, window_dimensions)
        return [(0, 0)] * operand_ndim
    if not padding:
        return [(0, 0)] * operand_ndim
    return [(p[0], p[1]) for p in padding]


def _create_sliding_window_view(operand: np.ndarray, config: WindowConfig) -> tuple[np.ndarray, tuple[int, ...]]:
    """Evaluate and process the create sliding window view operation.

    Args:
        operand (ndarray): Required parameter for operand.
        config (WindowConfig): Required parameter for config.

    Returns:
        tuple: The evaluated or processed output.
    """
    window_dimensions = config.window_dimensions
    window_strides = config.window_strides or [1] * len(window_dimensions)
    window_dilation = config.window_dilation or [1] * len(window_dimensions)
    out_shape = []
    for i in range(operand.ndim):
        wd = window_dimensions[i]
        wd_dilated = (wd - 1) * window_dilation[i] + 1
        out_dim = (operand.shape[i] - wd_dilated) // window_strides[i] + 1
        out_shape.append(out_dim)
    strided_shape = []
    strided_strides = []
    for i in range(operand.ndim):
        strided_shape.append(out_shape[i])
        strided_strides.append(operand.strides[i] * window_strides[i])
    for i in range(operand.ndim):
        strided_shape.append(window_dimensions[i])
        strided_strides.append(operand.strides[i] * window_dilation[i])
    view = as_strided(operand, shape=strided_shape, strides=strided_strides, writeable=False)
    axis_to_reduce = tuple(range(operand.ndim, 2 * operand.ndim))
    return (view, axis_to_reduce)


def _apply_base_dilation(operand: np.ndarray, base_dilation: typing.Optional[list[int]], init_value: object) -> np.ndarray:
    """Evaluate and process the apply base dilation operation.

    Args:
        operand (ndarray): Required parameter for operand.
        base_dilation (Any): Required parameter for base_dilation.
        init_value (object): Required parameter for init_value.

    Returns:
        ndarray: The evaluated or processed output.
    """
    if base_dilation is None or not any(d > 1 for d in base_dilation):
        return operand
    new_shape = [(operand.shape[i] - 1) * d + 1 for (i, d) in enumerate(base_dilation)]
    new_op = np.full(new_shape, init_value, dtype=operand.dtype)
    slices = tuple(slice(None, None, d) for d in base_dilation)
    new_op[slices] = operand
    return new_op


def _top_k(x: object, k: object, axis: object = -1) -> object:
    """Evaluate and process the top k operation.

    Args:
        x (object): Required parameter for x.
        k (object): Required parameter for k.
        axis (object): Required parameter for axis.

    Returns:
        object: The evaluated or processed output.
    """
    idx = np.argsort(x, axis=axis)
    if axis < 0:
        axis += x.ndim
    slc = [slice(None)] * x.ndim
    slc[axis] = slice(-1, -(k + 1), -1)
    idx_k = idx[tuple(slc)]
    val_k = np.take_along_axis(x, idx_k, axis=axis)
    return (val_k, idx_k)


def _reduce_window(operand: object, init_value: object, computation: str, config: WindowConfig) -> object:
    """Evaluate."""
    operand_arr = np.asarray(operand)
    if not operand_arr.shape:
        operand_arr = operand_arr.reshape((1,))
    operand_arr = _apply_base_dilation(operand_arr, config.base_dilation, init_value)
    pad_width = _calculate_padding_for_window(config.padding, operand_arr.ndim, config.window_dimensions)
    operand_arr = np.pad(operand_arr, pad_width, mode="constant", constant_values=init_value)
    (view, axis_to_reduce) = _create_sliding_window_view(operand_arr, config)
    strategies = {"max": np.max, "min": np.min, "sum": np.sum, "prod": np.prod}
    if computation not in strategies:
        raise ValueError(f"Unknown computation {computation}")
    return strategies[computation](view, axis=axis_to_reduce)


def _logsumexp(x: object, axis: object = None, keepdims: object = False) -> object:
    """Evaluate and process the logsumexp operation.

    Args:
        x (object): Required parameter for x.
        axis (object): Required parameter for axis.
        keepdims (object): Required parameter for keepdims.

    Returns:
        object: The evaluated or processed output.
    """
    xmax = np.max(x, axis=axis, keepdims=True)
    return np.log(np.sum(np.exp(x - xmax), axis=axis, keepdims=keepdims)) + (np.squeeze(xmax) if not keepdims else xmax)


def _segment_sum(data: object, segment_ids: object, num_segments: object = None) -> object:
    """Evaluate and process the segment sum operation.

    Args:
        data (object): Required parameter for data.
        segment_ids (object): Required parameter for segment_ids.
        num_segments (object): Required parameter for num_segments.

    Returns:
        object: The evaluated or processed output.
    """
    if num_segments is None:
        num_segments = np.max(segment_ids) + 1
    out = np.zeros((num_segments,) + data.shape[1:], dtype=data.dtype)
    for i in range(num_segments):
        out[i] = np.sum(data[segment_ids == i], axis=0)
    return out


@numpy_eager_registry.register("NonMaxSuppression")
def _np_nms(backend_module: object, boxes: object, scores: object, max_output_size: object, **kwargs: object) -> object:
    """Evaluate the nms logic eagerly backed by NumPy.

    Args:
        backend_module (object): Required parameter for backend_module.
        boxes (object): Required parameter for boxes.
        scores (object): Required parameter for scores.
        max_output_size (object): Required parameter for max_output_size.
        **kwargs (Any): Arbitrary keyword arguments.

    Returns:
        object: The evaluated or processed output.
    """
    return nms_eager(backend_module, boxes, scores, max_output_size, **kwargs)


@numpy_eager_registry.register("TrapezoidalIntegral")
def _np_trapezoidal_integral(backend_module: object, y: object, **kwargs: object) -> object:
    """Evaluate the trapezoidal integral logic eagerly backed by NumPy.

    Args:
        backend_module (object): Required parameter for backend_module.
        y (object): Required parameter for y.
        **kwargs (Any): Arbitrary keyword arguments.

    Returns:
        object: The evaluated or processed output.
    """
    x = kwargs.get("x", None)
    dx = kwargs.get("dx", 1.0)
    axis = kwargs.get("axis", -1)
    if x is not None:
        return np.trapz(y, x=x, axis=axis)
    return np.trapz(y, dx=dx, axis=axis)


@numpy_eager_registry.register("ConfusionMatrix")
def _np_confusion_matrix(backend_module: object, labels: object, predictions: object, **kwargs: object) -> object:
    """Evaluate the confusion matrix logic eagerly backed by NumPy.

    Args:
        backend_module (object): Required parameter for backend_module.
        labels (object): Required parameter for labels.
        predictions (object): Required parameter for predictions.
        **kwargs (Any): Arbitrary keyword arguments.

    Returns:
        object: The evaluated or processed output.
    """
    num_classes = kwargs.get("num_classes", None)
    weights = kwargs.get("weights", None)
    y_true = labels.flatten()
    y_pred = predictions.flatten()
    if num_classes is None:
        num_classes = max(np.max(y_true), np.max(y_pred)) + 1
    cm = np.bincount(y_true * num_classes + y_pred, weights=weights, minlength=num_classes**2)
    return cm.reshape((num_classes, num_classes))


@numpy_eager_registry.register("Cummax")
def _np_cummax(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate the cummax logic eagerly backed by NumPy.

    Args:
        backend_module (object): Required parameter for backend_module.
        *args (Any): Variable positional arguments.
        **kwargs (Any): Arbitrary keyword arguments.

    Returns:
        object: The evaluated or processed output.
    """
    a = args[0]
    axis = kwargs.get("axis", None)
    dtype = kwargs.get("dtype", None)
    if dtype is not None and str(dtype) != "None":
        dtype = getattr(dtype, "value", dtype)
        return np.maximum.accumulate(a, axis=axis, dtype=dtype)
    return np.maximum.accumulate(a, axis=axis)


@numpy_eager_registry.register("Cummin")
def _np_cummin(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate the cummin logic eagerly backed by NumPy.

    Args:
        backend_module (object): Required parameter for backend_module.
        *args (Any): Variable positional arguments.
        **kwargs (Any): Arbitrary keyword arguments.

    Returns:
        object: The evaluated or processed output.
    """
    a = args[0]
    axis = kwargs.get("axis", None)
    dtype = kwargs.get("dtype", None)
    if dtype is not None and str(dtype) != "None":
        dtype = getattr(dtype, "value", dtype)
        return np.minimum.accumulate(a, axis=axis, dtype=dtype)
    return np.minimum.accumulate(a, axis=axis)


@numpy_eager_registry.register("Cumprod")
def _np_cumprod(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate the cumprod logic eagerly backed by NumPy.

    Args:
        backend_module (object): Required parameter for backend_module.
        *args (Any): Variable positional arguments.
        **kwargs (Any): Arbitrary keyword arguments.

    Returns:
        object: The evaluated or processed output.
    """
    dtype = kwargs.pop("dtype", None)
    if dtype is not None and str(dtype) != "None":
        kwargs["dtype"] = getattr(dtype, "value", dtype)
    return np.cumprod(*args, **kwargs)


@numpy_eager_registry.register("Cumlogsumexp")
def _np_cumlogsumexp(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate the cumlogsumexp logic eagerly backed by NumPy.

    Args:
        backend_module (object): Required parameter for backend_module.
        *args (Any): Variable positional arguments.
        **kwargs (Any): Arbitrary keyword arguments.

    Returns:
        object: The evaluated or processed output.
    """
    a = args[0]
    axis = kwargs.get("axis", None)
    if axis is None:
        return np.logaddexp.accumulate(np.ravel(a))
    return np.logaddexp.accumulate(a, axis=axis)


@numpy_eager_registry.register("ApproxMaxK")
def _np_approx_max_k(backend_module: object, x: object, *args: object, **kwargs: object) -> object:
    """Evaluate the approx max k logic eagerly backed by NumPy.

    Args:
        backend_module (object): Required parameter for backend_module.
        x (object): Required parameter for x.
        *args (Any): Variable positional arguments.
        **kwargs (Any): Arbitrary keyword arguments.

    Returns:
        object: The evaluated or processed output.
    """
    k = args[0] if len(args) > 0 else kwargs.get("k", 1)
    reduction_dimension = kwargs.get("reduction_dimension", -1)
    if not hasattr(x, "shape"):
        x = backend_module.array(x)
    if x.size == 0:
        return (x, x)
    idx = np.argsort(x, axis=reduction_dimension)
    idx = np.take(
        idx,
        range(idx.shape[reduction_dimension] - 1, idx.shape[reduction_dimension] - 1 - k, -1),
        axis=reduction_dimension,
    )
    val = np.take_along_axis(x, idx, axis=reduction_dimension)
    return (val, idx)


@numpy_eager_registry.register("ApproxMinK")
def _np_approx_min_k(backend_module: object, x: object, *args: object, **kwargs: object) -> object:
    """Evaluate the approx min k logic eagerly backed by NumPy.

    Args:
        backend_module (object): Required parameter for backend_module.
        x (object): Required parameter for x.
        *args (Any): Variable positional arguments.
        **kwargs (Any): Arbitrary keyword arguments.

    Returns:
        object: The evaluated or processed output.
    """
    k = args[0] if len(args) > 0 else kwargs.get("k", 1)
    reduction_dimension = kwargs.get("reduction_dimension", -1)
    if not hasattr(x, "shape"):
        x = backend_module.array(x)
    if x.size == 0:
        return (x, x)
    idx = np.argsort(x, axis=reduction_dimension)
    idx = np.take(idx, range(k), axis=reduction_dimension)
    val = np.take_along_axis(x, idx, axis=reduction_dimension)
    return (val, idx)


def _get_k_val(k: object) -> int:
    if hasattr(k, "item"):
        return int(k.item())
    if hasattr(k, "data") and hasattr(k.data, "item"):
        return int(k.data.item())
    return int(k)


@numpy_eager_registry.register("TopK")
def _np_top_k(backend_module: object, x: object, *args: object, **kwargs: object) -> object:
    """Evaluate the top k logic eagerly backed by NumPy.

    Args:
        backend_module (object): Required parameter for backend_module.
        x (object): Required parameter for x.
        *args (Any): Variable positional arguments.
        **kwargs (Any): Arbitrary keyword arguments.

    Returns:
        object: The evaluated or processed output.
    """
    k_arg = args[0] if len(args) > 0 else kwargs.get("k", 1)
    k = _get_k_val(k_arg)
    axis = kwargs.get("axis", -1)
    return_indices = kwargs.get("return_indices", None)
    if not hasattr(x, "shape"):
        x = backend_module.array(x)
    kth = max(0, x.shape[axis] - k)
    if return_indices is False:
        val = backend_module.partition(x, kth, axis=axis)
        slc = [slice(None)] * len(x.shape)
        slc[axis] = slice(-k, None)
        return val[tuple(slc)]
    idx = backend_module.argpartition(x, kth, axis=axis)
    slc = [slice(None)] * len(x.shape)
    slc[axis] = slice(-k, None)
    idx_k = idx[tuple(slc)]
    if return_indices is True:
        return idx_k
    val_k = backend_module.take_along_axis(x, idx_k, axis=axis)
    return (val_k, idx_k)
