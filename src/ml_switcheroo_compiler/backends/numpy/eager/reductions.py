# ruff: noqa: E402, D100, D103, D104, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, D101, D102, D107, E701, E722, F403, E711, E712, PLR0913, PLR0915
"""Core abstractions and logic definitions for reductions.py."""

import typing
from typing import Any

import numpy as np
from numpy.lib.stride_tricks import as_strided

from ml_switcheroo_compiler.backends.eager import nms_eager
from ml_switcheroo_compiler.backends.eager_registry import numpy_eager_registry
from ml_switcheroo_compiler.ops.configs import WindowConfig


def _calc_same_padding(operand_ndim: int, window_dimensions: list) -> list:
    """Evaluate _calc_same_padding operation.

    Args:
        operand_ndim (int): The operand_ndim parameter.
        window_dimensions (list): The window_dimensions parameter.

    Returns:
        list: Result.
    """
    pad_total = [max(0, w - 1) for w in window_dimensions]
    if len(pad_total) < operand_ndim:
        pad_total = [0] * (operand_ndim - len(pad_total)) + pad_total
    return [(p // 2, p - p // 2) for p in pad_total]


def _calculate_padding_for_window(padding: typing.Union[str, list], operand_ndim: int, window_dimensions: list) -> list:
    """Evaluate _calculate_padding_for_window operation.

    Args:
        padding (object): The padding parameter.
        operand_ndim (int): The operand_ndim parameter.
        window_dimensions (list): The window_dimensions parameter.

    Returns:
        list: Result.
    """
    if isinstance(padding, str):
        if padding == "SAME":
            return _calc_same_padding(operand_ndim, window_dimensions)
        return [(0, 0)] * operand_ndim
    if not padding:
        return [(0, 0)] * operand_ndim
    return [(p[0], p[1]) for p in padding]


def _create_sliding_window_view(operand: np.ndarray, config: WindowConfig) -> tuple[np.ndarray, tuple[int, ...]]:
    """Evaluate _create_sliding_window_view operation.

    Args:
        operand (object): The operand parameter.
        config (WindowConfig): The config parameter.

    Returns:
        tuple: Result.
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


def _apply_base_dilation(operand: np.ndarray, base_dilation: typing.Optional[list[int]], init_value: Any) -> np.ndarray:
    """Evaluate _apply_base_dilation operation.

    Args:
        operand (object): The operand parameter.
        base_dilation (object): The base_dilation parameter.
        init_value (object): The init_value parameter.

    Returns: Any: Result.
    """
    if base_dilation is None or not any(d > 1 for d in base_dilation):
        return operand
    new_shape = [(operand.shape[i] - 1) * d + 1 for (i, d) in enumerate(base_dilation)]
    new_op = np.full(new_shape, init_value, dtype=operand.dtype)
    slices = tuple(slice(None, None, d) for d in base_dilation)
    new_op[slices] = operand
    return new_op


def _top_k(x: Any, k: Any, axis: Any = -1) -> Any:
    """Evaluate _top_k operation.

    Args:
        x (object): The x parameter.
        k (object): The k parameter.
        axis (object): The axis parameter.

    Returns: Any: Result.
    """
    idx = np.argsort(x, axis=axis)
    if axis < 0:
        axis += x.ndim
    slc = [slice(None)] * x.ndim
    slc[axis] = slice(-1, -(k + 1), -1)
    idx_k = idx[tuple(slc)]
    val_k = np.take_along_axis(x, idx_k, axis=axis)
    return (val_k, idx_k)


def _reduce_window(operand: Any, init_value: Any, computation: str, config: WindowConfig) -> Any:
    """Evaluate.

    Args:
        operand (object): The operand parameter.
        init_value (object): The init_value parameter.
        computation (str): The computation parameter.
        config (WindowConfig): The config parameter.

    Returns: Any: Result.

    Raises:
        ValueError: An exception.
    """
    operand_arr = np.asarray(operand)
    if not operand_arr.shape:
        operand_arr = operand_arr.reshape((1,))
    operand_arr = _apply_base_dilation(operand_arr, config.base_dilation, init_value)  # type: ignore  # Justification: Polymorphic / Duck Typing for Framework Agnosticism
    pad_width = _calculate_padding_for_window(config.padding, operand_arr.ndim, config.window_dimensions)  # type: ignore  # Justification: Polymorphic / Duck Typing for Framework Agnosticism
    operand_arr = np.pad(operand_arr, pad_width, mode="constant", constant_values=init_value)
    (view, axis_to_reduce) = _create_sliding_window_view(operand_arr, config)
    strategies = {"max": np.max, "min": np.min, "sum": np.sum, "prod": np.prod}
    if computation not in strategies:
        raise ValueError(f"Unknown computation {computation}")
    return strategies[computation](view, axis=axis_to_reduce)  # type: ignore  # Justification: Polymorphic / Duck Typing for Framework Agnosticism


def _logsumexp(x: Any, axis: Any = None, keepdims: Any = False) -> Any:
    """Evaluate _logsumexp operation.

    Args:
        x (object): The x parameter.
        axis (object): The axis parameter.
        keepdims (object): The keepdims parameter.

    Returns: Any: Result.
    """
    xmax = np.max(x, axis=axis, keepdims=True)
    return np.log(np.sum(np.exp(x - xmax), axis=axis, keepdims=keepdims)) + (np.squeeze(xmax) if not keepdims else xmax)


def _segment_sum(data: Any, segment_ids: Any, num_segments: Any = None) -> Any:
    """Evaluate _segment_sum operation.

    Args:
        data (object): The data parameter.
        segment_ids (object): The segment_ids parameter.
        num_segments (object): The num_segments parameter.

    Returns: Any: Result.
    """
    if num_segments is None:
        num_segments = np.max(segment_ids) + 1
    out = np.zeros((num_segments,) + data.shape[1:], dtype=data.dtype)
    for i in range(num_segments):
        out[i] = np.sum(data[segment_ids == i], axis=0)
    return out


@numpy_eager_registry.register("NonMaxSuppression")
def _np_nms(backend_module: Any, boxes: Any, scores: Any, max_output_size: Any, **kwargs: Any) -> Any:
    """Evaluate _np_nms operation.

    Args:
        backend_module (object): The backend_module parameter.
        boxes (object): The boxes parameter.
        scores (object): The scores parameter.
        max_output_size (object): The max_output_size parameter.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    return nms_eager(backend_module, boxes, scores, max_output_size, **kwargs)


@numpy_eager_registry.register("TrapezoidalIntegral")
def _np_trapezoidal_integral(backend_module: Any, y: Any, **kwargs: Any) -> Any:
    """Evaluate _np_trapezoidal_integral operation.

    Args:
        backend_module (object): The backend_module parameter.
        y (object): The y parameter.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    x = kwargs.get("x", None)
    dx = kwargs.get("dx", 1.0)
    axis = kwargs.get("axis", -1)
    if x is not None:
        return np.trapz(y, x=x, axis=axis)
    return np.trapz(y, dx=dx, axis=axis)


@numpy_eager_registry.register("ConfusionMatrix")
def _np_confusion_matrix(backend_module: Any, labels: Any, predictions: Any, **kwargs: Any) -> Any:
    """Evaluate _np_confusion_matrix operation.

    Args:
        backend_module (object): The backend_module parameter.
        labels (object): The labels parameter.
        predictions (object): The predictions parameter.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
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
def _np_cummax(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _np_cummax operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    a = args[0]
    axis = kwargs.get("axis", None)
    dtype = kwargs.get("dtype", None)
    if dtype is not None and str(dtype) != "None":
        dtype = getattr(dtype, "value", dtype)
        return np.maximum.accumulate(a, axis=axis, dtype=dtype)  # type: ignore  # Justification: Polymorphic / Duck Typing for Framework Agnosticism
    return np.maximum.accumulate(a, axis=axis)  # type: ignore  # Justification: Polymorphic / Duck Typing for Framework Agnosticism


@numpy_eager_registry.register("Cummin")
def _np_cummin(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _np_cummin operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    a = args[0]
    axis = kwargs.get("axis", None)
    dtype = kwargs.get("dtype", None)
    if dtype is not None and str(dtype) != "None":
        dtype = getattr(dtype, "value", dtype)
        return np.minimum.accumulate(a, axis=axis, dtype=dtype)  # type: ignore  # Justification: Polymorphic / Duck Typing for Framework Agnosticism
    return np.minimum.accumulate(a, axis=axis)  # type: ignore  # Justification: Polymorphic / Duck Typing for Framework Agnosticism


@numpy_eager_registry.register("Cumprod")
def _np_cumprod(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _np_cumprod operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    dtype = kwargs.pop("dtype", None)
    if dtype is not None and str(dtype) != "None":
        kwargs["dtype"] = getattr(dtype, "value", dtype)
    return np.cumprod(*args, **kwargs)


@numpy_eager_registry.register("Cumlogsumexp")
def _np_cumlogsumexp(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _np_cumlogsumexp operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    a = args[0]
    axis = kwargs.get("axis", None)
    if axis is None:
        return np.logaddexp.accumulate(np.ravel(a))
    return np.logaddexp.accumulate(a, axis=axis)


@numpy_eager_registry.register("ApproxMaxK")
def _np_approx_max_k(backend_module: Any, x: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _np_approx_max_k operation.

    Args:
        backend_module (object): The backend_module parameter.
        x (object): The x parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
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
def _np_approx_min_k(backend_module: Any, x: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _np_approx_min_k operation.

    Args:
        backend_module (object): The backend_module parameter.
        x (object): The x parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
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


def _get_k_val(k: Any) -> int:
    """Extract integer value from k.

    Args:
        k (object): The value to extract.

    Returns:
        int: Extracted integer.
    """
    if hasattr(k, "item"):
        return int(k.item())
    if hasattr(k, "data") and hasattr(k.data, "item"):
        return int(k.data.item())
    return int(k)


@numpy_eager_registry.register("TopK")
def _np_top_k(backend_module: Any, x: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _np_top_k operation.

    Args:
        backend_module (object): The backend_module parameter.
        x (object): The x parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
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
