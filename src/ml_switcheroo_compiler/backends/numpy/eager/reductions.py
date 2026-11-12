"""Implementation of reductions operations and utilities for the backends subsystem (`ml_switcheroo_compiler.backends.numpy.eager.reductions`)."""

from __future__ import annotations

import typing

import numpy as np
from numpy.lib.stride_tricks import as_strided

from ml_switcheroo_compiler.backends.eager import nms_eager
from ml_switcheroo_compiler.backends.eager_registry import (
    EagerCallableProtocol,
    EagerValue,
    ensure_callable,
    ensure_eager_tensor,
    ensure_int,
    numpy_eager_registry,
)
from ml_switcheroo_compiler.backends.numpy.eager import ensure_ndarray
from ml_switcheroo_compiler.backends.protocols import BackendModuleProtocol, get_backend_member, has_backend_member

if typing.TYPE_CHECKING:
    from ml_switcheroo_compiler.ops.configs import WindowConfig


def _calc_same_padding(operand_ndim: int, window_dimensions: EagerValue) -> EagerValue:
    """Execute the eager calc same padding operation on the active backend array.

    Args:
        operand_ndim (int): Input `operand_ndim` argument (operand ndim).
        window_dimensions (list): Input `window_dimensions` argument (window dimensions).

    Returns:
        list: Computed `list` output.
    """
    from ml_switcheroo_compiler.backends.eager_registry import ensure_int_list

    pad_total = [max(0, int(w) - 1) for w in ensure_int_list(window_dimensions)]
    if len(pad_total) < operand_ndim:
        pad_total = [0] * (operand_ndim - len(pad_total)) + pad_total
    return [(p // 2, p - p // 2) for p in pad_total]


def _calculate_padding_for_window(padding: str | list[tuple[int, int]], operand_ndim: int, window_dimensions: EagerValue) -> EagerValue:
    """Execute the eager calculate padding for window operation on the active backend array.

    Args:
        padding (typing.Union[str, list[tuple[int, int]]]): Input `padding` argument (padding).
        operand_ndim (int): Input `operand_ndim` argument (operand ndim).
        window_dimensions (list): Input `window_dimensions` argument (window dimensions).

    Returns:
        list: Computed `list` output.
    """
    if isinstance(padding, str):
        if padding == "SAME":
            return _calc_same_padding(operand_ndim, window_dimensions)
        return [(0, 0)] * operand_ndim
    return [(p[0], p[1]) for p in padding]


def _create_sliding_window_view(operand: np.ndarray[tuple[int, ...], np.dtype[np.float64]], config: WindowConfig) -> tuple[np.ndarray[tuple[int, ...], np.dtype[np.float64]], tuple[int, ...]]:
    """Execute the eager create sliding window view operation on the active backend array.

    Args:
        operand (np.ndarray[tuple[int, ...], np.dtype[np.float64]]): Input `operand` argument (operand).
        config (WindowConfig): Input `config` argument (config).

    Returns:
        tuple: Computed `tuple` output.
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


def _apply_base_dilation(operand: np.ndarray[tuple[int, ...], np.dtype[np.float64]], base_dilation: list[int] | None, init_value: EagerValue) -> np.ndarray[tuple[int, ...], np.dtype[np.float64]]:
    """Execute the eager apply base dilation operation on the active backend array.

    Args:
        operand (np.ndarray[tuple[int, ...], np.dtype[np.float64]]): Input `operand` argument (operand).
        base_dilation (typing.Optional[list[int]]): Input `base_dilation` argument (base dilation).
        init_value (EagerValue): Input `init_value` argument (init value).

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    if base_dilation is None or not any(d > 1 for d in base_dilation):
        return operand
    new_shape = [(operand.shape[i] - 1) * d + 1 for (i, d) in enumerate(base_dilation)]
    new_op = np.full(new_shape, init_value, dtype=operand.dtype)
    slices = tuple(slice(None, None, d) for d in base_dilation)
    new_op[slices] = operand
    return new_op


def _top_k(x: EagerValue, k: int, axis: int = -1) -> tuple[np.ndarray[tuple[int, ...], np.dtype[np.float64]], np.ndarray[tuple[int, ...], np.dtype[np.float64]]]:
    """Execute the eager top k operation on the active backend array.

    Args:
        x (EagerValue): Input `x` argument (x).
        k (int): Input `k` argument (k).
        axis (int): Input `axis` argument (axis).

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    idx = ensure_ndarray(get_backend_member(np, "argsort")(x, axis=axis))
    ndim_val = ensure_int(get_backend_member(x, "ndim"))
    if axis < 0:
        axis += ndim_val
    slc = [slice(None)] * ndim_val
    slc[axis] = slice(-1, -(k + 1), -1)
    idx_k = ensure_ndarray(idx[tuple(slc)])
    val_k: np.ndarray[tuple[int, ...], np.dtype[np.float64]] = ensure_ndarray(get_backend_member(np, "take_along_axis")(x, idx_k, axis=axis))
    return (val_k, idx_k)


def _reduce_window(operand: EagerValue, init_value: EagerValue, computation: str, config: WindowConfig) -> EagerValue:
    """Evaluate.

    Args:
        operand (EagerValue): Input `operand` argument (operand).
        init_value (EagerValue): Input `init_value` argument (init value).
        computation (str): Input `computation` argument (computation).
        config (WindowConfig): Input `config` argument (config).

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.

    Raises:
        ValueError: If invalid inputs or unsupported runtime conditions trigger `ValueError`.
    """
    operand_arr = np.asarray(operand)
    if not operand_arr.shape:
        operand_arr = operand_arr.reshape((1,))
    base_dil = list(config.base_dilation) if config.base_dilation is not None else None
    operand_arr = _apply_base_dilation(operand_arr, base_dil, init_value)
    pad_cfg: str | list[tuple[int, int]]
    if config.padding is None:
        pad_cfg = "VALID"
    elif isinstance(config.padding, str):
        pad_cfg = config.padding
    else:
        pad_cfg = [(p[0], p[1]) if isinstance(p, tuple) else (int(p), int(p)) for p in config.padding]
    pad_width = _calculate_padding_for_window(
        pad_cfg,
        operand_arr.ndim,
        config.window_dimensions,
    )
    operand_arr = ensure_ndarray(get_backend_member(np, "pad")(operand_arr, pad_width, mode="constant", constant_values=init_value))
    (view, axis_to_reduce) = _create_sliding_window_view(operand_arr, config)
    strategies: dict[str, EagerCallableProtocol] = {
        "max": ensure_callable(get_backend_member(np, "max")),
        "min": ensure_callable(get_backend_member(np, "min")),
        "sum": ensure_callable(get_backend_member(np, "sum")),
        "prod": ensure_callable(get_backend_member(np, "prod")),
    }
    if computation not in strategies:
        raise ValueError(f"Unknown computation {computation}")
    return strategies[computation](view, axis=axis_to_reduce)


def _logsumexp(x: EagerValue, axis: EagerValue = None, keepdims: EagerValue = False) -> EagerValue:
    """Execute the eager logsumexp operation on the active backend array.

    Args:
        x (EagerValue): Input `x` argument (x).
        axis (EagerValue): Input `axis` argument (axis).
        keepdims (EagerValue): Input `keepdims` argument (keepdims).

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    xmax = get_backend_member(np, "max")(x, axis=axis, keepdims=True)
    return ensure_eager_tensor(get_backend_member(np, "log")(get_backend_member(np, "sum")(get_backend_member(np, "exp")(ensure_eager_tensor(x) - xmax), axis=axis, keepdims=keepdims)) + (get_backend_member(np, "squeeze")(xmax) if not keepdims else xmax))


def _segment_sum(
    data: np.ndarray[tuple[int, ...], np.dtype[np.float64]],
    segment_ids: np.ndarray[tuple[int, ...], np.dtype[np.int64]],
    num_segments: int | None = None,
) -> np.ndarray[tuple[int, ...], np.dtype[np.float64]]:
    """Execute the eager segment sum operation on the active backend array.

    Args:
        data (np.ndarray[tuple[int, ...], np.dtype[np.float64]]): Input `data` argument (data).
        segment_ids (np.ndarray[tuple[int, ...], np.dtype[np.int64]]): Input `segment_ids` argument (segment ids).
        num_segments (int | None): Input `num_segments` argument (num segments).

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    if num_segments is None:
        num_segments = int(np.max(segment_ids)) + 1
    out = np.zeros((num_segments,) + data.shape[1:], dtype=data.dtype)
    for i in range(num_segments):
        out[i] = np.sum(data[segment_ids == i], axis=0)
    return out


@numpy_eager_registry.register("NonMaxSuppression")
def _np_nms(backend_module: BackendModuleProtocol, boxes: EagerValue, scores: EagerValue, max_output_size: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager np nms operation on the active backend array.

    Args:
        backend_module (ModuleType): Input `backend_module` argument (backend module).
        boxes (EagerValue): Input `boxes` argument (boxes).
        scores (EagerValue): Input `scores` argument (scores).
        max_output_size (EagerValue): Input `max_output_size` argument (max output size).
        **kwargs (EagerValue): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    return ensure_callable(nms_eager)(backend_module, boxes, scores, max_output_size, **kwargs)


@numpy_eager_registry.register("TrapezoidalIntegral")
def _np_trapezoidal_integral(backend_module: BackendModuleProtocol, y: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager np trapezoidal integral operation on the active backend array.

    Args:
        backend_module (ModuleType): Input `backend_module` argument (backend module).
        y (EagerValue): Input `y` argument (y).
        **kwargs (EagerValue): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    x = kwargs.get("x", None)
    dx = kwargs.get("dx", 1.0)
    axis = kwargs.get("axis", -1)
    trapz_fn = ensure_callable(get_backend_member(np, "trapezoid", get_backend_member(np, "trapz", None)))
    if x is not None:
        return trapz_fn(y, x=x, axis=axis)
    return trapz_fn(y, dx=dx, axis=axis)


@numpy_eager_registry.register("ConfusionMatrix")
def _np_confusion_matrix(backend_module: BackendModuleProtocol, labels: EagerValue, predictions: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager np confusion matrix operation on the active backend array.

    Args:
        backend_module (ModuleType): Input `backend_module` argument (backend module).
        labels (EagerValue): Input `labels` argument (labels).
        predictions (EagerValue): Input `predictions` argument (predictions).
        **kwargs (EagerValue): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    num_classes = kwargs.get("num_classes", None)
    weights = kwargs.get("weights", None)
    y_true = get_backend_member(labels, "flatten")()
    y_pred = get_backend_member(predictions, "flatten")()
    if num_classes is None:
        num_classes = max(np.max(y_true), np.max(y_pred)) + 1
    cm = get_backend_member(np, "bincount")(ensure_eager_tensor(y_true) * ensure_eager_tensor(num_classes) + ensure_eager_tensor(y_pred), weights=weights, minlength=ensure_eager_tensor(num_classes) ** 2)
    return ensure_eager_tensor(get_backend_member(cm, "reshape")((num_classes, num_classes)))


@numpy_eager_registry.register("Cummax")
def _np_cummax(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager np cummax operation on the active backend array.

    Args:
        backend_module (ModuleType): Input `backend_module` argument (backend module).
        *args (EagerValue): Positional args.
        **kwargs (EagerValue): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    a = args[0]
    axis = kwargs.get("axis", None)
    dtype = kwargs.get("dtype", None)
    if dtype is not None and str(dtype) != "None":
        dtype = get_backend_member(dtype, "value", dtype)
        return ensure_eager_tensor(get_backend_member(np.maximum, "accumulate")(a, axis=axis, dtype=dtype))
    return ensure_eager_tensor(get_backend_member(np.maximum, "accumulate")(a, axis=axis))


@numpy_eager_registry.register("Cummin")
def _np_cummin(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager np cummin operation on the active backend array.

    Args:
        backend_module (ModuleType): Input `backend_module` argument (backend module).
        *args (EagerValue): Positional args.
        **kwargs (EagerValue): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    a = args[0]
    axis = kwargs.get("axis", None)
    dtype = kwargs.get("dtype", None)
    if dtype is not None and str(dtype) != "None":
        dtype = get_backend_member(dtype, "value", dtype)
        return ensure_eager_tensor(get_backend_member(np.minimum, "accumulate")(a, axis=axis, dtype=dtype))
    return ensure_eager_tensor(get_backend_member(np.minimum, "accumulate")(a, axis=axis))


@numpy_eager_registry.register("Cumprod")
def _np_cumprod(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager np cumprod operation on the active backend array.

    Args:
        backend_module (ModuleType): Input `backend_module` argument (backend module).
        *args (EagerValue): Positional args.
        **kwargs (EagerValue): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    dtype = kwargs.pop("dtype", None)
    if dtype is not None and str(dtype) != "None":
        kwargs["dtype"] = get_backend_member(dtype, "value", dtype)
    return ensure_eager_tensor(get_backend_member(np, "cumprod")(*args, **kwargs))


@numpy_eager_registry.register("Cumlogsumexp")
def _np_cumlogsumexp(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager np cumlogsumexp operation on the active backend array.

    Args:
        backend_module (ModuleType): Input `backend_module` argument (backend module).
        *args (EagerValue): Positional args.
        **kwargs (EagerValue): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    a = args[0]
    axis = kwargs.get("axis", None)
    if axis is None:
        return np.logaddexp.accumulate(get_backend_member(np, "ravel")(a))
    return ensure_eager_tensor(get_backend_member(np.logaddexp, "accumulate")(a, axis=axis))


@numpy_eager_registry.register("ApproxMaxK")
def _np_approx_max_k(backend_module: BackendModuleProtocol, x: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager np approx max k operation on the active backend array.

    Args:
        backend_module (ModuleType): Input `backend_module` argument (backend module).
        x (EagerValue): Input `x` argument (x).
        *args (EagerValue): Positional args.
        **kwargs (EagerValue): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    k = args[0] if len(args) > 0 else kwargs.get("k", 1)
    reduction_dimension = kwargs.get("reduction_dimension", -1)
    if not has_backend_member(x, "shape"):
        x = backend_module.array(x)
    if get_backend_member(x, "size") == 0:
        return (x, x)
    idx = get_backend_member(np, "argsort")(x, axis=reduction_dimension)
    red_dim = int(ensure_int(reduction_dimension))
    dim_len = int(ensure_int(ensure_eager_tensor(idx.shape)[red_dim]))
    idx = get_backend_member(np, "take")(
        idx,
        range(dim_len - 1, dim_len - 1 - int(ensure_int(k)), -1),
        axis=reduction_dimension,
    )
    val = get_backend_member(np, "take_along_axis")(x, idx, axis=reduction_dimension)
    return (val, idx)


@numpy_eager_registry.register("ApproxMinK")
def _np_approx_min_k(backend_module: BackendModuleProtocol, x: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager np approx min k operation on the active backend array.

    Args:
        backend_module (ModuleType): Input `backend_module` argument (backend module).
        x (EagerValue): Input `x` argument (x).
        *args (EagerValue): Positional args.
        **kwargs (EagerValue): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    k = args[0] if len(args) > 0 else kwargs.get("k", 1)
    reduction_dimension = kwargs.get("reduction_dimension", -1)
    if not has_backend_member(x, "shape"):
        x = backend_module.array(x)
    if get_backend_member(x, "size") == 0:
        return (x, x)
    idx = get_backend_member(np, "argsort")(x, axis=reduction_dimension)
    idx = get_backend_member(np, "take")(idx, range(ensure_int(k)), axis=reduction_dimension)
    val = get_backend_member(np, "take_along_axis")(x, idx, axis=reduction_dimension)
    return (val, idx)


def _get_k_val(k: EagerValue) -> int:
    """Extract integer value from k.

    Args:
        k (EagerValue): The value to extract.

    Returns:
        int: Extracted integer.
    """
    if has_backend_member(k, "item"):
        return int(ensure_int(get_backend_member(k, "item")()))
    if has_backend_member(k, "data") and has_backend_member(get_backend_member(k, "data"), "item"):
        return int(ensure_int(get_backend_member(k, "data").item()))
    return int(ensure_int(k))


@numpy_eager_registry.register("TopK")
def _np_top_k(backend_module: BackendModuleProtocol, x: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager np top k operation on the active backend array.

    Args:
        backend_module (ModuleType): Input `backend_module` argument (backend module).
        x (EagerValue): Input `x` argument (x).
        *args (EagerValue): Positional args.
        **kwargs (EagerValue): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    k_arg = args[0] if len(args) > 0 else kwargs.get("k", 1)
    k = _get_k_val(k_arg)
    axis = ensure_int(kwargs.get("axis", -1))
    return_indices = kwargs.get("return_indices", None)
    if not has_backend_member(x, "shape"):
        x = backend_module.array(x)
    x_shape = get_backend_member(x, "shape")
    kth = max(0, ensure_int(x_shape[axis]) - k)
    if return_indices is False:
        val = backend_module.partition(x, kth, axis=axis)
        slc: list[EagerValue] = [slice(None)] * len(x_shape)
        slc[axis] = slice(-k, None)
        return val[tuple(slc)]
    idx = backend_module.argpartition(x, kth, axis=axis)
    slc = [slice(None)] * len(x_shape)
    slc[axis] = slice(-k, None)
    idx_k = idx[tuple(slc)]
    if return_indices is True:
        return idx_k
    val_k = backend_module.take_along_axis(x, idx_k, axis=axis)
    return (val_k, idx_k)
