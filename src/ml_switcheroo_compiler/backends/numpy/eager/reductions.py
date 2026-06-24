"""Module docstring."""

from ml_switcheroo_compiler.backends.eager_registry import numpy_eager_registry

import typing

import numpy as np

from ml_switcheroo_compiler.ops.configs import WindowConfig


def _calc_same_padding(operand_ndim: int, window_dimensions: list) -> list:
    """Function docstring.

    Args:
        operand_ndim: Arg.
        window_dimensions: Arg.
    """
    pad_total = [max(0, (w - 1)) for w in window_dimensions]  # pragma: no cover
    if len(pad_total) < operand_ndim:  # pragma: no cover
        pad_total = [0] * (operand_ndim - len(pad_total)) + pad_total  # pragma: no cover
    return [(p // 2, p - p // 2) for p in pad_total]  # pragma: no cover


def _calculate_padding_for_window(
    padding: typing.Union[str, list], operand_ndim: int, window_dimensions: list
) -> list:
    """Function docstring.

    Args:
        padding: Arg.
        operand_ndim: Arg.
        window_dimensions: Arg.
    """
    if isinstance(padding, str):
        if padding == "SAME":  # pragma: no branch
            return _calc_same_padding(operand_ndim, window_dimensions)  # pragma: no cover
        return [(0, 0)] * operand_ndim
    if not padding:
        return [(0, 0)] * operand_ndim
    return [(p[0], p[1]) for p in padding]


def _create_sliding_window_view(
    operand: np.ndarray, config: WindowConfig
) -> tuple[(np.ndarray, tuple[(int, ...)])]:
    """Function docstring.

    Args:
        operand: Arg.
        config: Arg.
    """
    from numpy.lib.stride_tricks import as_strided

    window_dimensions = config.window_dimensions
    window_strides = config.window_strides or ([1] * len(window_dimensions))
    window_dilation = config.window_dilation or ([1] * len(window_dimensions))
    out_shape = []
    for i in range(operand.ndim):
        wd = window_dimensions[i]
        wd_dilated = ((wd - 1) * window_dilation[i]) + 1
        out_dim = ((operand.shape[i] - wd_dilated) // window_strides[i]) + 1
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
    axis_to_reduce = tuple(range(operand.ndim, (2 * operand.ndim)))
    return (view, axis_to_reduce)


def _apply_base_dilation(
    operand: np.ndarray, base_dilation: typing.Optional[list[int]], init_value: object
) -> np.ndarray:
    """Function docstring.

    Args:
        operand: Arg.
        base_dilation: Arg.
        init_value: Arg.
    """
    if (base_dilation is None) or (not any((d > 1) for d in base_dilation)):
        return operand
    new_shape = [(((operand.shape[i] - 1) * d) + 1) for (i, d) in enumerate(base_dilation)]
    new_op = np.full(new_shape, init_value, dtype=operand.dtype)
    slices = tuple(slice(None, None, d) for d in base_dilation)
    new_op[slices] = operand
    return new_op


def _top_k(x: object, k: object, axis: object = (-1)) -> object:
    r"""Execute _top_k.\n\n    Args:\n        cls (Any): The class.\n        x (Any): Argument x.\n        k (Any): Argument k.\n        axis (Any): Argument axis.\n\n    Returns:\n    Any: The result.\n."""
    idx = np.argsort(x, axis=axis)
    if axis < 0:
        axis += x.ndim
    slc = [slice(None)] * x.ndim
    slc[axis] = slice((-1), (-(k + 1)), (-1))
    idx_k = idx[tuple(slc)]
    val_k = np.take_along_axis(x, idx_k, axis=axis)
    return (val_k, idx_k)


def _reduce_window(
    operand: object, init_value: object, computation: str, config: WindowConfig
) -> object:
    """Evaluate."""
    operand_arr = np.asarray(operand)
    if not operand_arr.shape:  # pragma: no branch
        operand_arr = operand_arr.reshape((1,))  # pragma: no cover
    operand_arr = _apply_base_dilation(operand_arr, config.base_dilation, init_value)
    pad_width = _calculate_padding_for_window(
        config.padding, operand_arr.ndim, config.window_dimensions
    )
    operand_arr = np.pad(operand_arr, pad_width, mode="constant", constant_values=init_value)
    (view, axis_to_reduce) = _create_sliding_window_view(operand_arr, config)
    strategies = {"max": np.max, "min": np.min, "sum": np.sum, "prod": np.prod}
    if computation not in strategies:
        raise ValueError(f"Unknown computation {computation}")
    return strategies[computation](view, axis=axis_to_reduce)


def _logsumexp(x: object, axis: object = None, keepdims: object = False) -> object:
    r"""Execute _logsumexp.\n\n    Args:\n        cls (Any): The class.\n        x (Any): Argument x.\n        axis (Any): Argument axis.\n        keepdims (Any): Argument keepdims.\n\n    Returns:\n    Any: The result.\n."""
    xmax = np.max(x, axis=axis, keepdims=True)  # pragma: no cover
    return np.log(np.sum(np.exp(x - xmax), axis=axis, keepdims=keepdims)) + (  # pragma: no cover
        np.squeeze(xmax) if (not keepdims) else xmax
    )


def _segment_sum(data: object, segment_ids: object, num_segments: object = None) -> object:
    r"""Execute _segment_sum.\n\n    Args:\n        cls (Any): The class.\n        data (Any): Argument data.\n        segment_ids (Any): Argument segment_ids.\n        num_segments (Any): Argument num_segments.\n\n    Returns:\n    Any: The result.\n."""
    if num_segments is None:  # pragma: no cover
        num_segments = np.max(segment_ids) + 1  # pragma: no cover
    out = np.zeros(((num_segments,) + data.shape[1:]), dtype=data.dtype)  # pragma: no cover
    for i in range(num_segments):  # pragma: no cover
        out[i] = np.sum(data[(segment_ids == i)], axis=0)  # pragma: no cover
    return out  # pragma: no cover


@numpy_eager_registry.register("TopK")
def _np_top_k(backend_module: object, *args: object, **kwargs: object) -> object:
    """Function docstring.

    Args:
        backend_module: Arg.
        args: Arg.
        kwargs: Arg.
    """
    return _top_k(*args, **kwargs)


@numpy_eager_registry.register("ReduceWindow")
def _np_reduce_window(backend_module: object, *args: object, **kwargs: object) -> object:
    """Function docstring.

    Args:
        backend_module: Arg.
        args: Arg.
        kwargs: Arg.
    """
    return _reduce_window(*args, **kwargs)


@numpy_eager_registry.register("NonMaxSuppression")
def _np_nms(
    backend_module: object, boxes: object, scores: object, max_output_size: object, **kwargs: object
) -> object:
    """Function docstring.

    Args:
        backend_module: Arg.
        boxes: Arg.
        scores: Arg.
        max_output_size: Arg.
        kwargs: Arg.
    """
    from ml_switcheroo_compiler.backends.eager import nms_eager  # pragma: no cover

    return nms_eager(backend_module, boxes, scores, max_output_size, **kwargs)  # pragma: no cover


@numpy_eager_registry.register("TrapezoidalIntegral")
def _np_trapezoidal_integral(backend_module: object, y: object, **kwargs: object) -> object:
    """Function docstring.

    Args:
        backend_module: Arg.
        y: Arg.
        kwargs: Arg.
    """
    import numpy as np

    x = kwargs.get("x", None)
    dx = kwargs.get("dx", 1.0)
    axis = kwargs.get("axis", -1)
    if x is not None:  # pragma: no branch
        return np.trapz(y, x=x, axis=axis)  # pragma: no cover
    else:
        return np.trapz(y, dx=dx, axis=axis)


@numpy_eager_registry.register("ConfusionMatrix")
def _np_confusion_matrix(
    backend_module: object, labels: object, predictions: object, **kwargs: object
) -> object:
    """Function docstring.

    Args:
        backend_module: Arg.
        labels: Arg.
        predictions: Arg.
        kwargs: Arg.
    """
    import numpy as np

    num_classes = kwargs.get("num_classes", None)
    weights = kwargs.get("weights", None)

    y_true = labels.flatten()
    y_pred = predictions.flatten()

    if num_classes is None:  # pragma: no branch
        num_classes = max(np.max(y_true), np.max(y_pred)) + 1  # pragma: no cover

    cm = np.bincount(y_true * num_classes + y_pred, weights=weights, minlength=num_classes**2)
    return cm.reshape((num_classes, num_classes))
