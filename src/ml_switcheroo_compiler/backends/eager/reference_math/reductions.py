"""Generic backend-parametric eager reference operations for `reference_math.reductions`."""

from __future__ import annotations

import types
import typing
from collections.abc import Sequence

from ml_switcheroo_compiler.backends.eager import nms_eager
from ml_switcheroo_compiler.backends.eager.utils import _resolve_eager_mod
from ml_switcheroo_compiler.backends.eager_registry import (
    BackendArg,
    EagerBoundCallable,
    EagerTensorProtocol,
    coerce_int,
    coerce_operand_list,
    global_eager_registry,
    require_callable,
)
from ml_switcheroo_compiler.backends.protocols import BackendModuleProtocol, adapt_backend_module

if typing.TYPE_CHECKING:
    from ml_switcheroo_compiler.ops.configs import WindowConfig


def _ensure_arr(val: BackendArg) -> EagerTensorProtocol:
    """Coerce `val` to an eager array via the active backend module.

    Args:
        val (BackendArg): Input array or scalar value.

    Returns:
        BackendArg: Coerced backend array.
    """
    return _resolve_eager_mod(None).asarray(val)


def _calc_same_padding(operand_ndim: int, window_dimensions: BackendArg) -> BackendArg:
    """Execute the eager calc same padding operation on the active backend array.

    Args:
        operand_ndim (int): Input `operand_ndim` argument (operand ndim).
        window_dimensions (BackendArg): Input `window_dimensions` argument (window dimensions).

    Returns:
        list: Computed `list` output.
    """
    from ml_switcheroo_compiler.backends.eager_registry import coerce_int_list

    pad_total = [max(0, int(w) - 1) for w in coerce_int_list(window_dimensions)]
    if len(pad_total) < operand_ndim:
        pad_total = [0] * (operand_ndim - len(pad_total)) + pad_total
    return [(p // 2, p - p // 2) for p in pad_total]


def _calculate_padding_for_window(
    padding: str | list[tuple[int, int]], operand_ndim: int, window_dimensions: BackendArg
) -> BackendArg:
    """Execute the eager calculate padding for window operation on the active backend array.

    Args:
        padding (typing.Union[str, list[tuple[int, int]]]): Input `padding` argument (padding).
        operand_ndim (int): Input `operand_ndim` argument (operand ndim).
        window_dimensions (BackendArg): Input `window_dimensions` argument (window dimensions).

    Returns:
        list: Computed `list` output.
    """
    if isinstance(padding, str):
        if padding == "SAME":
            return _calc_same_padding(operand_ndim, window_dimensions)
        return [(0, 0)] * operand_ndim
    return [(p[0], p[1]) for p in padding]


def _create_sliding_window_view(
    operand: BackendArg, config: WindowConfig
) -> tuple[EagerTensorProtocol, tuple[int, ...]]:
    """Execute the eager create sliding window view operation on the active backend array.

    Args:
        operand (BackendArg): Input `operand` argument (operand).
        config (WindowConfig): Input `config` argument (config).

    Returns:
        tuple: Computed `tuple` output.
    """
    from ml_switcheroo_compiler.backends.eager_registry import coerce_array

    arr_mod = _resolve_eager_mod(None)
    op_arr = coerce_array(arr_mod, operand)
    window_dimensions = config.window_dimensions
    window_strides = config.window_strides or [1] * len(window_dimensions)
    window_dilation = config.window_dilation or [1] * len(window_dimensions)
    out_shape = []
    for i in range(op_arr.ndim):
        wd = window_dimensions[i]
        wd_dilated = (wd - 1) * window_dilation[i] + 1
        out_dim = (op_arr.shape[i] - wd_dilated) // window_strides[i] + 1
        out_shape.append(out_dim)
    strided_shape = []
    strided_strides = []
    op_strides = [
        int(coerce_int(st)) for st in coerce_operand_list(adapt_backend_module(op_arr).require_symbol("strides"))
    ]
    for i in range(op_arr.ndim):
        strided_shape.append(out_shape[i])
        strided_strides.append(op_strides[i] * window_strides[i])
    for i in range(op_arr.ndim):
        strided_shape.append(window_dimensions[i])
        strided_strides.append(op_strides[i] * window_dilation[i])
    view = arr_mod.lib.stride_tricks.as_strided(op_arr, shape=strided_shape, strides=strided_strides, writeable=False)
    axis_to_reduce = tuple(range(op_arr.ndim, 2 * op_arr.ndim))
    return (view, axis_to_reduce)


def _apply_base_dilation(
    operand: BackendArg, base_dilation: list[int] | None, init_value: BackendArg
) -> EagerTensorProtocol:
    """Execute the eager apply base dilation operation on the active backend array.

    Args:
        operand (BackendArg): Input `operand` argument (operand).
        base_dilation (typing.Optional[list[int]]): Input `base_dilation` argument (base dilation).
        init_value (BackendArg): Input `init_value` argument (init value).

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    arr_mod = _resolve_eager_mod(None)
    op_arr = arr_mod.asarray(operand)
    if base_dilation is None or not any(d > 1 for d in base_dilation):
        return op_arr
    new_shape = [(op_arr.shape[i] - 1) * d + 1 for (i, d) in enumerate(base_dilation)]
    new_op = arr_mod.full(new_shape, init_value, dtype=op_arr.dtype)
    slices = tuple(slice(None, None, d) for d in base_dilation)
    new_op[slices] = op_arr
    return new_op


def _top_k(x: BackendArg, k: int, axis: int = -1) -> tuple[EagerTensorProtocol, EagerTensorProtocol]:
    """Execute the eager top k operation on the active backend array.

    Args:
        x (BackendArg): Input `x` argument (x).
        k (int): Input `k` argument (k).
        axis (int): Input `axis` argument (axis).

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    arr_mod = _resolve_eager_mod(None)
    idx = _ensure_arr(adapt_backend_module(arr_mod).require_symbol("argsort")(x, axis=axis))
    ndim_val = coerce_int(adapt_backend_module(x).require_symbol("ndim"))
    if axis < 0:
        axis += ndim_val
    slc = [slice(None)] * ndim_val
    slc[axis] = slice(-1, -(k + 1), -1)
    idx_k = _ensure_arr(idx[tuple(slc)])
    val_k: EagerTensorProtocol = _ensure_arr(
        adapt_backend_module(arr_mod).require_symbol("take_along_axis")(x, idx_k, axis=axis)
    )
    return (val_k, idx_k)


def _reduce_window(operand: BackendArg, init_value: BackendArg, computation: str, config: WindowConfig) -> BackendArg:
    """Evaluate.

    Args:
        operand (BackendArg): Input `operand` argument (operand).
        init_value (BackendArg): Input `init_value` argument (init value).
        computation (str): Input `computation` argument (computation).
        config (WindowConfig): Input `config` argument (config).

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.

    Raises:
        ValueError: If invalid inputs or unsupported runtime conditions trigger `ValueError`.
    """
    arr_mod = _resolve_eager_mod(None)
    operand_arr = arr_mod.asarray(operand)
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
    operand_arr = _ensure_arr(
        adapt_backend_module(arr_mod).require_symbol("pad")(
            operand_arr, pad_width, mode="constant", constant_values=init_value
        )
    )
    (view, axis_to_reduce) = _create_sliding_window_view(operand_arr, config)
    strategies: dict[str, EagerBoundCallable] = {
        "max": adapt_backend_module(arr_mod).require_op("max"),
        "min": adapt_backend_module(arr_mod).require_op("min"),
        "sum": adapt_backend_module(arr_mod).require_op("sum"),
        "prod": adapt_backend_module(arr_mod).require_op("prod"),
    }
    if computation not in strategies:
        raise ValueError(f"Unknown computation {computation}")
    return strategies[computation](view, axis=axis_to_reduce)


def _logsumexp(x: BackendArg, axis: BackendArg = None, keepdims: BackendArg = False) -> BackendArg:
    """Execute the eager logsumexp operation on the active backend array.

    Args:
        x (BackendArg): Input `x` argument (x).
        axis (BackendArg): Input `axis` argument (axis).
        keepdims (BackendArg): Input `keepdims` argument (keepdims).

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    from ml_switcheroo_compiler.backends.eager_registry import coerce_numeric

    arr_mod = _resolve_eager_mod(None)
    xmax = adapt_backend_module(arr_mod).require_symbol("max")(x, axis=axis, keepdims=True)
    return adapt_backend_module(arr_mod).require_symbol("log")(
        adapt_backend_module(arr_mod).require_symbol("sum")(
            adapt_backend_module(arr_mod).require_symbol("exp")(coerce_numeric(x) - xmax),
            axis=axis,
            keepdims=keepdims,
        )
    ) + (adapt_backend_module(arr_mod).require_symbol("squeeze")(xmax) if not keepdims else xmax)


def _segment_sum(
    data: BackendArg,
    segment_ids: BackendArg,
    num_segments: int | None = None,
) -> EagerTensorProtocol:
    """Execute the eager segment sum operation on the active backend array.

    Args:
        data (BackendArg): Input `data` argument (data).
        segment_ids (BackendArg): Input `segment_ids` argument (segment ids).
        num_segments (int | None): Input `num_segments` argument (num segments).

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    from ml_switcheroo_compiler.backends.eager_registry import coerce_array

    arr_mod = _resolve_eager_mod(None)
    data_arr = coerce_array(arr_mod, data)
    seg_arr = coerce_array(arr_mod, segment_ids)
    if num_segments is None:
        num_segments = int(arr_mod.max(seg_arr)) + 1
    out = arr_mod.zeros((num_segments,) + data_arr.shape[1:], dtype=data_arr.dtype)
    for i in range(num_segments):
        out[i] = arr_mod.sum(data_arr[seg_arr == i], axis=0)
    return out


@global_eager_registry.register("NonMaxSuppression")
def _np_nms(
    backend_module: BackendModuleProtocol | types.ModuleType | None,
    boxes: BackendArg,
    scores: BackendArg,
    max_output_size: BackendArg,
    **kwargs: BackendArg,
) -> BackendArg:
    """Execute the eager np nms operation on the active backend array.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None):
            Input `backend_module` argument (backend module).
        boxes (BackendArg): Input `boxes` argument (boxes).
        scores (BackendArg): Input `scores` argument (scores).
        max_output_size (BackendArg): Input `max_output_size` argument (max output size).
        **kwargs (BackendArg): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    from ml_switcheroo_compiler.backends.eager.vision_filtering import NMSConfig
    from ml_switcheroo_compiler.backends.eager_registry import coerce_float

    nms_cfg = NMSConfig(
        max_output_size=int(coerce_int(max_output_size)),
        iou_threshold=coerce_float(kwargs.get("iou_threshold", 0.5)),
        score_threshold=coerce_float(kwargs.get("score_threshold", float("-inf"))),
    )
    return nms_eager(backend_module, boxes, scores, nms_cfg)


@global_eager_registry.register("TrapezoidalIntegral")
def _np_trapezoidal_integral(
    backend_module: BackendModuleProtocol | types.ModuleType | None, y: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager np trapezoidal integral operation on the active backend array.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None):
            Input `backend_module` argument (backend module).
        y (BackendArg): Input `y` argument (y).
        **kwargs (BackendArg): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    arr_mod = _resolve_eager_mod(backend_module)
    x = kwargs.get("x", None)
    dx = kwargs.get("dx", 1.0)
    axis = kwargs.get("axis", -1)
    trapz_fn = adapt_backend_module(arr_mod).get_symbol(
        "trapezoid", adapt_backend_module(arr_mod).get_symbol("trapz", None)
    )
    if x is not None:
        return require_callable(trapz_fn)(y, x=x, axis=axis)
    return require_callable(trapz_fn)(y, dx=dx, axis=axis)


@global_eager_registry.register("ConfusionMatrix")
def _np_confusion_matrix(
    backend_module: BackendModuleProtocol | types.ModuleType | None,
    labels: BackendArg,
    predictions: BackendArg,
    **kwargs: BackendArg,
) -> BackendArg:
    """Execute the eager np confusion matrix operation on the active backend array.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None):
            Input `backend_module` argument (backend module).
        labels (BackendArg): Input `labels` argument (labels).
        predictions (BackendArg): Input `predictions` argument (predictions).
        **kwargs (BackendArg): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    arr_mod = _resolve_eager_mod(backend_module)
    raw_nc = kwargs.get("num_classes", None)
    weights = kwargs.get("weights", None)
    y_true = adapt_backend_module(labels).require_symbol("flatten")()
    y_pred = adapt_backend_module(predictions).require_symbol("flatten")()
    num_classes_int: int = (
        max(int(coerce_int(arr_mod.max(y_true))), int(coerce_int(arr_mod.max(y_pred)))) + 1
        if raw_nc is None
        else int(coerce_int(raw_nc))
    )
    cm = adapt_backend_module(arr_mod).require_symbol("bincount")(
        y_true * num_classes_int + y_pred,
        weights=weights,
        minlength=num_classes_int**2,
    )
    return adapt_backend_module(cm).require_symbol("reshape")((num_classes_int, num_classes_int))


@global_eager_registry.register("Cummax")
def _np_cummax(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager np cummax operation on the active backend array.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None):
            Input `backend_module` argument (backend module).
        *args (BackendArg): Positional args.
        **kwargs (BackendArg): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    arr_mod = _resolve_eager_mod(backend_module)
    a = args[0]
    axis = kwargs.get("axis", None)
    dtype = kwargs.get("dtype", None)
    if dtype is not None and str(dtype) != "None":
        dtype = adapt_backend_module(dtype).get_symbol("value", dtype)
        return adapt_backend_module(arr_mod.maximum).require_symbol("accumulate")(a, axis=axis, dtype=dtype)
    return adapt_backend_module(arr_mod.maximum).require_symbol("accumulate")(a, axis=axis)


@global_eager_registry.register("Cummin")
def _np_cummin(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager np cummin operation on the active backend array.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None):
            Input `backend_module` argument (backend module).
        *args (BackendArg): Positional args.
        **kwargs (BackendArg): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    arr_mod = _resolve_eager_mod(backend_module)
    a = args[0]
    axis = kwargs.get("axis", None)
    dtype = kwargs.get("dtype", None)
    if dtype is not None and str(dtype) != "None":
        dtype = adapt_backend_module(dtype).get_symbol("value", dtype)
        return adapt_backend_module(arr_mod.minimum).require_symbol("accumulate")(a, axis=axis, dtype=dtype)
    return adapt_backend_module(arr_mod.minimum).require_symbol("accumulate")(a, axis=axis)


@global_eager_registry.register("Cumprod")
def _np_cumprod(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager np cumprod operation on the active backend array.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None):
            Input `backend_module` argument (backend module).
        *args (BackendArg): Positional args.
        **kwargs (BackendArg): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    arr_mod = _resolve_eager_mod(backend_module)
    dtype = kwargs.pop("dtype", None)
    if dtype is not None and str(dtype) != "None":
        kwargs["dtype"] = adapt_backend_module(dtype).get_symbol("value", dtype)
    return adapt_backend_module(arr_mod).require_symbol("cumprod")(*args, **kwargs)


@global_eager_registry.register("Cumlogsumexp")
def _np_cumlogsumexp(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager np cumlogsumexp operation on the active backend array.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None):
            Input `backend_module` argument (backend module).
        *args (BackendArg): Positional args.
        **kwargs (BackendArg): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    arr_mod = _resolve_eager_mod(backend_module)
    a = args[0]
    axis = kwargs.get("axis", None)
    if axis is None:
        return arr_mod.logaddexp.accumulate(adapt_backend_module(arr_mod).require_symbol("ravel")(a))
    return adapt_backend_module(arr_mod.logaddexp).require_symbol("accumulate")(a, axis=axis)


@global_eager_registry.register("ApproxMaxK")
def _np_approx_max_k(
    backend_module: BackendModuleProtocol | types.ModuleType | None,
    x: BackendArg,
    *args: BackendArg,
    **kwargs: BackendArg,
) -> BackendArg:
    """Execute the eager np approx max k operation on the active backend array.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None):
            Input `backend_module` argument (backend module).
        x (BackendArg): Input `x` argument (x).
        *args (BackendArg): Positional args.
        **kwargs (BackendArg): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    arr_mod = _resolve_eager_mod(backend_module)
    k = args[0] if len(args) > 0 else kwargs.get("k", 1)
    reduction_dimension = kwargs.get("reduction_dimension", -1)
    if not adapt_backend_module(x).has_symbol("shape"):
        x = adapt_backend_module(backend_module).require_symbol("array")(x)
    if adapt_backend_module(x).require_symbol("size") == 0:
        return (x, x)
    idx = adapt_backend_module(arr_mod).require_symbol("argsort")(x, axis=reduction_dimension)
    red_dim = int(coerce_int(reduction_dimension))
    dim_len = int(coerce_int(idx.shape[red_dim]))
    idx = adapt_backend_module(arr_mod).require_symbol("take")(
        idx,
        range(dim_len - 1, dim_len - 1 - int(coerce_int(k)), -1),
        axis=reduction_dimension,
    )
    val = adapt_backend_module(arr_mod).require_symbol("take_along_axis")(x, idx, axis=reduction_dimension)
    return (val, idx)


@global_eager_registry.register("ApproxMinK")
def _np_approx_min_k(
    backend_module: BackendModuleProtocol | types.ModuleType | None,
    x: BackendArg,
    *args: BackendArg,
    **kwargs: BackendArg,
) -> BackendArg:
    """Execute the eager np approx min k operation on the active backend array.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None):
            Input `backend_module` argument (backend module).
        x (BackendArg): Input `x` argument (x).
        *args (BackendArg): Positional args.
        **kwargs (BackendArg): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    arr_mod = _resolve_eager_mod(backend_module)
    k = args[0] if len(args) > 0 else kwargs.get("k", 1)
    reduction_dimension = kwargs.get("reduction_dimension", -1)
    if not adapt_backend_module(x).has_symbol("shape"):
        x = adapt_backend_module(backend_module).require_symbol("array")(x)
    if adapt_backend_module(x).require_symbol("size") == 0:
        return (x, x)
    idx = adapt_backend_module(arr_mod).require_symbol("argsort")(x, axis=reduction_dimension)
    idx = adapt_backend_module(arr_mod).require_symbol("take")(idx, range(coerce_int(k)), axis=reduction_dimension)
    val = adapt_backend_module(arr_mod).require_symbol("take_along_axis")(x, idx, axis=reduction_dimension)
    return (val, idx)


def _get_k_val(k: BackendArg) -> int:
    """Extract integer value from k.

    Args:
        k (BackendArg): The value to extract.

    Returns:
        int: Extracted integer.
    """
    if adapt_backend_module(k).has_symbol("item"):
        return int(coerce_int(adapt_backend_module(k).require_symbol("item")()))
    if adapt_backend_module(k).has_symbol("data") and adapt_backend_module(
        adapt_backend_module(k).require_symbol("data")
    ).has_symbol("item"):
        return int(coerce_int(adapt_backend_module(k).require_symbol("data").item()))
    return int(coerce_int(k))


@global_eager_registry.register("TopK")
def _np_top_k(
    backend_module: BackendModuleProtocol | types.ModuleType | None,
    x: BackendArg,
    *args: BackendArg,
    **kwargs: BackendArg,
) -> BackendArg:
    """Execute the eager np top k operation on the active backend array.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None):
            Input `backend_module` argument (backend module).
        x (BackendArg): Input `x` argument (x).
        *args (BackendArg): Positional args.
        **kwargs (BackendArg): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    k_arg = args[0] if len(args) > 0 else kwargs.get("k", 1)
    k = _get_k_val(k_arg)
    axis = coerce_int(kwargs.get("axis", -1))
    return_indices = kwargs.get("return_indices", None)
    if not adapt_backend_module(x).has_symbol("shape"):
        x = adapt_backend_module(backend_module).require_symbol("array")(x)
    x_shape = adapt_backend_module(x).require_symbol("shape")
    kth = max(0, coerce_int(x_shape[axis]) - k)
    if return_indices is False:
        val = adapt_backend_module(backend_module).require_symbol("partition")(x, kth, axis=axis)
        slc: list[BackendArg] = [slice(None)] * len(x_shape)
        slc[axis] = slice(-k, None)
        return val[tuple(slc)]
    idx = adapt_backend_module(backend_module).require_symbol("argpartition")(x, kth, axis=axis)
    slc = [slice(None)] * len(x_shape)
    slc[axis] = slice(-k, None)
    idx_k = idx[tuple(slc)]
    if return_indices is True:
        return idx_k
    val_k = adapt_backend_module(backend_module).require_symbol("take_along_axis")(x, idx_k, axis=axis)
    return (val_k, idx_k)


@global_eager_registry.register("Sum")
def _np_sum(
    backend_module: BackendModuleProtocol | types.ModuleType | None,
    *args: BackendArg,
    **kwargs: int | float | tuple[int, ...] | bool | None,
) -> EagerTensorProtocol:
    """Evaluate sum reduction.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None): Active backend module.
        *args (BackendArg): Positional arguments to pass to sum.
        **kwargs (Union[int, float, tuple[int, ...], bool, None]): Keyword arguments.

    Returns:
        BackendArg: Computed sum reduction.
    """
    return _ensure_arr(adapt_backend_module(backend_module).require_symbol("sum")(*args, **kwargs))


@global_eager_registry.register("Mean")
def _np_mean(
    backend_module: BackendModuleProtocol | types.ModuleType | None,
    *args: BackendArg,
    **kwargs: int | float | tuple[int, ...] | bool | None,
) -> EagerTensorProtocol:
    """Evaluate mean reduction.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None): Active backend module.
        *args (BackendArg): Positional arguments to pass to mean.
        **kwargs (Union[int, float, tuple[int, ...], bool, None]): Keyword arguments.

    Returns:
        BackendArg: Computed mean reduction.
    """
    return _ensure_arr(adapt_backend_module(backend_module).require_symbol("mean")(*args, **kwargs))


@global_eager_registry.register("Max")
def _np_max(
    backend_module: BackendModuleProtocol | types.ModuleType | None,
    *args: BackendArg,
    **kwargs: int | float | tuple[int, ...] | bool | None,
) -> EagerTensorProtocol:
    """Evaluate max reduction.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None): Active backend module.
        *args (BackendArg): Positional arguments to pass to max.
        **kwargs (Union[int, float, tuple[int, ...], bool, None]): Keyword arguments.

    Returns:
        BackendArg: Computed max reduction.
    """
    return _ensure_arr(adapt_backend_module(backend_module).require_symbol("max")(*args, **kwargs))


@global_eager_registry.register("Min")
def _np_min(
    backend_module: BackendModuleProtocol | types.ModuleType | None,
    *args: BackendArg,
    **kwargs: int | float | tuple[int, ...] | bool | None,
) -> EagerTensorProtocol:
    """Evaluate min reduction.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None): Active backend module.
        *args (BackendArg): Positional arguments to pass to min.
        **kwargs (Union[int, float, tuple[int, ...], bool, None]): Keyword arguments.

    Returns:
        BackendArg: Computed min reduction.
    """
    return _ensure_arr(adapt_backend_module(backend_module).require_symbol("min")(*args, **kwargs))


@global_eager_registry.register("Variance")
def _np_variance(
    backend_module: BackendModuleProtocol | types.ModuleType | None,
    *args: BackendArg,
    **kwargs: int | float | tuple[int, ...] | bool | None,
) -> EagerTensorProtocol:
    """Evaluate variance reduction.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None): Active backend module.
        *args (BackendArg): Positional arguments to pass to var.
        **kwargs (Union[int, float, tuple[int, ...], bool, None]): Keyword arguments.

    Returns:
        BackendArg: Computed variance reduction.
    """
    kwargs.setdefault("ddof", 0)
    return _ensure_arr(adapt_backend_module(backend_module).require_symbol("var")(*args, **kwargs))


@global_eager_registry.register("Std")
def _np_std(
    backend_module: BackendModuleProtocol | types.ModuleType | None,
    *args: BackendArg,
    **kwargs: int | float | tuple[int, ...] | bool | None,
) -> EagerTensorProtocol:
    """Evaluate standard deviation reduction.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None): Active backend module.
        *args (BackendArg): Positional arguments to pass to std.
        **kwargs (Union[int, float, tuple[int, ...], bool, None]): Keyword arguments.

    Returns:
        BackendArg: Computed standard deviation reduction.
    """
    return _ensure_arr(adapt_backend_module(backend_module).require_symbol("std")(*args, **kwargs))


@global_eager_registry.register("Argmax")
def _np_argmax(
    backend_module: BackendModuleProtocol | types.ModuleType | None,
    *args: BackendArg,
    **kwargs: int | float | tuple[int, ...] | bool | None,
) -> EagerTensorProtocol:
    """Evaluate argmax reduction.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None): Active backend module.
        *args (BackendArg): Positional arguments to pass to argmax.
        **kwargs (Union[int, float, tuple[int, ...], bool, None]): Keyword arguments.

    Returns:
        BackendArg: Indices of maximum values.
    """
    return _ensure_arr(adapt_backend_module(backend_module).require_symbol("argmax")(*args, **kwargs))


@global_eager_registry.register("Argmin")
def _np_argmin(
    backend_module: BackendModuleProtocol | types.ModuleType | None,
    *args: BackendArg,
    **kwargs: int | float | tuple[int, ...] | bool | None,
) -> EagerTensorProtocol:
    """Evaluate argmin reduction.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None): Active backend module.
        *args (BackendArg): Positional arguments to pass to argmin.
        **kwargs (Union[int, float, tuple[int, ...], bool, None]): Keyword arguments.

    Returns:
        BackendArg: Indices of minimum values.
    """
    return _ensure_arr(adapt_backend_module(backend_module).require_symbol("argmin")(*args, **kwargs))


@global_eager_registry.register("Prod")
def _np_prod(
    backend_module: BackendModuleProtocol | types.ModuleType | None,
    *args: BackendArg,
    **kwargs: int | float | tuple[int, ...] | bool | None,
) -> EagerTensorProtocol:
    """Evaluate product reduction.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None): Active backend module.
        *args (BackendArg): Positional arguments to pass to prod.
        **kwargs (Union[int, float, tuple[int, ...], bool, None]): Keyword arguments.

    Returns:
        BackendArg: Computed product reduction.
    """
    return _ensure_arr(adapt_backend_module(backend_module).require_symbol("prod")(*args, **kwargs))


@global_eager_registry.register("AnyOp")
def _np_any_op(
    backend_module: BackendModuleProtocol | types.ModuleType | None,
    *args: BackendArg,
    **kwargs: int | float | tuple[int, ...] | bool | None,
) -> EagerTensorProtocol:
    """Evaluate any boolean reduction.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None): Active backend module.
        *args (BackendArg): Positional arguments to pass to any.
        **kwargs (Union[int, float, tuple[int, ...], bool, None]): Keyword arguments.

    Returns:
        BackendArg: Boolean reduction result.
    """
    return _ensure_arr(adapt_backend_module(backend_module).require_symbol("any")(*args, **kwargs))


@global_eager_registry.register("Cumsum")
def _np_cumsum(
    backend_module: BackendModuleProtocol | types.ModuleType | None,
    *args: BackendArg,
    **kwargs: int | float | tuple[int, ...] | bool | None,
) -> EagerTensorProtocol:
    """Evaluate cumulative sum.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None): Active backend module.
        *args (BackendArg): Positional arguments to pass to cumsum.
        **kwargs (Union[int, float, tuple[int, ...], bool, None]): Keyword arguments.

    Returns:
        BackendArg: Cumulative sum array.
    """
    return _ensure_arr(adapt_backend_module(backend_module).require_symbol("cumsum")(*args, **kwargs))


@global_eager_registry.register("AddN")
def _np_add_n(
    backend_module: BackendModuleProtocol | types.ModuleType | None,
    inputs: Sequence[BackendArg],
    **kwargs: int | float | bool | None,
) -> EagerTensorProtocol:
    """Evaluate addition of a sequence of tensors.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None): Active backend module.
        inputs (Sequence[BackendArg]): List of tensors to sum.
        **kwargs (Union[int, float, bool, None]): Keyword arguments.

    Returns:
        BackendArg: Element-wise sum of all tensors.

    Raises:
        ValueError: If inputs sequence is empty.
    """
    del kwargs
    if not inputs:
        raise ValueError("inputs must not be empty")
    res: BackendArg = inputs[0]
    for i in range(1, len(inputs)):
        res = adapt_backend_module(backend_module).require_symbol("add")(res, inputs[i])
    return _ensure_arr(res)


@global_eager_registry.register("AccumulateN")
def _np_accumulate_n(
    backend_module: BackendModuleProtocol | types.ModuleType | None,
    inputs: Sequence[BackendArg],
    **kwargs: int | float | bool | None,
) -> EagerTensorProtocol:
    """Evaluate accumulation of a sequence of tensors.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None): Active backend module.
        inputs (Sequence[BackendArg]): List of tensors to accumulate.
        **kwargs (Union[int, float, bool, None]): Keyword arguments.

    Returns:
        BackendArg: Element-wise sum of all tensors.

    Raises:
        ValueError: If inputs sequence is empty.
    """
    del kwargs
    if not inputs:
        raise ValueError("inputs must not be empty")
    res: BackendArg = inputs[0]
    for i in range(1, len(inputs)):
        res = adapt_backend_module(backend_module).require_symbol("add")(res, inputs[i])
    return _ensure_arr(res)


@global_eager_registry.register("CumulativeLogsumexp")
def _np_cumulative_logsumexp(
    backend_module: BackendModuleProtocol | types.ModuleType | None,
    x: BackendArg,
    axis: int = 0,
    **kwargs: int | float | bool | None,
) -> EagerTensorProtocol:
    """Evaluate cumulative logsumexp along a specified axis.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None): Active backend module.
        x (BackendArg): Input tensor.
        axis (int): Axis along which to compute cumulative logsumexp.
        **kwargs (Union[int, float, bool, None]): Keyword arguments.

    Returns:
        BackendArg: Result of cumulative log-sum-exp.
    """
    del kwargs
    exp_x = adapt_backend_module(backend_module).require_symbol("exp")(x)
    cumsum_exp = adapt_backend_module(backend_module).require_symbol("cumsum")(exp_x, axis=axis)
    return _ensure_arr(adapt_backend_module(backend_module).require_symbol("log")(cumsum_exp))


@global_eager_registry.register("ReducePrecision")
def _np_reduce_precision(
    backend_module: BackendModuleProtocol | types.ModuleType | None,
    x: BackendArg,
    exponent_bits: int,
    mantissa_bits: int,
) -> EagerTensorProtocol:
    """Reduce the precision of a tensor to a specified number of exponent and mantissa bits.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None): The backend module.
        x (BackendArg): Input array.
        exponent_bits (int): The exponent bits count.
        mantissa_bits (int): The mantissa bits count.

    Returns:
        BackendArg: The computed result.
    """
    arr_mod = _resolve_eager_mod(backend_module)
    del backend_module, exponent_bits, mantissa_bits
    return arr_mod.asarray(x).astype(arr_mod.float16).astype(arr_mod.asarray(x).dtype)


@global_eager_registry.register("SparseReduceMax")
def _np_sparsereducemax(
    backend_module: BackendModuleProtocol | types.ModuleType | None,
    *args: BackendArg,
    **kwargs: float | int | str | bool | None,
) -> EagerTensorProtocol:
    """Implement SparseReduceMax.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None): The backend module.
        *args (BackendArg): Positional args.
        **kwargs (Union[float, int, str, bool, None]): Keyword args.

    Returns:
        BackendArg: Result array.
    """
    del kwargs
    return _ensure_arr(adapt_backend_module(backend_module).require_symbol("max")(args[0], axis=-1))
