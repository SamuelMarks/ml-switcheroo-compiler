# ruff: noqa: E402, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, E701, E722, F403, E711, E712, PLR0913, PLR0915
"""Core utilities."""

import builtins
import typing
from typing import Any, Optional

from ml_switcheroo_compiler.backends.eager_registry import global_eager_registry


def _get_reduction_axes(reshaped_dims: list[int], axis: int) -> tuple[int, ...]:
    """Evaluate _get_reduction_axes operation.

    Args:
        reshaped_dims (list[int]): The reshaped_dims parameter.
        axis (int): The axis parameter.

    Returns:
        tuple[int, ...]: Result.
    """
    return tuple(i for i in range(len(reshaped_dims)) if i not in (0, axis))


def _invoke_grouped_op(backend_module: Any, op_name: str, reshaped_x: object, reduction_axes: tuple[int, ...]) -> Any:
    """Evaluate _invoke_grouped_op operation.

    Args:
        backend_module: The backend_module parameter.
        op_name (str): The op_name parameter.
        reshaped_x: The reshaped_x parameter.
        reduction_axes (tuple[int, ...]): The reduction_axes parameter.

    Returns:
            object: Result.

    Raises:
        ValueError: An exception.
    """
    if op_name == "mean":
        try:
            return backend_module.mean(reshaped_x, axis=reduction_axes, keepdims=True)
        except TypeError:
            return backend_module.mean(reshaped_x, dim=reduction_axes, keepdim=True)
    if op_name == "variance":
        try:
            return backend_module.var(reshaped_x, axis=reduction_axes, keepdims=True)
        except TypeError:
            return backend_module.var(reshaped_x, dim=reduction_axes, keepdim=True, unbiased=False)
    msg: str = f"Unknown grouped reduction op: {op_name}"
    raise ValueError(msg)


def _apply_grouped_reduction(backend_module: Any, op_name: str, x: object, **kwargs: int) -> Any:
    """Evaluate _apply_grouped_reduction operation.

    Args:
        backend_module: The backend_module parameter.
        op_name (str): The op_name parameter.
        x: The x parameter.
        **kwargs: Keyword args.

    Returns:
            object: Result.
    """
    groups: int = kwargs["groups"]
    axis: int = kwargs["axis"]
    shape: list[int] = list(x.shape)
    ndims: int = len(shape)
    if axis < 0:
        axis += ndims
    C: int = shape[axis]
    C_per_group: int = C // groups
    reshaped_dims: list[int] = shape.copy()
    reshaped_dims[axis : axis + 1] = [groups, C_per_group]
    reshaped_x = backend_module.reshape(x, reshaped_dims)
    reduction_axes: tuple[int, ...] = _get_reduction_axes(reshaped_dims, axis)
    return _invoke_grouped_op(backend_module, op_name, reshaped_x, reduction_axes)


@global_eager_registry.register("GroupMean")
def _group_mean(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _group_mean operation.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            object: Result.
    """
    x = args[0]
    groups: int = int(kwargs.get("groups") if "groups" in kwargs else getattr(args[1], "__int__", lambda: int(str(args[1])))())
    axis: int = int(kwargs.get("axis", -1))
    return _apply_grouped_reduction(backend_module, "mean", x, groups=groups, axis=axis)


@global_eager_registry.register("GroupVariance")
def _group_variance(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _group_variance operation.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            object: Result.
    """
    x = args[0]
    groups: int = int(kwargs.get("groups") if "groups" in kwargs else getattr(args[1], "__int__", lambda: int(str(args[1])))())
    axis: int = int(kwargs.get("axis", -1))
    return _apply_grouped_reduction(backend_module, "variance", x, groups=groups, axis=axis)


def _apply_affine_transform(backend_module: Any, out: object, axis: int, **kwargs: Any) -> Any:
    """Apply affine transform scaling to normalized output.

    Args:
        backend_module: The backend_module parameter.
        out: The out parameter.
        axis (int): The axis parameter.
        **kwargs: Keyword args.

    Returns:
            object: Result.
    """
    weight = kwargs.get("weight")
    bias = kwargs.get("bias")
    shape: list[int] = list(out.shape)
    ndims: int = len(shape)
    if weight is not None:
        w_shape: list[int] = [1] * ndims
        w_shape[axis] = shape[axis]
        w = backend_module.reshape(weight, w_shape)
        out = out * w
    if bias is not None:
        b_shape: list[int] = [1] * ndims
        b_shape[axis] = shape[axis]
        b = backend_module.reshape(bias, b_shape)
        out = out + b
    return out


def _parse_group_norm_args(args: tuple[object, ...], kwargs: dict[str, object]) -> tuple[object, int, Optional[object], Optional[object], int, float]:
    """Evaluate _parse_group_norm_args operation.

    Args:
        args: The args parameter.
        kwargs: The kwargs parameter.

    Returns:
        tuple[object, int, Optional[object], Optional[object], int, float]: Result.
    """
    x = args[0]
    groups: int = int(kwargs.get("groups") if "groups" in kwargs else getattr(args[1], "__int__", lambda: int(str(args[1])))())
    weight = kwargs.get("weight", None)
    bias = kwargs.get("bias", None)
    axis: int = int(kwargs.get("axis", -1))
    epsilon: float = float(kwargs.get("epsilon", 1e-05))
    return (x, groups, weight, bias, axis, epsilon)


def _compute_group_norm(backend_module: Any, x: object, shape: list[int], group_params: tuple[int, int], stats: tuple[object, object, float]) -> Any:
    """Evaluate _compute_group_norm operation.

    Args:
        backend_module: The backend_module parameter.
        x: The x parameter.
        shape (list[int]): The shape parameter.
        group_params (tuple[int, int]): The group_params parameter.
        stats: The stats parameter.

    Returns:
            object: Result.
    """
    axis, groups = group_params
    mean, var, epsilon = stats
    C_per_group: int = shape[axis] // groups
    reshaped_dims: list[int] = shape.copy()
    reshaped_dims[axis : axis + 1] = [groups, C_per_group]
    reshaped_x = backend_module.reshape(x, reshaped_dims)
    normalized = (reshaped_x - mean) / backend_module.sqrt(var + epsilon)
    return backend_module.reshape(normalized, shape)


@global_eager_registry.register("GroupNorm")
def _group_norm(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _group_norm operation.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            object: Result.
    """
    x, groups, weight, bias, axis, epsilon = _parse_group_norm_args(args, kwargs)
    shape: list[int] = list(x.shape)
    ndims: int = len(shape)
    if axis < 0:
        axis += ndims
    mean = _group_mean(backend_module, x, groups=groups, axis=axis)
    var = _group_variance(backend_module, x, groups=groups, axis=axis)
    out = _compute_group_norm(backend_module, x, shape, (axis, groups), (mean, var, epsilon))
    return _apply_affine_transform(backend_module, out, axis, weight=weight, bias=bias)
