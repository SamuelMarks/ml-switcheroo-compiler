"""Core utilities."""

import builtins  # noqa: F401
import typing  # noqa: F401
from types import ModuleType
from typing import Optional, cast

from ml_switcheroo_compiler.backends.eager_registry import (
    EagerTensorProtocol,
    EagerValue,
    global_eager_registry,
)


def _get_reduction_axes(reshaped_dims: list[int], axis: int) -> tuple[int, ...]:
    """Execute the eager get reduction axes operation on the active backend array.

    Args:
        reshaped_dims (list[int]): The reshaped_dims parameter.
        axis (int): The axis parameter.

    Returns:
        tuple[int, ...]: Result.
    """
    return tuple(i for i in range(len(reshaped_dims)) if i not in (0, axis))


def _invoke_grouped_op(backend_module: EagerValue, op_name: str, reshaped_x: EagerValue, reduction_axes: tuple[int, ...]) -> EagerValue:
    """Execute the eager invoke grouped op operation on the active backend array.

    Args:
        backend_module: The backend_module parameter.
        op_name (str): The op_name parameter.
        reshaped_x: The reshaped_x parameter.
        reduction_axes (tuple[int, ...]): The reduction_axes parameter.

    Returns:
            EagerValue: Result.

    Raises:
        ValueError: An exception.
    """
    mod = cast(ModuleType, backend_module)
    if op_name == "mean":
        try:
            return cast(EagerValue, mod.mean(reshaped_x, axis=reduction_axes, keepdims=True))
        except TypeError:
            return cast(EagerValue, mod.mean(reshaped_x, dim=reduction_axes, keepdim=True))
    if op_name == "variance":
        try:
            return cast(EagerValue, mod.var(reshaped_x, axis=reduction_axes, keepdims=True))
        except TypeError:
            return cast(EagerValue, mod.var(reshaped_x, dim=reduction_axes, keepdim=True, unbiased=False))
    msg: str = f"Unknown grouped reduction op: {op_name}"
    raise ValueError(msg)


def _apply_grouped_reduction(backend_module: EagerValue, op_name: str, x: EagerValue, **kwargs: int) -> EagerValue:
    """Execute the eager apply grouped reduction operation on the active backend array.

    Args:
        backend_module: The backend_module parameter.
        op_name (str): The op_name parameter.
        x: The x parameter.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Result.
    """
    mod = cast(ModuleType, backend_module)
    groups: int = kwargs["groups"]
    axis: int = kwargs["axis"]
    shape: list[int] = list(cast(EagerTensorProtocol, x).shape)
    ndims: int = len(shape)
    if axis < 0:
        axis += ndims
    C: int = shape[axis]
    C_per_group: int = C // groups
    reshaped_dims: list[int] = shape.copy()
    reshaped_dims[axis : axis + 1] = [groups, C_per_group]
    reshaped_x = mod.reshape(x, reshaped_dims)
    reduction_axes: tuple[int, ...] = _get_reduction_axes(reshaped_dims, axis)
    return _invoke_grouped_op(backend_module, op_name, reshaped_x, reduction_axes)


@global_eager_registry.register("GroupMean")
def _group_mean(backend_module: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager group mean operation on the active backend array.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Result.
    """
    x = args[0]
    groups: int = int(cast(int, kwargs.get("groups") if "groups" in kwargs else getattr(args[1], "__int__", lambda: int(cast(int, str(args[1]))))()))
    axis: int = int(cast(int, kwargs.get("axis", -1)))
    return _apply_grouped_reduction(backend_module, "mean", x, groups=groups, axis=axis)


@global_eager_registry.register("GroupVariance")
def _group_variance(backend_module: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager group variance operation on the active backend array.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Result.
    """
    x = args[0]
    groups: int = int(cast(int, kwargs.get("groups") if "groups" in kwargs else getattr(args[1], "__int__", lambda: int(cast(int, str(args[1]))))()))
    axis: int = int(cast(int, kwargs.get("axis", -1)))
    return _apply_grouped_reduction(backend_module, "variance", x, groups=groups, axis=axis)


def _apply_affine_transform(backend_module: EagerValue, out: EagerValue, axis: int, **kwargs: EagerValue) -> EagerValue:
    """Apply affine transform scaling to normalized output.

    Args:
        backend_module: The backend_module parameter.
        out: The out parameter.
        axis (int): The axis parameter.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Result.
    """
    mod = cast(ModuleType, backend_module)
    weight = kwargs.get("weight")
    bias = kwargs.get("bias")
    out_t = cast(EagerTensorProtocol, out)
    shape: list[int] = list(out_t.shape)
    ndims: int = len(shape)
    if weight is not None:
        w_shape: list[int] = [1] * ndims
        w_shape[axis] = shape[axis]
        w: EagerTensorProtocol = mod.reshape(weight, w_shape)
        out_t = out_t * w
    if bias is not None:
        b_shape: list[int] = [1] * ndims
        b_shape[axis] = shape[axis]
        b: EagerTensorProtocol = mod.reshape(bias, b_shape)
        out_t = out_t + b
    return out_t


def _parse_group_norm_args(args: tuple[EagerValue, ...], kwargs: dict[str, EagerValue]) -> tuple[EagerValue, int, Optional[EagerValue], Optional[EagerValue], int, float]:
    """Execute the eager parse group norm args operation on the active backend array.

    Args:
        args: The args parameter.
        kwargs: The kwargs parameter.

    Returns:
        tuple[object, int, Optional[object], Optional[object], int, float]: Result.
    """
    x = args[0]
    groups: int = int(cast(int, kwargs.get("groups") if "groups" in kwargs else getattr(args[1], "__int__", lambda: int(cast(int, str(args[1]))))()))
    weight = kwargs.get("weight", None)
    bias = kwargs.get("bias", None)
    axis: int = int(cast(int, kwargs.get("axis", -1)))
    epsilon: float = float(cast(float, kwargs.get("epsilon", 1e-05)))
    return (x, groups, weight, bias, axis, epsilon)


def _compute_group_norm(backend_module: EagerValue, x: EagerValue, shape: list[int], group_params: tuple[int, int], stats: tuple[EagerValue, EagerValue, float]) -> EagerValue:
    """Execute the eager compute group norm operation on the active backend array.

    Args:
        backend_module: The backend_module parameter.
        x: The x parameter.
        shape (list[int]): The shape parameter.
        group_params (tuple[int, int]): The group_params parameter.
        stats: The stats parameter.

    Returns:
            EagerValue: Result.
    """
    mod = cast(ModuleType, backend_module)
    axis, groups = group_params
    mean, var, epsilon = stats
    C_per_group: int = shape[axis] // groups
    reshaped_dims: list[int] = shape.copy()
    reshaped_dims[axis : axis + 1] = [groups, C_per_group]
    reshaped_x: EagerTensorProtocol = mod.reshape(x, reshaped_dims)
    normalized = (reshaped_x - cast(EagerTensorProtocol, mean)) / mod.sqrt(cast(EagerTensorProtocol, var) + cast(EagerTensorProtocol, epsilon))
    return cast(EagerValue, mod.reshape(normalized, shape))


@global_eager_registry.register("GroupNorm")
def _group_norm(backend_module: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager group norm operation on the active backend array.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Result.
    """
    x, groups, weight, bias, axis, epsilon = _parse_group_norm_args(args, kwargs)
    shape: list[int] = list(cast(EagerTensorProtocol, x).shape)
    ndims: int = len(shape)
    if axis < 0:
        axis += ndims
    mean = _group_mean(backend_module, x, groups=groups, axis=axis)
    var = _group_variance(backend_module, x, groups=groups, axis=axis)
    out = _compute_group_norm(backend_module, x, shape, (axis, groups), (mean, var, epsilon))
    return _apply_affine_transform(backend_module, out, axis, weight=weight, bias=bias)
