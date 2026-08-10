# ruff: noqa: E402, D100, D103, D104, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, D101, D102, D107, E701, E722, F403, E711, E712, PLR0913, PLR0915
"""Core utilities."""

from typing import Any

from ml_switcheroo_compiler.backends.eager_registry import global_eager_registry


def _get_reduction_axes(reshaped_dims: list, axis: int) -> tuple:
    """Evaluate _get_reduction_axes operation.

    Args:
        reshaped_dims (list): The reshaped_dims parameter.
        axis (int): The axis parameter.

    Returns:
        tuple: Result.
    """
    return tuple(i for i in range(len(reshaped_dims)) if i not in (0, axis))


def _invoke_grouped_op(backend_module: Any, op_name: str, reshaped_x: Any, reduction_axes: tuple) -> Any:
    """Evaluate _invoke_grouped_op operation.

    Args:
        backend_module (object): The backend_module parameter.
        op_name (str): The op_name parameter.
        reshaped_x (object): The reshaped_x parameter.
        reduction_axes (tuple): The reduction_axes parameter.

    Returns: Any: Result.

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
    msg = f"Unknown grouped reduction op: {op_name}"
    raise ValueError(msg)


def _apply_grouped_reduction(backend_module: Any, op_name: str, x: Any, **kwargs: int) -> Any:
    """Evaluate _apply_grouped_reduction operation.

    Args:
        backend_module (object): The backend_module parameter.
        op_name (str): The op_name parameter.
        x (object): The x parameter.
        **kwargs (int): Keyword args.

    Returns: Any: Result.
    """
    groups = kwargs["groups"]
    axis = kwargs["axis"]
    shape = list(x.shape)
    ndims = len(shape)
    if axis < 0:
        axis += ndims
    C = shape[axis]
    C_per_group = C // groups
    reshaped_dims = shape.copy()
    reshaped_dims[axis : axis + 1] = [groups, C_per_group]
    reshaped_x = backend_module.reshape(x, reshaped_dims)
    reduction_axes = _get_reduction_axes(reshaped_dims, axis)
    return _invoke_grouped_op(backend_module, op_name, reshaped_x, reduction_axes)


@global_eager_registry.register("GroupMean")
def _group_mean(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _group_mean operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    x = args[0]
    groups = kwargs.get("groups") if "groups" in kwargs else args[1]
    axis = kwargs.get("axis", -1)
    return _apply_grouped_reduction(backend_module, "mean", x, groups=groups, axis=axis)  # type: ignore  # Justification: Polymorphic / Duck Typing for Framework Agnosticism


@global_eager_registry.register("GroupVariance")
def _group_variance(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _group_variance operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    x = args[0]
    groups = kwargs.get("groups") if "groups" in kwargs else args[1]
    axis = kwargs.get("axis", -1)
    return _apply_grouped_reduction(backend_module, "variance", x, groups=groups, axis=axis)  # type: ignore  # Justification: Polymorphic / Duck Typing for Framework Agnosticism


def _apply_affine_transform(backend_module: Any, out: Any, axis: int, **kwargs: Any) -> Any:
    """Apply affine transform scaling to normalized output.

    Args:
        backend_module (object): The backend_module parameter.
        out (object): The out parameter.
        axis (int): The axis parameter.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    weight = kwargs.get("weight")
    bias = kwargs.get("bias")
    shape = out.shape
    ndims = len(shape)
    if weight is not None:
        w_shape = [1] * ndims
        w_shape[axis] = shape[axis]
        w = backend_module.reshape(weight, w_shape)
        out = out * w
    if bias is not None:
        b_shape = [1] * ndims
        b_shape[axis] = shape[axis]
        b = backend_module.reshape(bias, b_shape)
        out = out + b
    return out


def _parse_group_norm_args(args: tuple, kwargs: dict) -> tuple:
    """Evaluate _parse_group_norm_args operation.

    Args:
        args (tuple): The args parameter.
        kwargs (dict): The kwargs parameter.

    Returns:
        tuple: Result.
    """
    x = args[0]
    groups = kwargs.get("groups") if "groups" in kwargs else args[1]
    weight = kwargs.get("weight", None)
    bias = kwargs.get("bias", None)
    axis = kwargs.get("axis", -1)
    epsilon = kwargs.get("epsilon", 1e-05)
    return (x, groups, weight, bias, axis, epsilon)


def _compute_group_norm(backend_module: Any, x: Any, shape: list, group_params: tuple[int, int], stats: tuple[Any, Any, float]) -> Any:
    """Evaluate _compute_group_norm operation.

    Args:
        backend_module (object): The backend_module parameter.
        x (object): The x parameter.
        shape (list): The shape parameter.
        group_params (tuple): The group_params parameter.
        stats (tuple): The stats parameter.

    Returns: Any: Result.
    """
    (axis, groups) = group_params
    (mean, var, epsilon) = stats
    C_per_group = shape[axis] // groups
    reshaped_dims = shape.copy()
    reshaped_dims[axis : axis + 1] = [groups, C_per_group]
    reshaped_x = backend_module.reshape(x, reshaped_dims)
    normalized = (reshaped_x - mean) / backend_module.sqrt(var + epsilon)
    return backend_module.reshape(normalized, shape)


@global_eager_registry.register("GroupNorm")
def _group_norm(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _group_norm operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    (x, groups, weight, bias, axis, epsilon) = _parse_group_norm_args(args, kwargs)
    shape = list(x.shape)
    ndims = len(shape)
    if axis < 0:
        axis += ndims
    mean = _group_mean(backend_module, x, groups=groups, axis=axis)
    var = _group_variance(backend_module, x, groups=groups, axis=axis)
    out = _compute_group_norm(backend_module, x, shape, (axis, groups), (mean, var, epsilon))
    return _apply_affine_transform(backend_module, out, axis, weight=weight, bias=bias)


__all__ = [
    "__cached__",
    "__doc__",
    "__file__",
    "__loader__",
    "__name__",
    "__package__",
    "__spec__",
    "_apply_affine_transform",
    "_apply_grouped_reduction",
    "_compute_group_norm",
    "_get_reduction_axes",
    "_group_mean",
    "_group_norm",
    "_group_variance",
    "_invoke_grouped_op",
    "_parse_group_norm_args",
    "global_eager_registry",
]
