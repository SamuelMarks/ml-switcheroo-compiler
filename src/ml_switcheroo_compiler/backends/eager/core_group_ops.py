# ruff: noqa: F405, F403
"""Core utilities."""

from ml_switcheroo_compiler.backends.eager_registry import global_eager_registry


def _get_reduction_axes(reshaped_dims: list, axis: int) -> tuple:
    """Function docstring.

    Args:
        reshaped_dims: Arg.
        axis: Arg.
    """
    return tuple(i for i in range(len(reshaped_dims)) if i not in (0, axis))


def _invoke_grouped_op(
    backend_module: object, op_name: str, reshaped_x: object, reduction_axes: tuple, is_torch: bool
) -> object:
    """Function docstring.

    Args:
        backend_module: Arg.
        op_name: Arg.
        reshaped_x: Arg.
        reduction_axes: Arg.
        is_torch: Arg.
    """
    if op_name == "mean":
        if is_torch:  # pragma: no branch
            return backend_module.mean(
                reshaped_x, dim=reduction_axes, keepdim=True
            )  # pragma: no cover
        return backend_module.mean(reshaped_x, axis=reduction_axes, keepdims=True)
    if op_name == "variance":  # pragma: no branch
        if is_torch:  # pragma: no branch
            return backend_module.var(
                reshaped_x, dim=reduction_axes, keepdim=True, unbiased=False
            )  # pragma: no cover
        return backend_module.var(reshaped_x, axis=reduction_axes, keepdims=True)
    msg = f"Unknown grouped reduction op: {op_name}"  # pragma: no cover
    raise ValueError(msg)  # pragma: no cover


def _apply_grouped_reduction(
    backend_module: object, op_name: str, x: object, **kwargs: int
) -> object:
    """Function docstring.

    Args:
        backend_module: Arg.
        op_name: Arg.
        x: Arg.
        kwargs: Arg.
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

    is_torch = backend_module.__name__ == "torch"
    return _invoke_grouped_op(backend_module, op_name, reshaped_x, reduction_axes, is_torch)


@global_eager_registry.register("GroupMean")
def _group_mean(backend_module: object, *args: object, **kwargs: object) -> object:
    """Function docstring.

    Args:
        backend_module: Arg.
        args: Arg.
        kwargs: Arg.
    """
    x = args[0]
    groups = kwargs.get("groups") if "groups" in kwargs else args[1]
    axis = kwargs.get("axis", -1)
    return _apply_grouped_reduction(backend_module, "mean", x, groups=groups, axis=axis)


@global_eager_registry.register("GroupVariance")
def _group_variance(backend_module: object, *args: object, **kwargs: object) -> object:
    """Function docstring.

    Args:
        backend_module: Arg.
        args: Arg.
        kwargs: Arg.
    """
    x = args[0]
    groups = kwargs.get("groups") if "groups" in kwargs else args[1]
    axis = kwargs.get("axis", -1)
    return _apply_grouped_reduction(backend_module, "variance", x, groups=groups, axis=axis)


def _apply_affine_transform(
    backend_module: object, out: object, axis: int, **kwargs: object
) -> object:
    """Apply affine transform scaling to normalized output."""
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
    """Function docstring.

    Args:
        args: Arg.
        kwargs: Arg.
    """
    x = args[0]
    groups = kwargs.get("groups") if "groups" in kwargs else args[1]
    weight = kwargs.get("weight", None)
    bias = kwargs.get("bias", None)
    axis = kwargs.get("axis", -1)
    epsilon = kwargs.get("epsilon", 1e-5)
    return x, groups, weight, bias, axis, epsilon


def _compute_group_norm(
    backend_module: object,
    x: object,
    shape: list,
    group_params: tuple[int, int],
    stats: tuple[object, object, float],
) -> object:
    """Function docstring.

    Args:
        backend_module: Arg.
        x: Arg.
        shape: Arg.
        group_params: Arg.
        stats: Arg.
    """
    axis, groups = group_params
    mean, var, epsilon = stats
    C_per_group = shape[axis] // groups
    reshaped_dims = shape.copy()
    reshaped_dims[axis : axis + 1] = [groups, C_per_group]

    reshaped_x = backend_module.reshape(x, reshaped_dims)
    normalized = (reshaped_x - mean) / backend_module.sqrt(var + epsilon)
    return backend_module.reshape(normalized, shape)


@global_eager_registry.register("GroupNorm")
def _group_norm(backend_module: object, *args: object, **kwargs: object) -> object:
    """Function docstring.

    Args:
        backend_module: Arg.
        args: Arg.
        kwargs: Arg.
    """
    x, groups, weight, bias, axis, epsilon = _parse_group_norm_args(args, kwargs)

    shape = list(x.shape)
    ndims = len(shape)
    if axis < 0:  # pragma: no branch
        axis += ndims

    mean = _group_mean(backend_module, x, groups=groups, axis=axis)
    var = _group_variance(backend_module, x, groups=groups, axis=axis)

    out = _compute_group_norm(backend_module, x, shape, (axis, groups), (mean, var, epsilon))

    return _apply_affine_transform(backend_module, out, axis, weight=weight, bias=bias)


__all__ = [n for n in globals().keys() if n != "__builtins__"]
