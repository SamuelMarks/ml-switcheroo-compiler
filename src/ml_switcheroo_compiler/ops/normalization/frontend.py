"""Normalization frontend operations."""

import typing
from ml_switcheroo_compiler.ops.base import get_op


def group_mean(
    x: object,
    groups: int,
    axis: typing.Union[int, tuple[int, ...]] = -1,
    keepdims: bool = False,
) -> object:
    """Computes the mean over groups."""
    return get_op("GroupMean")()(x, groups=groups, axis=axis, keepdims=keepdims)


def group_variance(
    x: object,
    groups: int,
    axis: typing.Union[int, tuple[int, ...]] = -1,
    keepdims: bool = False,
) -> object:
    """Computes the variance over groups."""
    return get_op("GroupVariance")()(x, groups=groups, axis=axis, keepdims=keepdims)


def group_norm(
    x: object,
    groups: int,
    weight: object = None,
    bias: object = None,
    axis: typing.Union[int, tuple[int, ...]] = -1,
    epsilon: float = 1e-5,
) -> object:
    """Computes the group normalization."""
    return get_op("GroupNorm")()(
        x, groups=groups, weight=weight, bias=bias, axis=axis, epsilon=epsilon
    )
