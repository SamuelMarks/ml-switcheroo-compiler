"""Normalization frontend operations."""

import typing
from dataclasses import dataclass

from ml_switcheroo_compiler.ops.base import get_op
from ml_switcheroo_compiler.ops.binary import divide  # pragma: no cover
from ml_switcheroo_compiler.ops.linalg import power_iteration  # pragma: no cover


@dataclass
class NormConfig:
    """Configuration for normalization operations.

    Attributes:
        weight (object): Optional scale tensor.
        bias (object): Optional shift tensor.
        epsilon (float): Small value to avoid division by zero.
    """

    weight: object = None
    bias: object = None
    epsilon: float = 1e-5


def group_mean(
    x: object,
    groups: int,
    axis: typing.Union[int, tuple[int, ...]] = -1,
    keepdims: bool = False,
) -> object:
    """Computes the mean over groups."""
    return get_op("GroupMean")()(x, groups=groups, axis=axis, keepdims=keepdims)  # pragma: no cover


def group_variance(
    x: object,
    groups: int,
    axis: typing.Union[int, tuple[int, ...]] = -1,
    keepdims: bool = False,
) -> object:
    """Computes the variance over groups."""
    return get_op("GroupVariance")()(x, groups=groups, axis=axis, keepdims=keepdims)  # pragma: no cover


def group_norm(
    x: object,
    groups: int,
    config: typing.Optional[NormConfig] = None,
    axis: typing.Union[int, tuple[int, ...]] = -1,
) -> object:
    """Computes the group normalization.

    Args:
        x (object): Input tensor.
        groups (int): Number of groups.
        config (Optional[NormConfig]): Normalization configuration.
        axis (Union[int, tuple[int, ...]]): Axis to normalize over.

    Returns:
        object: Normalized tensor.
    """
    if config is None:  # pragma: no cover
        config = NormConfig()  # pragma: no cover
    return get_op("GroupNorm")()(  # pragma: no cover
        x, groups=groups, weight=config.weight, bias=config.bias, axis=axis, epsilon=config.epsilon
    )


def spectral_normalization(
    w: object,
    u: object,
    num_iters: int = 1,
) -> tuple[object, object]:
    """Computes the spectral normalization.

    Args:
        w (object): Weight tensor.
        u (object): Left singular vector estimate.
        num_iters (int): Number of power iterations.

    Returns:
        tuple[object, object]: Normalized weight and new u.
    """
    _, u_new, sigma = power_iteration(w, num_iters=num_iters, u=u)  # pragma: no cover
    return divide(w, sigma), u_new  # pragma: no cover
