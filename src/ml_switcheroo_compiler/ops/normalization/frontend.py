# ruff: noqa: E402, D100, D103, D104, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, D101, D102, D107, E701, E722, F403, E711, E712, PLR0913, PLR0915
"""Apply normalization frontend operations."""

import typing
from dataclasses import dataclass
from typing import Any

from ml_switcheroo_compiler.ops.base import get_op
from ml_switcheroo_compiler.ops.binary import divide
from ml_switcheroo_compiler.ops.linalg import power_iteration


@dataclass
class NormConfig:
    """Configuration for normalization operations.

    Attributes:
        weight (object): Optional scale tensor.
        bias (object): Optional shift tensor.
        epsilon (float): Small value to avoid division by zero.
    """

    weight: Any = None
    bias: Any = None
    epsilon: float = 1e-5


def group_mean(
    x: Any,
    groups: int,
    axis: typing.Union[int, tuple[int, ...]] = -1,
    keepdims: bool = False,
) -> Any:
    """Compute the mean over groups.

    Args:
        x (object): The x parameter.
        groups (int): The groups parameter.
        axis (object): The axis parameter.
        keepdims (bool): The keepdims parameter.

    Returns: Any: Result.
    """
    return get_op("GroupMean")()(x, groups=groups, axis=axis, keepdims=keepdims)


def group_variance(
    x: Any,
    groups: int,
    axis: typing.Union[int, tuple[int, ...]] = -1,
    keepdims: bool = False,
) -> Any:
    """Compute the variance over groups.

    Args:
        x (object): The x parameter.
        groups (int): The groups parameter.
        axis (object): The axis parameter.
        keepdims (bool): The keepdims parameter.

    Returns: Any: Result.
    """
    return get_op("GroupVariance")()(x, groups=groups, axis=axis, keepdims=keepdims)


def group_norm(
    x: Any,
    groups: int,
    config: typing.Optional[NormConfig] = None,
    axis: typing.Union[int, tuple[int, ...]] = -1,
) -> Any:
    """Compute the group normalization.

    Args:
        x (object): Input tensor.
        groups (int): Number of groups.
        config (Optional[NormConfig]): Normalization configuration.
        axis (Union[int, tuple[int, ...]]): Axis to normalize over.

    Returns: Any: Normalized tensor.
    """
    if config is None:
        config = NormConfig()
    return get_op("GroupNorm")()(x, groups=groups, weight=config.weight, bias=config.bias, axis=axis, epsilon=config.epsilon)


def spectral_normalization(
    w: Any,
    u: Any,
    num_iters: int = 1,
) -> tuple[Any, Any]:
    """Compute the spectral normalization.

    Args:
        w (object): Weight tensor.
        u (object): Left singular vector estimate.
        num_iters (int): Number of power iterations.

    Returns:
        tuple[Any, Any]: Normalized weight and new u.
    """
    _, u_new, sigma = power_iteration(w, num_iters=num_iters, u=u)
    return divide(w, sigma), u_new
