# ruff: noqa: E402, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, E701, E722, F403, E711, E712, PLR0913, PLR0915
"""Apply normalization frontend operations."""

from __future__ import annotations

from dataclasses import dataclass

from ml_switcheroo_compiler.core.tensor import Tensor
from ml_switcheroo_compiler.ops.base import get_op
from ml_switcheroo_compiler.ops.binary import divide
from ml_switcheroo_compiler.ops.linalg import power_iteration


@dataclass
class NormConfig:
    """Configuration for normalization operations.

    Attributes:
        weight (Tensor | None): Optional scale tensor.
        bias (Tensor | None): Optional shift tensor.
        epsilon (float): Small value to avoid division by zero.
    """

    weight: Tensor | None = None
    bias: Tensor | None = None
    epsilon: float = 1e-5


def group_mean(
    x: Tensor,
    groups: int,
    axis: int | tuple[int, ...] = -1,
    keepdims: bool = False,
) -> Tensor:
    """Compute the mean over groups.

    Args:
        x (Tensor): Input tensor.
        groups (int): The number of groups.
        axis (int | tuple[int, ...]): The reduction axis or axes.
        keepdims (bool): Whether to keep reduced dimensions.

    Returns:
        Tensor: Group mean result.
    """
    return get_op("GroupMean")()(x, groups=groups, axis=axis, keepdims=keepdims)


def group_variance(
    x: Tensor,
    groups: int,
    axis: int | tuple[int, ...] = -1,
    keepdims: bool = False,
) -> Tensor:
    """Compute the variance over groups.

    Args:
        x (Tensor): Input tensor.
        groups (int): The number of groups.
        axis (int | tuple[int, ...]): The reduction axis or axes.
        keepdims (bool): Whether to keep reduced dimensions.

    Returns:
        Tensor: Group variance result.
    """
    return get_op("GroupVariance")()(x, groups=groups, axis=axis, keepdims=keepdims)


def group_norm(
    x: Tensor,
    groups: int,
    config: NormConfig | None = None,
    axis: int | tuple[int, ...] = -1,
) -> Tensor:
    """Compute the group normalization.

    Args:
        x (Tensor): Input tensor.
        groups (int): Number of groups.
        config (NormConfig | None): Normalization configuration.
        axis (int | tuple[int, ...]): Axis to normalize over.

    Returns:
        Tensor: Normalized tensor.
    """
    if config is None:
        config = NormConfig()
    return get_op("GroupNorm")()(x, groups=groups, weight=config.weight, bias=config.bias, axis=axis, epsilon=config.epsilon)


def spectral_normalization(
    w: Tensor,
    u: Tensor,
    num_iters: int = 1,
) -> tuple[Tensor, Tensor]:
    """Compute the spectral normalization.

    Args:
        w (Tensor): Weight tensor.
        u (Tensor): Left singular vector estimate.
        num_iters (int): Number of power iterations.

    Returns:
        tuple[Tensor, Tensor]: Normalized weight and new u.
    """
    _, u_new, sigma = power_iteration(w, num_iters=num_iters, u=u)
    return divide(w, sigma), u_new
