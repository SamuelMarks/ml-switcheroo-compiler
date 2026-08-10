from __future__ import annotations

# ruff: noqa: E402, D100, D103, D104, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, D101, D102, D107, E701, E722, F403, E711, E712, PLR0913, PLR0915
from typing import Any

"""Vision operations."""


from ml_switcheroo_compiler.backends.registry import get_active_backend
from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.ops.shape.utils import _emit_shape_node


def mixup(images1: Tensor, images2: Tensor, alpha: float = 0.2, seed: int | None = None) -> Any:
    """Apply mixup to a pair of batches of images.

    Args:
        images1 (Tensor): First batch of input images.
        images2 (Tensor): Second batch of input images.
        alpha (float): Alpha parameter for Beta distribution.
        seed (int | None): Random seed.

    Returns:
        Tensor: Mixed up images.
    """
    if config.eager_mode:
        backend = get_active_backend()
        data = backend.execute_op("Mixup", images1.data, images2=images2.data, alpha=alpha, seed=seed)
        return Tensor(
            backend.array(data),
            TensorConfig(backend.array(data).shape, images1.dtype, images1.device),
        )
    return _emit_shape_node("Mixup", [images1, images2], {"alpha": alpha, "seed": seed}, (), images1.dtype)


def cutmix(images1: Tensor, images2: Tensor, alpha: float = 1.0, seed: int | None = None) -> Any:
    """Apply cutmix to a pair of batches of images.

    Args:
        images1 (Tensor): First batch of input images.
        images2 (Tensor): Second batch of input images.
        alpha (float): Alpha parameter for Beta distribution.
        seed (int | None): Random seed.

    Returns:
        Tensor: Cutmixed images.
    """
    if config.eager_mode:
        backend = get_active_backend()
        data = backend.execute_op("Cutmix", images1.data, images2=images2.data, alpha=alpha, seed=seed)
        return Tensor(
            backend.array(data),
            TensorConfig(backend.array(data).shape, images1.dtype, images1.device),
        )
    return _emit_shape_node("Cutmix", [images1, images2], {"alpha": alpha, "seed": seed}, (), images1.dtype)
