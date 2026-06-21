"""Vision operations."""

from __future__ import annotations

from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.ops.shape.utils import _emit_shape_node


def mixup(images1: Tensor, images2: Tensor, alpha: float = 0.2, seed: int | None = None) -> Tensor:
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
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        backend = get_active_backend()
        data = backend.execute_op(
            "Mixup", images1.data, images2=images2.data, alpha=alpha, seed=seed
        )
        return Tensor(
            backend.array(data),
            TensorConfig(backend.array(data).shape, images1.dtype, images1.device),
        )
    return _emit_shape_node(
        "Mixup", [images1, images2], {"alpha": alpha, "seed": seed}, (), images1.dtype
    )


def cutmix(images1: Tensor, images2: Tensor, alpha: float = 1.0, seed: int | None = None) -> Tensor:
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
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        backend = get_active_backend()
        data = backend.execute_op(
            "Cutmix", images1.data, images2=images2.data, alpha=alpha, seed=seed
        )
        return Tensor(
            backend.array(data),
            TensorConfig(backend.array(data).shape, images1.dtype, images1.device),
        )
    return _emit_shape_node(
        "Cutmix", [images1, images2], {"alpha": alpha, "seed": seed}, (), images1.dtype
    )
