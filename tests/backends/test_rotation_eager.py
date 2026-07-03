"""Module docstring."""

import numpy as backend_module
import numpy as np

from ml_switcheroo_compiler.backends.eager.vision_augmentation import (
    RotationConfig,
    random_rotation_eager,
)


def test_random_rotation_eager_direct() -> object:
    """Function docstring."""
    images = np.ones((2, 10, 10, 3), dtype=np.float32)
    config = RotationConfig(
        factor=0.5,
        fill_mode="reflect",
        interpolation="bilinear",
        seed=42,
        fill_value=0.0,
        data_format="channels_last",
    )

    res = random_rotation_eager(backend_module, images, config)
    assert res.shape == (2, 10, 10, 3)


def test_random_rotation_eager_nearest() -> object:
    """Function docstring."""
    images = np.ones((1, 10, 10, 3), dtype=np.float32)
    config = RotationConfig(
        factor=0.5,
        fill_mode="reflect",
        interpolation="nearest",
        seed=42,
        fill_value=0.0,
        data_format="channels_last",
    )

    res = random_rotation_eager(backend_module, images, config)
    assert res.shape == (1, 10, 10, 3)
