import numpy as np
from ml_switcheroo_compiler.backends.eager.vision_augmentation import (
    random_rotation_eager,
    RotationConfig,
)


def test_random_rotation_eager_direct():
    images = np.ones((2, 10, 10, 3), dtype=np.float32)
    config = RotationConfig(
        factor=0.5,
        fill_mode="reflect",
        interpolation="bilinear",
        seed=42,
        fill_value=0.0,
        data_format="channels_last",
    )
    import numpy as backend_module

    res = random_rotation_eager(backend_module, images, config)
    assert res.shape == (2, 10, 10, 3)


def test_random_rotation_eager_nearest():
    images = np.ones((1, 10, 10, 3), dtype=np.float32)
    config = RotationConfig(
        factor=0.5,
        fill_mode="reflect",
        interpolation="nearest",
        seed=42,
        fill_value=0.0,
        data_format="channels_last",
    )
    import numpy as backend_module

    res = random_rotation_eager(backend_module, images, config)
    assert res.shape == (1, 10, 10, 3)
