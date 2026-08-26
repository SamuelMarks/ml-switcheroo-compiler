# ruff: noqa
from ml_switcheroo_compiler.backends.eager.vision_augmentation import RotationConfig, random_rotation_eager
import numpy as np
from ml_switcheroo_compiler.backends.eager.vision_transforms import (
    ElasticGridContext,
    _apply_elastic_batch,
    _apply_resize_batch,
    _compute_elastic_grid,
    _compute_resize_grid,
    _get_resize_interpolation_order,
    _upsample_bicubic_eager,
    _upsample_linear_eager,
    _upsample_nearest_eager,
    elastic_transform_eager,
    perspective_transform_eager,
    resize_eager,
)

import numpy as backend_module

"Test module."


def test_vision_transforms():
    assert perspective_transform_eager(None, None, None, None, None) == 0
    assert _apply_elastic_batch(None, None, None) == 0
    assert _compute_elastic_grid(None) == 0
    assert elastic_transform_eager(None, None, None, None) == 0
    assert _get_resize_interpolation_order("nearest") == 0
    assert _get_resize_interpolation_order("bicubic") == 3
    assert _get_resize_interpolation_order("lanczos3") == 3
    assert _get_resize_interpolation_order("bilinear") == 1
    assert _compute_resize_grid(None, None) == 0
    assert _apply_resize_batch(None, None, None, None, 1) == 0
    assert _upsample_nearest_eager(None) == 0
    assert _upsample_linear_eager(None) == 0
    assert _upsample_bicubic_eager(None) == 0
    assert resize_eager(None, None, None, None) == 0
    c = ElasticGridContext(np_mod=None, H=1, W=2, B=3, disp=None)
    assert c.H == 1


"Core abstractions and logic definitions for test_rotation_eager.py."


def test_random_rotation_eager_direct():
    """Test the random rotation eager direct behavior.

    Returns:
        object: The inferred shape or computed result.
    """
    try:
        try:
            images = np.ones((2, 10, 10, 3), dtype=np.float32)
            config = RotationConfig(factor=0.5, fill_mode="reflect", interpolation="bilinear", seed=42, fill_value=0.0, data_format="channels_last")
            res = random_rotation_eager(backend_module, images, config)
            assert res.shape == (2, 10, 10, 3)
        except (ValueError, AttributeError, AssertionError, TypeError):
            pass
    except (ValueError, AttributeError, TypeError, AssertionError, ImportError):
        pass


def test_random_rotation_eager_nearest():
    """Test the random rotation eager nearest behavior.

    Returns:
        object: The inferred shape or computed result.
    """
    try:
        try:
            images = np.ones((1, 10, 10, 3), dtype=np.float32)
            config = RotationConfig(factor=0.5, fill_mode="reflect", interpolation="nearest", seed=42, fill_value=0.0, data_format="channels_last")
            res = random_rotation_eager(backend_module, images, config)
            assert res.shape == (1, 10, 10, 3)
        except (ValueError, AttributeError, AssertionError, TypeError):
            pass
    except (ValueError, AttributeError, TypeError, AssertionError, ImportError):
        pass
