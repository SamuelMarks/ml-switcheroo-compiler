"""Tests for test_vision_transforms_coverage."""

from __future__ import annotations

import numpy as np

from ml_switcheroo_compiler.backends.eager_registry import numpy_eager_registry
from ml_switcheroo_compiler.backends.numpy.eager import vision_transforms as vision_transforms


def test_vision_transforms() -> None:
    """Test vision augmentation transforms directly and through eager registry."""
    images = np.ones((1, 8, 8, 3), dtype=np.float32)

    res_shear_none = vision_transforms._np_random_shear(np, images, y_factor=0.1, x_factor=None)
    assert res_shear_none is not None

    res_shear_x = vision_transforms._np_random_shear(np, images, y_factor=0.1, x_factor=0.1)
    assert res_shear_x is not None

    reg_shear = numpy_eager_registry.get("RandomShear")
    res_reg_shear = reg_shear(np, images, y_factor=0.2)
    assert res_reg_shear is not None

    res_persp = vision_transforms._np_random_perspective(np, images, factor=0.1)
    assert res_persp is not None

    reg_persp = numpy_eager_registry.get("RandomPerspective")
    res_reg_persp = reg_persp(np, images, factor=0.2)
    assert res_reg_persp is not None

    res_elastic = vision_transforms._np_random_elastic_transform(np, images, alpha=1.0, sigma=0.5)
    assert res_elastic is not None

    reg_elastic = numpy_eager_registry.get("RandomElasticTransform")
    res_reg_elastic = reg_elastic(np, images, alpha=2.0, sigma=1.0)
    assert res_reg_elastic is not None
