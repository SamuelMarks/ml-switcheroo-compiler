"""Behavioral unit tests for vision transforms."""

from __future__ import annotations

import numpy as np

from ml_switcheroo_compiler.backends.eager.vision_ops import transforms as vision_transforms
from ml_switcheroo_compiler.backends.eager_registry import numpy_eager_registry
from tests.testing_utils import require_callable


def test_vision_transforms() -> None:
    """Verify `RandomShear`, `RandomPerspective`, and `RandomElasticTransform` directly and via registry."""
    images = np.ones((1, 8, 8, 3), dtype=np.float32)

    res_shear_none = vision_transforms._np_random_shear(np, images, y_factor=0.1, x_factor=None)
    assert tuple(res_shear_none.shape) == (1, 8, 8, 3)

    res_shear_x = vision_transforms._np_random_shear(np, images, y_factor=0.1, x_factor=0.1)
    assert tuple(res_shear_x.shape) == (1, 8, 8, 3)

    reg_shear = numpy_eager_registry.get("RandomShear")
    res_reg_shear = require_callable(reg_shear)(np, images, y_factor=0.2)
    assert tuple(res_reg_shear.shape) == (1, 8, 8, 3)

    res_persp = vision_transforms._np_random_perspective(np, images, factor=0.1)
    assert tuple(res_persp.shape) == (2,)

    reg_persp = numpy_eager_registry.get("RandomPerspective")
    res_reg_persp = require_callable(reg_persp)(np, images, factor=0.2)
    assert tuple(res_reg_persp.shape) == (2,)

    res_elastic = vision_transforms._np_random_elastic_transform(np, images, alpha=1.0, sigma=0.5)
    assert tuple(res_elastic.shape) == (1, 8, 8, 3)

    reg_elastic = numpy_eager_registry.get("RandomElasticTransform")
    res_reg_elastic = require_callable(reg_elastic)(np, images, alpha=2.0, sigma=1.0)
    assert tuple(res_reg_elastic.shape) == (1, 8, 8, 3)
