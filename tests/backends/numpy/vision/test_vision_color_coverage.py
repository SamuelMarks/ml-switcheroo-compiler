"""Tests for test_vision_color_coverage."""

from __future__ import annotations

import typing

import numpy as np

from ml_switcheroo_compiler.backends.eager_registry import numpy_eager_registry
from ml_switcheroo_compiler.backends.numpy.eager import vision_color as color_mod
from tests.testing_utils import ensure_callable, ensure_eager_tensor

if typing.TYPE_CHECKING:
    pass


def test_vision_color_ops() -> None:
    """Test all vision color transformations in vision_color.

    Returns:
        None
    """
    img = np.array([[[[0.2, 0.4, 0.6], [0.8, 0.5, 0.1]]]], dtype=np.float32)

    # AdjustBrightness
    b_adj = color_mod._np_adjust_brightness(ensure_eager_tensor(np), img, 0.1)
    assert ensure_callable(np.allclose)(
        ensure_eager_tensor(b_adj)[
            (
                0,
                0,
                0,
            )
        ],
        [0.3, 0.5, 0.7],
    )
    assert ensure_callable(numpy_eager_registry.get)("AdjustBrightness")(np, img, 0.1) is not None

    # AdjustContrast
    c_adj = color_mod._np_adjust_contrast(ensure_eager_tensor(np), img, 1.5)
    assert ensure_eager_tensor(c_adj).shape == ensure_eager_tensor(img).shape
    assert ensure_callable(numpy_eager_registry.get)("AdjustContrast")(np, img, 1.5) is not None

    # AdjustHue
    h_adj = color_mod._np_adjust_hue(ensure_eager_tensor(np), img, 0.2)
    assert ensure_callable(np.array_equal)(h_adj, img)
    assert ensure_callable(numpy_eager_registry.get)("AdjustHue")(np, img, 0.2) is not None

    # AdjustSaturation
    s_adj = color_mod._np_adjust_saturation(ensure_eager_tensor(np), img, 1.2)
    assert ensure_eager_tensor(s_adj).shape == ensure_eager_tensor(img).shape
    assert ensure_callable(numpy_eager_registry.get)("AdjustSaturation")(np, img, 1.2) is not None

    # AutoContrast: non-constant diff vs constant diff
    ac1 = color_mod._np_auto_contrast(ensure_eager_tensor(np), img, value_range=(0, 255))
    assert ensure_eager_tensor(ac1).shape == ensure_eager_tensor(img).shape

    const_img = np.ones((1, 2, 2, 3), dtype=np.float32) * 0.5
    ac_const = color_mod._np_auto_contrast(ensure_eager_tensor(np), const_img)
    assert ensure_eager_tensor(ac_const).shape == ensure_eager_tensor(const_img).shape
    assert ensure_callable(numpy_eager_registry.get)("AutoContrast")(np, img) is not None

    # Equalization: cdf_m diff == 0 vs diff != 0
    zero_img = np.zeros((1, 2, 2, 1), dtype=np.float32)
    eq_zero = color_mod._np_equalization(ensure_eager_tensor(np), zero_img)
    assert ensure_callable(np.array_equal)(eq_zero, zero_img)

    grad_img = np.linspace(0.0, 1.0, 16, dtype=np.float32).reshape((1, 4, 4, 1))
    eq_grad = color_mod._np_equalization(ensure_eager_tensor(np), grad_img)
    assert ensure_eager_tensor(eq_grad).shape == ensure_eager_tensor(grad_img).shape
    assert ensure_callable(numpy_eager_registry.get)("Equalization")(np, grad_img) is not None

    # Invert
    inv = color_mod._np_invert(ensure_eager_tensor(np), img, value_range=(0, 255))
    assert ensure_eager_tensor(inv).shape == ensure_eager_tensor(img).shape
    assert ensure_callable(numpy_eager_registry.get)("Invert")(np, img) is not None

    # Posterize
    post = color_mod._np_posterize(ensure_eager_tensor(np), img, bits=4)
    assert ensure_eager_tensor(post).shape == ensure_eager_tensor(img).shape
    assert ensure_callable(numpy_eager_registry.get)("Posterize")(np, img) is not None

    # RgbToGrayscale
    gray = color_mod._np_rgb_to_grayscale(ensure_eager_tensor(np), img, data_format="channels_last")
    assert ensure_eager_tensor(gray).shape == (1, 1, 2, 1)
    assert ensure_callable(numpy_eager_registry.get)("RgbToGrayscale")(np, img) is not None

    # Solarize
    sol = color_mod._np_solarize(ensure_eager_tensor(np), img, threshold=0.5, value_range=(0, 1))
    assert ensure_eager_tensor(sol).shape == ensure_eager_tensor(img).shape
    assert ensure_callable(numpy_eager_registry.get)("Solarize")(np, img) is not None
