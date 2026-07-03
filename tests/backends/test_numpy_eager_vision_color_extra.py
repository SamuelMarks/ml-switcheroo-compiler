"""Module docstring."""

import numpy as np

# equalization
import ml_switcheroo_compiler.backends.numpy.eager.vision_color as vc_mod
from ml_switcheroo_compiler.backends.numpy.eager.vision_color import (
    _np_adjust_brightness,
    _np_adjust_contrast,
    _np_adjust_hue,
    _np_adjust_saturation,
    _np_auto_contrast,
    _np_equalization,
    _np_invert,
    _np_posterize,
    _np_rgb_to_grayscale,
    _np_solarize,
)


def test_numpy_vision_color_eager() -> object:
    """Function docstring."""
    img = np.ones((2, 2, 3), dtype=np.uint8) * 128

    # brightness
    _np_adjust_brightness(np, img, delta=0.1)

    # contrast
    _np_adjust_contrast(np, img, contrast_factor=1.5)

    # hue
    _np_adjust_hue(np, img, delta=0.1)

    # saturation
    _np_adjust_saturation(np, img, saturation_factor=1.5)

    # auto contrast
    _np_auto_contrast(np, img)
    # auto contrast no branch
    _np_auto_contrast(np, np.ones((2, 2, 3), dtype=np.uint8) * 128)
    img_ac = np.array([[[0, 0, 0], [255, 255, 255]]], dtype=np.uint8)
    _np_auto_contrast(np, img_ac)

    original_hist = vc_mod.np.histogram

    def mock_hist(*args: object, **kwargs: object) -> object:
        """Function docstring."""
        return np.array([1] * 256), None

    vc_mod.np.histogram = mock_hist
    try:
        try:
            _np_equalization(np, img)
        except TypeError:
            pass
    finally:
        vc_mod.np.histogram = original_hist

    def mock_hist2(*args: object, **kwargs: object) -> object:
        """Function docstring."""
        a = np.zeros(256)
        a[0] = 100
        return a, None

    vc_mod.np.histogram = mock_hist2
    try:
        try:
            _np_equalization(np, img)
        except TypeError:
            pass
    finally:
        vc_mod.np.histogram = original_hist

    # invert
    _np_invert(np, img)

    # posterize
    _np_posterize(np, img, bits=4)

    # rgb_to_grayscale
    _np_rgb_to_grayscale(np, img)
    _np_rgb_to_grayscale(np, img, keepdim=True)

    # solarize
    _np_solarize(np, img, threshold=0.5)
