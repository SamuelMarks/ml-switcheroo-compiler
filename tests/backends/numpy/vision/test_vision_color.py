"""Core abstractions and logic definitions for test_numpy_eager_vision_color_edge_cases.py."""

from __future__ import annotations

import typing

import numpy as np
import pytest

from ml_switcheroo_compiler.backends.eager.vision_ops import color as color_mod
from ml_switcheroo_compiler.backends.eager.vision_ops.color import (
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
from ml_switcheroo_compiler.backends.eager_registry import numpy_eager_registry
from tests.testing_utils import require_callable

if typing.TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import BackendArg


def test_numpy_vision_color_eager() -> None:
    """Test the numpy vision color eager behavior.

    Returns:
        OpValue: The inferred shape or computed result.
    """
    img = np.ones((2, 2, 3), dtype=np.uint8) * 128
    _np_adjust_brightness(np, img, delta=0.1)
    _np_adjust_contrast(np, img, contrast_factor=1.5)
    _np_adjust_hue(np, img, delta=0.1)
    _np_adjust_saturation(np, img, saturation_factor=1.5)
    _np_auto_contrast(np, img)
    _np_auto_contrast(np, np.ones((2, 2, 3), dtype=np.uint8) * 128)
    img_ac = np.array([[[0, 0, 0], [255, 255, 255]]], dtype=np.uint8)
    _np_auto_contrast(np, img_ac)
    original_hist = np.histogram

    def mock_hist(*args: BackendArg, **kwargs: BackendArg) -> BackendArg:
        """Evaluate and process the mock hist operation.

        Args:
            *args (BackendArg): Variable positional arguments.
            **kwargs (BackendArg): Arbitrary keyword arguments.

        Returns:
            BackendArg: The evaluated or processed output.
        """
        return (np.array([1] * 256), None)

    pytest.MonkeyPatch().setattr(np, "histogram", mock_hist, raising=False)
    try:
        _np_equalization(np, img)
    finally:
        pytest.MonkeyPatch().setattr(np, "histogram", original_hist, raising=False)

    def mock_hist2(*args: BackendArg, **kwargs: BackendArg) -> BackendArg:
        """Evaluate and process the mock hist2 operation.

        Args:
            *args (BackendArg): Variable positional arguments.
            **kwargs (BackendArg): Arbitrary keyword arguments.

        Returns:
            BackendArg: The evaluated or processed output.
        """
        a = np.zeros(256)
        a[0] = 100
        return (a, None)

    pytest.MonkeyPatch().setattr(np, "histogram", mock_hist2, raising=False)
    try:
        _np_equalization(np, img)
    finally:
        pytest.MonkeyPatch().setattr(np, "histogram", original_hist, raising=False)
    _np_invert(np, img)
    _np_posterize(np, img, bits=4)
    _np_rgb_to_grayscale(np, img)
    _np_rgb_to_grayscale(np, img, keepdim=True)
    _out = _np_solarize(np, img, threshold=0.5)
    assert np.asarray(_out).shape == (2, 2, 3)


def test_vision_color_ops() -> None:
    """Test all vision color transformations in vision_color.

    Returns:
        None
    """
    img = np.array([[[[0.2, 0.4, 0.6], [0.8, 0.5, 0.1]]]], dtype=np.float32)

    # AdjustBrightness
    b_adj = color_mod._np_adjust_brightness(np, img, 0.1)
    assert np.allclose(
        b_adj[
            (
                0,
                0,
                0,
            )
        ],
        [0.3, 0.5, 0.7],
    )
    assert require_callable(numpy_eager_registry.get)("AdjustBrightness")(np, img, 0.1) is not None

    # AdjustContrast
    c_adj = color_mod._np_adjust_contrast(np, img, 1.5)
    assert c_adj.shape == img.shape
    assert require_callable(numpy_eager_registry.get)("AdjustContrast")(np, img, 1.5) is not None

    # AdjustHue
    h_adj = color_mod._np_adjust_hue(np, img, 0.2)
    assert np.array_equal(h_adj, img)
    assert require_callable(numpy_eager_registry.get)("AdjustHue")(np, img, 0.2) is not None

    # AdjustSaturation
    s_adj = color_mod._np_adjust_saturation(np, img, 1.2)
    assert s_adj.shape == img.shape
    assert require_callable(numpy_eager_registry.get)("AdjustSaturation")(np, img, 1.2) is not None

    # AutoContrast: non-constant diff vs constant diff
    ac1 = color_mod._np_auto_contrast(np, img, value_range=(0, 255))
    assert ac1.shape == img.shape

    const_img = np.ones((1, 2, 2, 3), dtype=np.float32) * 0.5
    ac_const = color_mod._np_auto_contrast(np, const_img)
    assert ac_const.shape == const_img.shape
    assert require_callable(numpy_eager_registry.get)("AutoContrast")(np, img) is not None

    # Equalization: cdf_m diff == 0 vs diff != 0
    zero_img = np.zeros((1, 2, 2, 1), dtype=np.float32)
    eq_zero = color_mod._np_equalization(np, zero_img)
    assert np.array_equal(eq_zero, zero_img)

    grad_img = np.linspace(0.0, 1.0, 16, dtype=np.float32).reshape((1, 4, 4, 1))
    eq_grad = color_mod._np_equalization(np, grad_img)
    assert eq_grad.shape == grad_img.shape
    assert require_callable(numpy_eager_registry.get)("Equalization")(np, grad_img) is not None

    # Invert
    inv = color_mod._np_invert(np, img, value_range=(0, 255))
    assert inv.shape == img.shape
    assert require_callable(numpy_eager_registry.get)("Invert")(np, img) is not None

    # Posterize
    post = color_mod._np_posterize(np, img, bits=4)
    assert post.shape == img.shape
    assert require_callable(numpy_eager_registry.get)("Posterize")(np, img) is not None

    # RgbToGrayscale
    gray = color_mod._np_rgb_to_grayscale(np, img, data_format="channels_last")
    assert gray.shape == (1, 1, 2, 1)
    assert require_callable(numpy_eager_registry.get)("RgbToGrayscale")(np, img) is not None

    # Solarize
    sol = color_mod._np_solarize(np, img, threshold=0.5, value_range=(0, 1))
    assert sol.shape == img.shape
    assert require_callable(numpy_eager_registry.get)("Solarize")(np, img) is not None
