"""Core abstractions and logic definitions for test_numpy_eager_vision_color_edge_cases.py."""

from __future__ import annotations

import typing

import numpy as np

import ml_switcheroo_compiler.backends.numpy.eager.vision_color as vc_mod
from ml_switcheroo_compiler.backends.numpy.eager.vision_color import _np_adjust_brightness, _np_adjust_contrast, _np_adjust_hue, _np_adjust_saturation, _np_auto_contrast, _np_equalization, _np_invert, _np_posterize, _np_rgb_to_grayscale, _np_solarize

if typing.TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import EagerTensorProtocol, EagerValue


def test_numpy_vision_color_eager() -> None:
    """Test the numpy vision color eager behavior.

    Returns:
        TensorOrScalar: The inferred shape or computed result.
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
    original_hist = typing.cast("EagerTensorProtocol", getattr(vc_mod, "np")).histogram

    def mock_hist(*args: EagerValue, **kwargs: EagerValue) -> EagerValue:
        """Evaluate and process the mock hist operation.

        Args:
            *args (EagerValue): Variable positional arguments.
            **kwargs (EagerValue): Arbitrary keyword arguments.

        Returns:
            EagerValue: The evaluated or processed output.
        """
        return (np.array([1] * 256), None)

    setattr(typing.cast("EagerTensorProtocol", getattr(vc_mod, "np")), "histogram", mock_hist)
    try:
        _np_equalization(np, img)
    finally:
        setattr(typing.cast("EagerTensorProtocol", getattr(vc_mod, "np")), "histogram", original_hist)

    def mock_hist2(*args: EagerValue, **kwargs: EagerValue) -> EagerValue:
        """Evaluate and process the mock hist2 operation.

        Args:
            *args (EagerValue): Variable positional arguments.
            **kwargs (EagerValue): Arbitrary keyword arguments.

        Returns:
            EagerValue: The evaluated or processed output.
        """
        a = np.zeros(256)
        a[0] = 100
        return (a, None)

    setattr(typing.cast("EagerTensorProtocol", getattr(vc_mod, "np")), "histogram", mock_hist2)
    try:
        _np_equalization(np, img)
    finally:
        setattr(typing.cast("EagerTensorProtocol", getattr(vc_mod, "np")), "histogram", original_hist)
    _np_invert(np, img)
    _np_posterize(np, img, bits=4)
    _np_rgb_to_grayscale(np, img)
    _np_rgb_to_grayscale(np, img, keepdim=True)
    _np_solarize(np, img, threshold=0.5)
    assert callable(test_numpy_vision_color_eager)
