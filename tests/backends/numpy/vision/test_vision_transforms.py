"""Tests for test_vision_transforms."""

from __future__ import annotations

import typing
from unittest.mock import patch

import numpy as np

import ml_switcheroo_compiler.backends.numpy.eager.vision_transforms as vt
from ml_switcheroo_compiler.core.shape import _set_attr

if typing.TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import EagerTensorProtocol


@patch("ml_switcheroo_compiler.backends.numpy.eager.vision_transforms.random_shear_eager")
@patch("ml_switcheroo_compiler.backends.numpy.eager.vision_transforms.random_perspective_eager")
@patch("ml_switcheroo_compiler.backends.numpy.eager.vision_transforms.random_elastic_transform_eager")
def test_vision_transforms_coverage(mock_elastic: EagerTensorProtocol, mock_perspective: EagerTensorProtocol, mock_shear: EagerTensorProtocol) -> None:
    """Test function test_vision_transforms_coverage.

    Args:
        mock_elastic: Test parameter `mock_elastic`.
        mock_perspective: Test parameter `mock_perspective`.
        mock_shear: Test parameter `mock_shear`.
    """
    _set_attr(mock_shear, "return_value", 1)
    res = vt._np_random_shear(np, np.zeros((2, 2, 3)), 1.0)
    assert res == 1

    _set_attr(mock_perspective, "return_value", 2)
    res = vt._np_random_perspective(np, np.zeros((2, 2, 3)), 1.0)
    assert res == 2

    _set_attr(mock_elastic, "return_value", 3)
    res = vt._np_random_elastic_transform(np, np.zeros((2, 2, 3)), 1.0, 1.0)
    assert res == 3
