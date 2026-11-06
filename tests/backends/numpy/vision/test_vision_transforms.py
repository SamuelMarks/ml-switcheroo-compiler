"""Tests for test_vision_transforms."""

from __future__ import annotations

import typing
from unittest.mock import patch

import numpy as np

import ml_switcheroo_compiler.backends.numpy.eager.vision_transforms as vt
from tests.testing_utils import ensure_eager_tensor, ensure_mutable

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
    ensure_mutable(mock_shear).return_value = ensure_eager_tensor(1)
    res = vt._np_random_shear(ensure_eager_tensor(np), np.zeros((2, 2, 3)), 1.0)
    assert res == 1

    ensure_mutable(mock_perspective).return_value = ensure_eager_tensor(2)
    res = vt._np_random_perspective(ensure_eager_tensor(np), np.zeros((2, 2, 3)), 1.0)
    assert res == 2

    ensure_mutable(mock_elastic).return_value = ensure_eager_tensor(3)
    res = vt._np_random_elastic_transform(ensure_eager_tensor(np), np.zeros((2, 2, 3)), 1.0, 1.0)
    assert res == 3
