"""Test Numpy eager vision geometry coverage."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np

from ml_switcheroo_compiler.backends.numpy.eager.vision_geometry import _np_affine_grid, _np_elastic_transform

if TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import EagerCallableProtocol, EagerTensorProtocol


def test_elastic_transform() -> None:
    # 1D array fallback
    """Test function test_elastic_transform."""
    arr_1d = np.array([1, 2, 3])
    disp = np.array([[[1, 1]]])
    res_1d = _np_elastic_transform(None, arr_1d, disp)
    assert cast("EagerCallableProtocol", np.array_equal)(res_1d, arr_1d)


def test_affine_grid() -> None:
    # Test len(s) == 4 branch
    """Test function test_affine_grid."""
    theta = np.ones((1, 6))
    size_4d = (1, 3, 10, 10)
    res_4d = _np_affine_grid(None, theta, size_4d)
    assert cast("EagerTensorProtocol", getattr(res_4d, "shape")) == (1, 10, 10, 2)
