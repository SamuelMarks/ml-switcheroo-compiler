"""Test Numpy eager vision geometry coverage."""

from __future__ import annotations

import typing

import numpy as np

from ml_switcheroo_compiler.backends.numpy.eager.vision_geometry import _np_affine_grid, _np_elastic_transform
from tests.testing_utils import ensure_callable, ensure_eager_tensor, ensure_mod

if typing.TYPE_CHECKING:
    pass


def test_elastic_transform() -> None:
    # 1D array fallback
    """Test function test_elastic_transform."""
    arr_1d = np.array([1, 2, 3])
    disp = np.array([[[1, 1]]])
    res_1d = _np_elastic_transform(ensure_mod(None), arr_1d, disp)
    assert ensure_callable(np.array_equal)(res_1d, arr_1d)


def test_affine_grid() -> None:
    # Test len(s) == 4 branch
    """Test function test_affine_grid."""
    theta = np.ones((1, 6))
    size_4d = (1, 3, 10, 10)
    res_4d = _np_affine_grid(ensure_mod(None), theta, size_4d)
    assert ensure_eager_tensor(res_4d).shape == (1, 10, 10, 2)
