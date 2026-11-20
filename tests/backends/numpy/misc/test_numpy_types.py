"""Test module."""

from __future__ import annotations

import typing

from ml_switcheroo_compiler.backends.numpy.eager import ensure_ndarray
from ml_switcheroo_compiler.backends.numpy.types import array, asarray, item, zeros
from tests.testing_utils import extract_test_dtype, extract_test_shape

if typing.TYPE_CHECKING:
    pass


def test_numpy_types() -> None:
    """Test function test_numpy_types."""
    assert extract_test_shape(zeros(None, (1,))) == (1,)

    assert ensure_ndarray(array(None, [1])).tolist() == [1]
    import numpy as np

    assert extract_test_dtype(array(None, [1], dtype=np.int32)) == np.int32

    assert ensure_ndarray(asarray(None, [1])).tolist() == [1]

    assert item(None, [42]) == 42.0
