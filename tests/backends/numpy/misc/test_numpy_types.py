"""Test module."""

from __future__ import annotations

import typing

from ml_switcheroo_compiler.backends.numpy.types import array, asarray, item, zeros
from tests.testing_utils import as_callable, as_tensor

if typing.TYPE_CHECKING:
    pass


def test_numpy_types() -> None:
    """Test function test_numpy_types."""
    assert as_tensor(getattr(as_callable(zeros)(None, (1,)), "shape")) == (1,)

    assert as_callable(as_tensor(getattr(as_callable(array)(None, [1]), "tolist")))() == [1]
    import numpy as np

    assert as_tensor(getattr(as_callable(array)(None, [1], dtype=np.int32), "dtype")) == np.int32

    assert as_callable(as_tensor(getattr(as_callable(asarray)(None, [1]), "tolist")))() == [1]

    assert as_callable(item)(None, [42]) == 42.0
