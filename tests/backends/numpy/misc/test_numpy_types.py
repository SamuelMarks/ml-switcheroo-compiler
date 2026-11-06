"""Test module."""

from __future__ import annotations

import typing

from ml_switcheroo_compiler.backends.numpy.types import array, asarray, item, zeros
from tests.testing_utils import ensure_callable, ensure_eager_tensor

if typing.TYPE_CHECKING:
    pass


def test_numpy_types() -> None:
    """Test function test_numpy_types."""
    assert ensure_eager_tensor(ensure_callable(zeros)(None, (1,))).shape == (1,)

    assert ensure_callable(ensure_eager_tensor(ensure_eager_tensor(ensure_callable(array)(None, [1])).tolist))() == [1]
    import numpy as np

    assert ensure_eager_tensor(ensure_eager_tensor(ensure_callable(array)(None, [1], dtype=np.int32)).dtype) == np.int32

    assert ensure_callable(ensure_eager_tensor(ensure_eager_tensor(ensure_callable(asarray)(None, [1])).tolist))() == [1]

    assert ensure_callable(item)(None, [42]) == 42.0
