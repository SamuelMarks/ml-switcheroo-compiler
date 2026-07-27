# ruff: noqa: E501
import pytest

from ml_switcheroo_compiler.core.errors import ShapeMismatchError
from ml_switcheroo_compiler.core.shape import _broadcast_dim, broadcast_shapes

"Test module."


def test_shape_coverage():
    assert _broadcast_dim(2, 2, (2,), (2,)) == 2
    assert _broadcast_dim(1, 2, (1,), (2,)) == 2
    assert _broadcast_dim(2, 1, (2,), (1,)) == 2
    with pytest.raises(ShapeMismatchError):
        _broadcast_dim(2, 3, (2,), (3,))
    assert broadcast_shapes((2, 2), (2, 2)) == (2, 2)
    assert broadcast_shapes((1, 2), (2, 2)) == (2, 2)
    assert broadcast_shapes((2, 1), (2, 2)) == (2, 2)
    assert broadcast_shapes((2,), (2, 2)) == (2, 2)
    with pytest.raises(ShapeMismatchError):
        broadcast_shapes((2, 3), (2, 4))
