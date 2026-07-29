import numpy as np
import pytest

from ml_switcheroo_compiler.backends.numpy.eager.nn_ops import _np_block_masked_mm, _np_dropout2d


def test_dropout2d_non_4d():
    x = np.ones((2, 2, 2))
    with pytest.raises(ValueError, match="Dropout2d requires a 4D tensor"):
        _np_dropout2d(None, x, p=0.5, training=True)


def test_block_masked_mm():
    a = np.ones((2, 3))
    b = np.ones((3, 2))
    res = _np_block_masked_mm(None, a, b)
    assert res.shape == (2, 2)
    assert np.allclose(res, np.matmul(a, b))
