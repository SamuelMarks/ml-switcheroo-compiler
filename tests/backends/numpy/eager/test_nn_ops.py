import numpy as np

import ml_switcheroo_compiler.backends.numpy.eager.nn_ops as nn_ops


def test_missing_nn_ops():

    # dropout2d logic
    x_4d = np.ones((1, 2, 3, 3))

    # Check that it drops elements or scales them
    res_drop = nn_ops._np_dropout2d(np, x_4d, p=0.5, training=True)
    assert res_drop.shape == x_4d.shape

    res_no_drop = nn_ops._np_dropout2d(np, x_4d, p=0.0, training=True)
    np.testing.assert_array_equal(res_no_drop, x_4d)

    res_test = nn_ops._np_dropout2d(np, x_4d, p=0.5, training=False)
    np.testing.assert_array_equal(res_test, x_4d)


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
