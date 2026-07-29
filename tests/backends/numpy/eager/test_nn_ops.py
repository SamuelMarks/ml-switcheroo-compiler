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
