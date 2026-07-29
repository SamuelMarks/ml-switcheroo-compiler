import numpy as np

from ml_switcheroo_compiler.backends.numpy.eager.math_binary import _np_polygamma, _np_zeta


def test_polygamma_zeta_missing_args():
    a = np.array([1.0])
    res1 = _np_polygamma(np, a)
    assert np.all(res1 == 0.0)
    res2 = _np_zeta(np, a)
    assert np.all(res2 == 0.0)


def test_polygamma_zeta_kwargs():
    a = np.array([1.0])
    res1 = _np_polygamma(np, a, x=a)
    assert res1.shape == (1,)
    res2 = _np_zeta(np, a, q=a)
    assert res2.shape == (1,)
