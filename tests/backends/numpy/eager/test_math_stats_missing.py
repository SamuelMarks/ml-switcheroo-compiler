import numpy as np

from ml_switcheroo_compiler.backends.numpy.eager.math_stats import _np_randomcategorical, _np_randomshuffle


def test_np_random_shuffle_list():
    x = [1, 2, 3]
    res = _np_randomshuffle(None, x)
    assert isinstance(res, np.ndarray)
    assert set(res) == {1, 2, 3}


def test_randomcategorical_shape():
    # provide logits as kwargs and shape
    res = _np_randomcategorical(None, None, logits=np.array([1.0, 2.0]), shape=(1, 2))
    assert res.shape == (1, 2)

    # test missing shape
    res = _np_randomcategorical(None, None, logits=np.array([[1.0, 2.0]]))
    assert res.shape == (1,)
