import numpy as np

from ml_switcheroo_compiler.backends.numpy.eager.loss_ops import _np_ctc_loss_single


def test_ctc_loss_missing_branches():
    probs = np.ones((2, 2))
    labels = np.array([])
    res = _np_ctc_loss_single(probs, labels, 2, 0)
    assert isinstance(res, float) or isinstance(res, np.floating)

    probs = np.ones((0, 2))
    labels = np.array([0, 1])
    try:
        _np_ctc_loss_single(probs, labels, 0, 2)
    except IndexError:
        pass
    except ValueError:
        pass
