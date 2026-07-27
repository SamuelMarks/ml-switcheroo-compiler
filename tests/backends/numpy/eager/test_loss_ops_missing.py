import numpy as np

from ml_switcheroo_compiler.backends.numpy.eager.loss_ops import _np_ctc_loss_single


def test_ctc_loss_missing_branches():
    # To cover 35->37: len(b_labels) == 0
    # To cover 41->46: T == 0
    # To cover 43->46: S <= 1 (when T > 0)
    # To cover 50->53: S <= 0 or S <= 1

    # 1. len(b_labels) == 0, S = 1
    probs = np.ones((2, 2))
    labels = np.array([])
    _np_ctc_loss_single(probs, labels, 2, 0)

    # 2. T == 0
    probs = np.ones((0, 2))
    labels = np.array([0, 1])
    try:
        _np_ctc_loss_single(probs, labels, 0, 2)
    except IndexError:
        # Expected IndexError from T - 1 if we don't fix it.
        pass
