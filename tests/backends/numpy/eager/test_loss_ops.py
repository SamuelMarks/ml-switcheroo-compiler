import numpy as np

from ml_switcheroo_compiler.backends.numpy.eager.loss_ops import (
    _np_categorical_generalized_cross_entropy,
    _np_circle_loss,
    _np_ctc_loss,
)


class DummyBackend:
    @staticmethod
    def zeros(shape):
        return np.zeros(shape)


def test_loss_ops():
    # CTC Loss
    labels = np.array([[1, 2]])
    logits = np.ones((5, 1, 3))
    label_length = np.array([2])
    logit_length = np.array([5])
    logit_length_3 = np.array([3])

    assert _np_ctc_loss(DummyBackend(), labels, logits, label_length, logit_length).shape == (1,)

    # 2D logits
    logits2 = np.ones((5, 3))
    assert _np_ctc_loss(DummyBackend(), labels, logits2, label_length, logit_length).shape == (1,)

    # 1D logits
    logits1 = np.ones((3,))
    assert _np_ctc_loss(DummyBackend(), labels, logits1, label_length, logit_length_3).shape == (1,)

    # Time major = False
    logits_batch_major = np.ones((1, 5, 3))
    assert _np_ctc_loss(DummyBackend(), labels, logits_batch_major, label_length, logit_length, logits_time_major=False).shape == (1,)

    # 1D labels
    labels_1d = np.array([1, 2])
    assert _np_ctc_loss(DummyBackend(), labels_1d, logits, label_length, logit_length).shape == (1,)

    # No labels
    assert _np_ctc_loss(DummyBackend(), np.array([[]], dtype=int), logits, np.array([0]), logit_length).shape == (1,)

    # Circle Loss
    y_true = np.array([1, 0])
    y_pred = np.array([0.9, 0.1])
    assert _np_circle_loss(DummyBackend(), y_true, y_pred).shape == ()
    assert _np_circle_loss(DummyBackend(), 1) == 0  # < 2 args returns 0

    # Categorical GCE
    y_true_cce = np.array([[1, 0], [0, 1]])
    y_pred_cce = np.array([[0.9, 0.1], [0.2, 0.8]])
    assert _np_categorical_generalized_cross_entropy(DummyBackend(), y_true_cce, y_pred_cce).shape == ()
    assert _np_categorical_generalized_cross_entropy(DummyBackend(), 1) == 0  # < 2 args returns 0


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
