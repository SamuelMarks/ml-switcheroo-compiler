import numpy as np


def test_loss_ops_stubs():
    from ml_switcheroo_compiler.backends.numpy.eager.loss_ops import _np_categorical_generalized_cross_entropy, _np_circle_loss, _np_ctc_loss

    labels = np.array([[1, 2]])
    logits = np.ones((5, 1, 3))
    label_length = np.array([2])
    logit_length = np.array([5])

    assert _np_ctc_loss(np, labels, logits, label_length, logit_length).shape == (1,)

    logits2 = np.ones((5, 3))
    assert _np_ctc_loss(np, labels, logits2, label_length, logit_length).shape == (1,)

    y_true = np.array([1, 0])
    y_pred = np.array([0.9, 0.1])
    assert _np_circle_loss(np, y_true, y_pred).shape == ()

    y_true_cce = np.array([[1, 0], [0, 1]])
    y_pred_cce = np.array([[0.9, 0.1], [0.2, 0.8]])
    assert _np_categorical_generalized_cross_entropy(np, y_true_cce, y_pred_cce).shape == ()
