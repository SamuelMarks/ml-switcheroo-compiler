# ruff: noqa: E501
import numpy as np

from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.tensor import Device, DType, Tensor, TensorConfig
from ml_switcheroo_compiler.ops.nn.loss import (
    _clip_and_convert_logits,
    _compute_bce_loss,
    _compute_circle_logits,
    _compute_circle_loss_reduction,
    _compute_circle_margins,
    adaptive_log_softmax_with_loss,
    binary_crossentropy,
    categorical_crossentropy,
    categorical_generalized_cross_entropy,
    circle_loss,
    ctc_decode,
    dice_loss,
    in_top_k,
    l2_loss,
    log_poisson_loss,
    scale_regularization_loss,
    sparse_categorical_crossentropy,
    tversky_loss,
)


def test_loss_coverage():
    config.eager_mode = True
    t_true = Tensor(np.array([[1, 0]]), TensorConfig(shape=(1, 2), dtype=DType("int32"), device=Device("cpu")))
    t_pred = Tensor(np.array([[0.9, 0.1]]), TensorConfig(shape=(1, 2), dtype=DType("float32"), device=Device("cpu")))

    assert dice_loss(t_true, t_pred) is not None
    assert categorical_generalized_cross_entropy(t_true, t_pred) is not None
    assert _compute_circle_margins(0.25) == (1.25, -0.25, 0.75, 0.25)

    lp, ln = _compute_circle_logits(t_pred, 0.25, 256.0)
    assert lp is not None and ln is not None

    assert _compute_circle_loss_reduction(lp, ln, t_true, Tensor(np.array([[0.0, 1.0]]), TensorConfig(shape=(1, 2), dtype=DType("float32"), device=Device("cpu")))) is not None
    assert circle_loss(t_true, t_pred) is not None

    assert tversky_loss(t_true, t_pred) is not None
    assert _clip_and_convert_logits(t_pred, False) is not None
    assert _clip_and_convert_logits(t_pred, True) is not None

    assert _compute_bce_loss(t_true, t_pred, True) is not None
    assert _compute_bce_loss(t_true, t_pred, False) is not None
    assert binary_crossentropy(t_true, t_pred) is not None
    assert categorical_crossentropy(t_true, t_pred) is not None
    assert sparse_categorical_crossentropy(t_true, t_pred) is not None
    assert l2_loss(t_true) is not None
    assert scale_regularization_loss(t_true) is not None
    assert log_poisson_loss(t_true, t_pred) is not None
    assert in_top_k(t_true, t_pred, 1) is not None
    assert ctc_decode(t_true, t_pred) is not None
    assert adaptive_log_softmax_with_loss(t_true, t_pred, [1]) is not None
