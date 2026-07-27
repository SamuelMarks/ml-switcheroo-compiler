# ruff: noqa: E501
import numpy as np

from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.ops.loss import (
    categorical_generalized_cross_entropy,
    circle_loss,
    cosine_similarity_loss,
    ctc_loss,
    gaussian_nll_loss,
    hinge_loss,
    huber_loss,
    kl_div_loss,
    l1_loss,
    log_cosh_loss,
    mape_loss,
    margin_ranking_loss,
    mse_loss,
    msle_loss,
    nll_loss,
    smooth_l1_loss,
    triplet_loss,
)


def test_losses_eager() -> None:
    config.eager_mode = True
    t = Tensor(np.array([[1.0, 2.0]], dtype=np.float32), TensorConfig((1, 2), "float32", "cpu"))
    t2 = Tensor(np.array([[1.5, 2.5]], dtype=np.float32), TensorConfig((1, 2), "float32", "cpu"))
    assert isinstance(l1_loss(t, t2), Tensor)
    assert isinstance(mse_loss(t, t2), Tensor)
    assert isinstance(huber_loss(t, t2), Tensor)
    assert isinstance(smooth_l1_loss(t, t2), Tensor)
    assert isinstance(smooth_l1_loss(t, t2, beta=0.0), Tensor)
    assert isinstance(cosine_similarity_loss(t, t2), Tensor)
    assert isinstance(kl_div_loss(t, t2), Tensor)
    assert isinstance(hinge_loss(t, t2), Tensor)
    var = Tensor(np.array([[1.0, 1.0]], dtype=np.float32), TensorConfig((1, 2), "float32", "cpu"))
    assert isinstance(gaussian_nll_loss(t, t2, var), Tensor)
    assert isinstance(log_cosh_loss(t, t2), Tensor)
    target = Tensor(np.array([[1.0]], dtype=np.float32), TensorConfig((1, 1), "float32", "cpu"))
    assert isinstance(margin_ranking_loss(t, t2, target), Tensor)
    y_true_indices = Tensor(np.array([1], dtype=np.int32), TensorConfig((1,), "int32", "cpu"))
    assert isinstance(nll_loss(t, y_true_indices), Tensor)
    anc = Tensor(np.array([[1.0, 2.0]], dtype=np.float32), TensorConfig((1, 2), "float32", "cpu"))
    pos = Tensor(np.array([[1.1, 2.1]], dtype=np.float32), TensorConfig((1, 2), "float32", "cpu"))
    neg = Tensor(np.array([[0.0, 0.0]], dtype=np.float32), TensorConfig((1, 2), "float32", "cpu"))
    assert isinstance(triplet_loss(anc, pos, neg), Tensor)
    assert isinstance(triplet_loss(anc, pos, neg, p=1.0), Tensor)
    assert isinstance(msle_loss(t, t2), Tensor)
    assert isinstance(mape_loss(t, t2), Tensor)


def test_dummy_losses() -> None:
    config.eager_mode = False
    t = Tensor(np.array([[1.0, 2.0]], dtype=np.float32), TensorConfig((1, 2), "float32", "cpu"))
    try:
        ctc_loss(t, t)
    except Exception:
        pass
    try:
        circle_loss(t, t)
    except Exception:
        pass
    try:
        categorical_generalized_cross_entropy(t, t)
    except Exception:
        pass
