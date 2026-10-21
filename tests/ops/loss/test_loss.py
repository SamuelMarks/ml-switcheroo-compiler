"""Tests for loss operations in `ml_switcheroo_compiler.ops.loss`."""

from __future__ import annotations

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
from tests.testing_utils import as_tensor_obj


def test_losses_eager() -> None:
    """Verify numerical outputs of eager loss functions."""
    config.eager_mode = True
    t = Tensor(np.array([[1.0, 2.0]], dtype=np.float32), TensorConfig((1, 2), "float32", "cpu"))
    t2 = Tensor(np.array([[1.5, 2.5]], dtype=np.float32), TensorConfig((1, 2), "float32", "cpu"))

    res_l1 = as_tensor_obj(l1_loss(t, t2))
    np.testing.assert_allclose(np.asarray(res_l1.data), 0.5, rtol=1e-5)

    res_mse = as_tensor_obj(mse_loss(t, t2))
    np.testing.assert_allclose(np.asarray(res_mse.data), 0.25, rtol=1e-5)

    res_huber = as_tensor_obj(huber_loss(t, t2))
    np.testing.assert_allclose(np.asarray(res_huber.data), 0.125, rtol=1e-5)

    res_sl1 = as_tensor_obj(smooth_l1_loss(t, t2))
    np.testing.assert_allclose(np.asarray(res_sl1.data), 0.125, rtol=1e-5)

    res_sl1_zero = as_tensor_obj(smooth_l1_loss(t, t2, beta=0.0))
    np.testing.assert_allclose(np.asarray(res_sl1_zero.data), 0.5, rtol=1e-5)

    res_cos = as_tensor_obj(cosine_similarity_loss(t, t2))
    assert float(np.asarray(res_cos.data)) >= 0.0

    res_kl = as_tensor_obj(kl_div_loss(t, t2))
    assert isinstance(res_kl, Tensor)
    assert np.isfinite(np.asarray(res_kl.data)).all()

    res_hinge = as_tensor_obj(hinge_loss(t, t2))
    assert isinstance(res_hinge, Tensor)
    assert np.isfinite(np.asarray(res_hinge.data)).all()

    var = Tensor(np.array([[1.0, 1.0]], dtype=np.float32), TensorConfig((1, 2), "float32", "cpu"))
    res_gnll = as_tensor_obj(gaussian_nll_loss(t, t2, var))
    np.testing.assert_allclose(np.asarray(res_gnll.data), 0.125, rtol=1e-5)

    res_log_cosh = as_tensor_obj(log_cosh_loss(t, t2))
    np.testing.assert_allclose(np.asarray(res_log_cosh.data), float(np.log(np.cosh(-0.5))), rtol=1e-5)

    target = Tensor(np.array([[1.0]], dtype=np.float32), TensorConfig((1, 1), "float32", "cpu"))
    res_mrl = as_tensor_obj(margin_ranking_loss(t, t2, target))
    assert float(np.asarray(res_mrl.data)) > 0.0

    y_true_indices = Tensor(np.array([1], dtype=np.int32), TensorConfig((1,), "int32", "cpu"))
    res_nll = as_tensor_obj(nll_loss(t, y_true_indices))
    np.testing.assert_allclose(np.asarray(res_nll.data), -2.0, rtol=1e-5)

    anc = Tensor(np.array([[1.0, 2.0]], dtype=np.float32), TensorConfig((1, 2), "float32", "cpu"))
    pos = Tensor(np.array([[1.1, 2.1]], dtype=np.float32), TensorConfig((1, 2), "float32", "cpu"))
    neg = Tensor(np.array([[0.0, 0.0]], dtype=np.float32), TensorConfig((1, 2), "float32", "cpu"))
    res_trip2 = as_tensor_obj(triplet_loss(anc, pos, neg))
    assert float(np.asarray(res_trip2.data)) >= 0.0
    res_trip1 = as_tensor_obj(triplet_loss(anc, pos, neg, p=1.0))
    assert float(np.asarray(res_trip1.data)) >= 0.0

    res_msle = as_tensor_obj(msle_loss(t, t2))
    assert float(np.asarray(res_msle.data)) > 0.0
    res_mape = as_tensor_obj(mape_loss(t, t2))
    assert float(np.asarray(res_mape.data)) > 0.0


def test_specialized_losses_eager() -> None:
    """Verify eager evaluation of ctc_loss, circle_loss, and categorical_generalized_cross_entropy."""
    config.eager_mode = True
    y_true = Tensor(np.array([[1.0, 0.0]], dtype=np.float32), TensorConfig((1, 2), "float32", "cpu"))
    y_pred = Tensor(np.array([[0.8, 0.2]], dtype=np.float32), TensorConfig((1, 2), "float32", "cpu"))

    res_circle = as_tensor_obj(circle_loss(y_true, y_pred))
    assert isinstance(res_circle, Tensor)
    assert np.isfinite(np.asarray(res_circle.data)).all()

    res_gce = as_tensor_obj(categorical_generalized_cross_entropy(y_true, y_pred))
    assert isinstance(res_gce, Tensor)
    assert np.isfinite(np.asarray(res_gce.data)).all()

    logits = Tensor(np.zeros((2, 1, 3), dtype=np.float32), TensorConfig((2, 1, 3), "float32", "cpu"))
    labels = Tensor(np.array([[1]], dtype=np.int32), TensorConfig((1, 1), "int32", "cpu"))
    label_len = Tensor(np.array([1], dtype=np.int32), TensorConfig((1,), "int32", "cpu"))
    logit_len = Tensor(np.array([2], dtype=np.int32), TensorConfig((1,), "int32", "cpu"))
    res_ctc = ctc_loss(labels, logits, label_len, logit_len)
    assert isinstance(res_ctc, Tensor)
