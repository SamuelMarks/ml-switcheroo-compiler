import ml_switcheroo_compiler.ops.loss as loss_mod
from ml_switcheroo_compiler.core.config import ConfigContext
from ml_switcheroo_compiler import ops
import numpy as np


def test_loss_import():
    assert loss_mod is not None


def test_loss_functions():
    with ConfigContext(eager_mode=True):
        y_true = ops.array(np.array([1.0, 0.0, 1.0]))
        y_pred = ops.array(np.array([0.9, 0.1, 0.8]))
        y_pred_2d = ops.array(np.array([[0.9, 0.1], [0.1, 0.9], [0.8, 0.2]]))
        y_true_int = ops.array(np.array([1, 0, 1], dtype=np.int32))
        var = ops.array(np.array([0.1, 0.1, 0.1]))

        loss_mod.l1_loss(y_true, y_pred)
        loss_mod.mse_loss(y_true, y_pred)
        loss_mod.huber_loss(y_true, y_pred)
        loss_mod.smooth_l1_loss(y_true, y_pred)
        loss_mod.smooth_l1_loss(y_true, y_pred, beta=0.0)
        loss_mod.cosine_similarity_loss(y_true, y_pred)
        loss_mod.kl_div_loss(y_true, y_pred)
        loss_mod.hinge_loss(y_true, y_pred)
        loss_mod.gaussian_nll_loss(y_pred, y_true, var)
        loss_mod.log_cosh_loss(y_true, y_pred)
        loss_mod.margin_ranking_loss(y_true, y_pred, y_true)
        loss_mod.nll_loss(y_pred_2d, y_true_int)
        loss_mod.triplet_loss(y_true, y_pred, y_pred)

        loss_mod.triplet_loss(y_true, y_pred, y_pred, p=1)
