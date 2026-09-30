"""Tests for test_nn_coverage."""

from __future__ import annotations

import numpy as np

from ml_switcheroo_compiler.backends.eager_registry import numpy_eager_registry
from ml_switcheroo_compiler.backends.numpy.eager import nn as nn_mod


def test_nn_ops() -> None:
    """Test neural network operations in nn.

    Returns:
        None
    """
    # Gelu
    x = np.array([-1.0, 0.0, 1.0], dtype=np.float32)
    gelu_res = nn_mod._gelu(x)
    assert gelu_res.shape == (3,)

    # Relu
    relu_res = nn_mod._np_relu(np, x)
    assert np.array_equal(relu_res, [0.0, 0.0, 1.0])
    assert np.array_equal(numpy_eager_registry.get("Relu")(np, x), [0.0, 0.0, 1.0])

    # AlphaDropout: not training, rate == 0, training with custom noise_shape and default noise_shape
    assert np.array_equal(nn_mod._np_alpha_dropout(np, x, training=False), x)
    assert np.array_equal(nn_mod._np_alpha_dropout(np, x, rate=0.0, training=True), x)
    ad_train1 = nn_mod._np_alpha_dropout(np, x, rate=0.2, training=True, seed=42)
    assert ad_train1.shape == (3,)
    ad_train2 = nn_mod._np_alpha_dropout(np, x, rate=0.2, training=True, seed=42, noise_shape=(3,))
    assert ad_train2.shape == (3,)
    assert numpy_eager_registry.get("AlphaDropout")(np, x, training=False) is not None

    # ActivityRegularization
    assert np.array_equal(nn_mod._np_activity_regularization(np, x), x)
    assert numpy_eager_registry.get("ActivityRegularization")(np, x) is not None

    # Dropout: not training, rate == 0, training with custom noise_shape and default noise_shape
    assert np.array_equal(nn_mod._np_dropout(np, x, training=False), x)
    assert np.array_equal(nn_mod._np_dropout(np, x, rate=0.0, training=True), x)
    do_train1 = nn_mod._np_dropout(np, x, rate=0.2, training=True, seed=42)
    assert do_train1.shape == (3,)
    do_train2 = nn_mod._np_dropout(np, x, rate=0.2, training=True, seed=42, noise_shape=(3,))
    assert do_train2.shape == (3,)
    assert numpy_eager_registry.get("Dropout")(np, x, training=False) is not None

    # TimeDistributed: len(shape) < 3 vs len(shape) >= 3
    x_2d = np.array([[-1.0, 2.0], [3.0, -4.0]])
    td_2d = nn_mod._np_time_distributed(np, x_2d, wrapped_op_name="Relu")
    assert np.array_equal(td_2d, [[0.0, 2.0], [3.0, 0.0]])

    x_3d = np.array([[[-1.0, 2.0]], [[3.0, -4.0]]])
    td_3d = nn_mod._np_time_distributed(np, x_3d, wrapped_op_name="Relu")
    assert np.array_equal(td_3d, [[[0.0, 2.0]], [[3.0, 0.0]]])
    assert numpy_eager_registry.get("TimeDistributed")(np, x_2d, wrapped_op_name="Relu") is not None

    # Rope
    rope_input = np.ones((2, 4, 8), dtype=np.float32)
    rope_res1 = nn_mod._np_rope(np, rope_input)
    assert rope_res1.shape == (2, 4, 8)
    rope_res2 = nn_mod._np_rope(np, rope_input, dim=8, offset=2, base=5000.0)
    assert rope_res2.shape == (2, 4, 8)
    assert numpy_eager_registry.get("Rope")(np, rope_input).shape == (2, 4, 8)

    # Rrelu: not training vs training
    rrelu_notrain = nn_mod._np_rrelu(np, x, training=False)
    assert rrelu_notrain.shape == (3,)
    rrelu_train = nn_mod._np_rrelu(np, x, training=True)
    assert rrelu_train.shape == (3,)
    assert numpy_eager_registry.get("Rrelu")(np, x, training=False).shape == (3,)
