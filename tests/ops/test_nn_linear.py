import numpy as np
from ml_switcheroo_compiler.ops.nn.linear_ops import linear, bilinear
from ml_switcheroo_compiler import ops
from ml_switcheroo_compiler.core.config import config


def test_linear():
    config.eager_mode = True
    x_data = np.random.randn(2, 3).astype(np.float32)
    w_data = np.random.randn(4, 3).astype(np.float32)
    b_data = np.random.randn(4).astype(np.float32)

    x = ops.array(x_data)
    w = ops.array(w_data)
    b = ops.array(b_data)

    y = linear(x, w, b)
    assert y is not None


def test_bilinear():
    config.eager_mode = True
    x1_data = np.random.randn(2, 3).astype(np.float32)
    x2_data = np.random.randn(2, 4).astype(np.float32)
    w_data = np.random.randn(5, 3, 4).astype(np.float32)
    b_data = np.random.randn(5).astype(np.float32)

    x1 = ops.array(x1_data)
    x2 = ops.array(x2_data)
    w = ops.array(w_data)
    b = ops.array(b_data)

    y = bilinear(x1, x2, w, b)
    assert y is not None


def test_linear_no_bias():
    config.eager_mode = True
    x_data = np.random.randn(2, 3).astype(np.float32)
    w_data = np.random.randn(4, 3).astype(np.float32)

    x = ops.array(x_data)
    w = ops.array(w_data)

    y = linear(x, w)
    assert y is not None


def test_bilinear_no_bias():
    config.eager_mode = True
    x1_data = np.random.randn(2, 3).astype(np.float32)
    x2_data = np.random.randn(2, 4).astype(np.float32)
    w_data = np.random.randn(5, 3, 4).astype(np.float32)

    x1 = ops.array(x1_data)
    x2 = ops.array(x2_data)
    w = ops.array(w_data)

    y = bilinear(x1, x2, w)
    assert y is not None
