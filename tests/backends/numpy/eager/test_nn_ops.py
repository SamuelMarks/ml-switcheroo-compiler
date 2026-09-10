"""Tests for numpy eager NN operations."""

import numpy as np
import pytest

import ml_switcheroo_compiler.backends.numpy.eager.nn_ops as nn_ops
from ml_switcheroo_compiler.backends.numpy.eager.nn_ops import (
    _np_batch_norm,
    _np_batch_norm_input_grad,
    _np_block_masked_mm,
    _np_dropout2d,
)


def test_missing_nn_ops() -> None:
    """Test dropout2d behaviour with dropping and eval modes."""
    x_4d = np.ones((1, 2, 3, 3))

    res_drop = nn_ops._np_dropout2d(np, x_4d, p=0.5, training=True)
    assert res_drop.shape == x_4d.shape

    res_no_drop = nn_ops._np_dropout2d(np, x_4d, p=0.0, training=True)
    np.testing.assert_array_equal(res_no_drop, x_4d)

    res_test = nn_ops._np_dropout2d(np, x_4d, p=0.5, training=False)
    np.testing.assert_array_equal(res_test, x_4d)


def test_dropout2d_non_4d() -> None:
    """Test dropout2d requires a 4D tensor."""
    x = np.ones((2, 2, 2))
    with pytest.raises(ValueError, match="Dropout2d requires a 4D tensor"):
        _np_dropout2d(None, x, p=0.5, training=True)


def test_block_masked_mm() -> None:
    """Test block masked matrix multiplication."""
    a = np.ones((2, 3))
    b = np.ones((3, 2))
    res = _np_block_masked_mm(None, a, b)
    assert res.shape == (2, 2)
    assert np.allclose(res, np.matmul(a, b))

    # Test _np_mul registered helper
    res_mul = nn_ops._np_mul(np, a, a)
    assert np.allclose(res_mul, a * a)


def test_batch_norm_1d() -> None:
    """Test 1D input branch for _np_batch_norm and _np_batch_norm_input_grad."""
    x_1d = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    gamma_1d = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    beta_1d = np.array([0.0, 0.0, 0.0], dtype=np.float32)

    out = _np_batch_norm(np, x_1d, gamma_1d, beta_1d)
    assert out.shape == (3,)
    assert not np.isnan(out).any()

    d_out = np.ones_like(x_1d)
    grad_in = _np_batch_norm_input_grad(np, d_out, x_1d, gamma_1d)
    assert grad_in.shape == (3,)
    assert not np.isnan(grad_in).any()


def test_max_pool2d_with_argmax_forward() -> None:
    """Test _np_max_pool2d_with_argmax_forward with various kernel and stride configs."""
    x = np.arange(32, dtype=np.float32).reshape(1, 2, 4, 4)

    # Int kernel and stride
    out1 = nn_ops._np_max_pool2d_with_argmax_forward(np, x, kernel_size=2, stride=2)
    assert out1.shape == (1, 2, 2, 2)

    # Tuple pool_size and strides
    out2 = nn_ops._np_max_pool2d_with_argmax_forward(np, x, pool_size=(2, 2), strides=(2, 2))
    assert out2.shape == (1, 2, 2, 2)
    assert np.allclose(out1, out2)

    # Default stride (derived from kernel_size tuple)
    out3 = nn_ops._np_max_pool2d_with_argmax_forward(np, x, kernel_size=(2, 2))
    assert out3.shape == (1, 2, 2, 2)


def test_batch_norm_forward_and_grads() -> None:
    """Test BatchNorm forward and gradient operations on 4D and 1D inputs."""
    # 4D case
    x_4d = np.random.randn(2, 3, 4, 4).astype(np.float32)
    gamma = np.ones(3, dtype=np.float32)
    beta = np.zeros(3, dtype=np.float32)
    d_out_4d = np.ones_like(x_4d)

    # Forward with explicit gamma, beta, eps
    out_4d = nn_ops._np_batch_norm(np, x_4d, gamma, beta, eps=1e-5)
    assert out_4d.shape == x_4d.shape

    # Forward with defaults (None gamma, None beta, epsilon kwarg)
    out_defaults = nn_ops._np_batch_norm(np, x_4d, None, None, epsilon=1e-4)
    assert out_defaults.shape == x_4d.shape

    # Input grad 4D with gamma and without gamma
    d_x_4d = nn_ops._np_batch_norm_input_grad(np, d_out_4d, x_4d, gamma, eps=1e-5)
    assert d_x_4d.shape == x_4d.shape
    d_x_no_g = nn_ops._np_batch_norm_input_grad(np, d_out_4d, x_4d, None, epsilon=1e-5)
    assert d_x_no_g.shape == x_4d.shape

    # Gamma grad 4D and 1D
    d_gamma_4d = nn_ops._np_batch_norm_gamma_grad(np, d_out_4d, x_4d, eps=1e-5)
    assert d_gamma_4d.shape == (3,)
    x_1d = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    d_gamma_1d = nn_ops._np_batch_norm_gamma_grad(np, np.ones_like(x_1d), x_1d, epsilon=1e-5)
    assert d_gamma_1d.shape == (1,)

    # Beta grad 4D and 1D
    d_beta_4d = nn_ops._np_batch_norm_beta_grad(np, d_out_4d)
    assert d_beta_4d.shape == (3,)
    d_beta_1d = nn_ops._np_batch_norm_beta_grad(np, np.ones_like(x_1d))
    assert d_beta_1d.shape == (1,)


def test_layer_norm_forward_and_grads() -> None:
    """Test LayerNorm forward and gradient operations."""
    x = np.random.randn(2, 3, 4).astype(np.float32)
    gamma = np.ones(4, dtype=np.float32)
    beta = np.zeros(4, dtype=np.float32)
    d_out = np.ones_like(x)

    # Forward with explicit params and axis=-1
    out = nn_ops._np_layer_norm(np, x, gamma, beta, eps=1e-5, axis=-1)
    assert out.shape == x.shape

    # Forward with defaults and positive axis
    out_axis1 = nn_ops._np_layer_norm(np, x, None, None, epsilon=1e-5, axis=1)
    assert out_axis1.shape == x.shape

    # Input grad with gamma and without gamma, positive axis
    dx = nn_ops._np_layer_norm_input_grad(np, d_out, x, gamma, eps=1e-5, axis=-1)
    assert dx.shape == x.shape
    dx_axis1 = nn_ops._np_layer_norm_input_grad(np, d_out, x, None, epsilon=1e-5, axis=1)
    assert dx_axis1.shape == x.shape

    # Gamma grad negative and positive axis
    d_gamma = nn_ops._np_layer_norm_gamma_grad(np, d_out, x, eps=1e-5, axis=-1)
    assert d_gamma.shape == (4,)
    d_gamma_axis1 = nn_ops._np_layer_norm_gamma_grad(np, d_out, x, epsilon=1e-5, axis=1)
    assert d_gamma_axis1.shape == (3,)

    # Beta grad negative and positive axis
    d_beta = nn_ops._np_layer_norm_beta_grad(np, d_out, axis=-1)
    assert d_beta.shape == (4,)
    d_beta_axis1 = nn_ops._np_layer_norm_beta_grad(np, d_out, axis=1)
    assert d_beta_axis1.shape == (3,)


def test_rms_norm_forward_and_grads() -> None:
    """Test RMSNorm forward and gradient operations."""
    x = np.random.randn(2, 3, 4).astype(np.float32)
    gamma = np.ones(4, dtype=np.float32)
    d_out = np.ones_like(x)

    # Forward with gamma and without gamma
    out = nn_ops._np_rms_norm(np, x, gamma, eps=1e-5)
    assert out.shape == x.shape
    out_no_g = nn_ops._np_rms_norm(np, x, None, epsilon=1e-5)
    assert out_no_g.shape == x.shape

    # Input grad with gamma and without gamma
    dx = nn_ops._np_rms_norm_input_grad(np, d_out, x, gamma, eps=1e-5)
    assert dx.shape == x.shape
    dx_no_g = nn_ops._np_rms_norm_input_grad(np, d_out, x, None, epsilon=1e-5)
    assert dx_no_g.shape == x.shape

    # Gamma grad
    d_gamma = nn_ops._np_rms_norm_gamma_grad(np, d_out, x, eps=1e-5)
    assert d_gamma.shape == (4,)


def test_group_norm_forward_and_grads() -> None:
    """Test GroupNorm forward and gradient operations."""
    x = np.random.randn(2, 4, 3, 3).astype(np.float32)
    gamma = np.ones(4, dtype=np.float32)
    beta = np.zeros(4, dtype=np.float32)
    d_out = np.ones_like(x)

    # Forward with num_groups, gamma, beta
    out = nn_ops._np_group_norm(np, x, gamma, beta, num_groups=2, eps=1e-5)
    assert out.shape == x.shape

    # Forward with groups, None gamma, None beta
    out_g = nn_ops._np_group_norm(np, x, None, None, groups=2, epsilon=1e-5)
    assert out_g.shape == x.shape

    # Input grad with gamma and without gamma
    dx = nn_ops._np_group_norm_input_grad(np, d_out, x, gamma, num_groups=2, eps=1e-5)
    assert dx.shape == x.shape
    dx_no_g = nn_ops._np_group_norm_input_grad(np, d_out, x, None, groups=2, epsilon=1e-5)
    assert dx_no_g.shape == x.shape

    # Gamma grad with num_groups vs groups
    dg = nn_ops._np_group_norm_gamma_grad(np, d_out, x, num_groups=2, eps=1e-5)
    assert dg.shape == (4,)
    dg2 = nn_ops._np_group_norm_gamma_grad(np, d_out, x, groups=2, epsilon=1e-5)
    assert dg2.shape == (4,)

    # Beta grad
    db = nn_ops._np_group_norm_beta_grad(np, d_out)
    assert db.shape == (4,)


def test_avg_pool2d_forward_and_grads() -> None:
    """Test AvgPool2D forward and gradient operations."""
    x = np.arange(32, dtype=np.float32).reshape(1, 2, 4, 4)

    # Forward int kernel and stride
    out1 = nn_ops._np_avg_pool2d(np, x, kernel_size=2, stride=2)
    assert out1.shape == (1, 2, 2, 2)

    # Forward tuple pool_size, strides, and default stride
    out2 = nn_ops._np_avg_pool2d(np, x, pool_size=(2, 2), strides=(2, 2))
    assert np.allclose(out1, out2)
    out3 = nn_ops._np_avg_pool2d(np, x, pool_size=(2, 2))
    assert out3.shape == (1, 2, 2, 2)

    # Grad int kernel and stride
    d_out = np.ones((1, 2, 2, 2), dtype=np.float32)
    dx1 = nn_ops._np_avg_pool2d_grad(np, d_out, x, kernel_size=2, stride=2)
    assert dx1.shape == x.shape

    # Grad tuple pool_size and default stride
    dx2 = nn_ops._np_avg_pool2d_grad(np, d_out, x, pool_size=(2, 2), strides=(2, 2))
    assert np.allclose(dx1, dx2)
    dx3 = nn_ops._np_avg_pool2d_grad(np, d_out, x, pool_size=(2, 2))
    assert dx3.shape == x.shape


def test_max_pool2d_with_argmax_grad() -> None:
    """Test MaxPool2DWithArgmaxGrad operation with different parameter types."""
    x = np.arange(32, dtype=np.float32).reshape(1, 2, 4, 4)
    d_out = np.ones((1, 2, 2, 2), dtype=np.float32)

    # Grad int kernel and stride
    dx1 = nn_ops._np_max_pool2d_with_argmax_grad(np, d_out, x, kernel_size=2, stride=2)
    assert dx1.shape == x.shape

    # Grad tuple pool_size, strides, and default stride
    dx2 = nn_ops._np_max_pool2d_with_argmax_grad(np, d_out, x, pool_size=(2, 2), strides=(2, 2))
    assert np.allclose(dx1, dx2)
    dx3 = nn_ops._np_max_pool2d_with_argmax_grad(np, d_out, x, pool_size=(2, 2))
    assert dx3.shape == x.shape
