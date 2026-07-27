"""Tests for numpy eager activation ops."""

import numpy as np

from ml_switcheroo_compiler.backends.numpy.eager.activation_ops import (
    _np_celu,
    _np_elu,
    _np_log_sigmoid,
    _np_mish,
    _np_relu,
    _np_softplus,
    _np_softsign,
)


def test_relu() -> None:
    """Test relu.

    Returns:
        None
    """
    x = np.array([-1.0, 1.0])
    res = _np_relu(np, x)
    np.testing.assert_allclose(res, [0.0, 1.0])


def test_elu() -> None:
    """Test elu.

    Returns:
        None
    """
    x = np.array([-1.0, 1.0])
    res = _np_elu(np, x, alpha=2.0)
    expected = np.where(x > 0, x, 2.0 * (np.exp(x) - 1.0))
    np.testing.assert_allclose(res, expected)


def test_celu() -> None:
    """Test celu.

    Returns:
        None
    """
    x = np.array([-1.0, 1.0])
    res = _np_celu(np, x, alpha=2.0)
    expected = np.maximum(0.0, x) + np.minimum(0.0, 2.0 * (np.exp(x / 2.0) - 1.0))
    np.testing.assert_allclose(res, expected)


def test_softplus() -> None:
    """Test softplus.

    Returns:
        None
    """
    x = np.array([-1.0, 1.0])
    res = _np_softplus(np, x)
    expected = np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0.0)
    np.testing.assert_allclose(res, expected)


def test_softsign() -> None:
    """Test softsign.

    Returns:
        None
    """
    x = np.array([-1.0, 1.0])
    res = _np_softsign(np, x)
    expected = x / (1.0 + np.abs(x))
    np.testing.assert_allclose(res, expected)


def test_mish() -> None:
    """Test mish.

    Returns:
        None
    """
    x = np.array([-1.0, 1.0])
    res = _np_mish(np, x)
    softplus_x = np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0.0)
    expected = x * np.tanh(softplus_x)
    np.testing.assert_allclose(res, expected)


def test_log_sigmoid() -> None:
    """Test log sigmoid.

    Returns:
        None
    """
    x = np.array([-1.0, 1.0])
    res = _np_log_sigmoid(np, x)
    expected = -np.log1p(np.exp(-x))
    np.testing.assert_allclose(res, expected)
