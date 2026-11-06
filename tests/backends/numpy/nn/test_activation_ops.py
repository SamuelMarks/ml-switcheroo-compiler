"""Tests for numpy eager activation ops."""

from __future__ import annotations

import typing

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
from tests.testing_utils import ensure_callable, ensure_eager_tensor

if typing.TYPE_CHECKING:
    pass


def test_relu() -> None:
    """Test relu.

    Returns:
        None
    """
    x = np.array([-1.0, 1.0])
    res = _np_relu(ensure_eager_tensor(np), x)
    ensure_callable(np.testing.assert_allclose)(res, [0.0, 1.0])


def test_elu() -> None:
    """Test elu.

    Returns:
        None
    """
    x = np.array([-1.0, 1.0])
    res = _np_elu(ensure_eager_tensor(np), x, alpha=2.0)
    expected = np.where(x > 0, x, 2.0 * (np.exp(x) - 1.0))
    ensure_callable(np.testing.assert_allclose)(res, expected)


def test_celu() -> None:
    """Test celu.

    Returns:
        None
    """
    x = np.array([-1.0, 1.0])
    res = _np_celu(ensure_eager_tensor(np), x, alpha=2.0)
    expected = np.maximum(0.0, x) + np.minimum(0.0, 2.0 * (np.exp(x / 2.0) - 1.0))
    ensure_callable(np.testing.assert_allclose)(res, expected)


def test_softplus() -> None:
    """Test softplus.

    Returns:
        None
    """
    x = np.array([-1.0, 1.0])
    res = _np_softplus(ensure_eager_tensor(np), x)
    expected = np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0.0)
    ensure_callable(np.testing.assert_allclose)(res, expected)


def test_softsign() -> None:
    """Test softsign.

    Returns:
        None
    """
    x = np.array([-1.0, 1.0])
    res = _np_softsign(ensure_eager_tensor(np), x)
    expected = x / (1.0 + np.abs(x))
    ensure_callable(np.testing.assert_allclose)(res, expected)


def test_mish() -> None:
    """Test mish.

    Returns:
        None
    """
    x = np.array([-1.0, 1.0])
    res = _np_mish(ensure_eager_tensor(np), x)
    softplus_x = np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0.0)
    expected = x * np.tanh(softplus_x)
    ensure_callable(np.testing.assert_allclose)(res, expected)


def test_log_sigmoid() -> None:
    """Test log sigmoid.

    Returns:
        None
    """
    x = np.array([-1.0, 1.0])
    res = _np_log_sigmoid(ensure_eager_tensor(np), x)
    expected = -np.log1p(np.exp(-x))
    ensure_callable(np.testing.assert_allclose)(res, expected)
