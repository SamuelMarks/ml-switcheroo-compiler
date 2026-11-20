"""Tests for eager optimizers."""

from __future__ import annotations

import typing

import numpy as np

from ml_switcheroo_compiler.backends.eager.optimizers import apply_adagrad, apply_adam, apply_ftrl, apply_rmsprop
from ml_switcheroo_compiler.backends.numpy.eager import ensure_ndarray
from tests.testing_utils import MockTestStub, coerce_numeric, extract_test_shape

if typing.TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import BackendArg


class MockBackend(MockTestStub):
    """Mock backend for optimizers."""

    def add(self, a: BackendArg, b: BackendArg) -> BackendArg:
        """Test function add.

        Args:
            a: Test parameter `a`.
            b: Test parameter `b`.

        Returns:
            BackendArg: Computed output value from `add`.
        """
        return coerce_numeric(a) + coerce_numeric(b)

    def subtract(self, a: BackendArg, b: BackendArg) -> BackendArg:
        """Test function subtract.

        Args:
            a: Test parameter `a`.
            b: Test parameter `b`.

        Returns:
            BackendArg: Computed output value from `subtract`.
        """
        return coerce_numeric(a) - coerce_numeric(b)

    def multiply(self, a: BackendArg, b: BackendArg) -> BackendArg:
        """Test function multiply.

        Args:
            a: Test parameter `a`.
            b: Test parameter `b`.

        Returns:
            BackendArg: Computed output value from `multiply`.
        """
        return coerce_numeric(a) * coerce_numeric(b)

    def divide(self, a: BackendArg, b: BackendArg) -> BackendArg:
        """Test function divide.

        Args:
            a: Test parameter `a`.
            b: Test parameter `b`.

        Returns:
            BackendArg: Computed output value from `divide`.
        """
        return coerce_numeric(a) / coerce_numeric(b)

    def sqrt(self, a: BackendArg) -> BackendArg:
        """Test function sqrt.

        Args:
            a: Test parameter `a`.

        Returns:
            BackendArg: Computed output value from `sqrt`.
        """
        return np.sqrt(ensure_ndarray(a))


def test_apply_adam() -> None:
    """Test function test_apply_adam."""
    backend = MockBackend()
    param = np.array([1.0, 2.0])
    m = np.array([0.1, 0.2])
    v = np.array([0.01, 0.04])
    grad = np.array([0.5, 0.5])
    lr = 0.01

    p_new, m_new, v_new = apply_adam(backend, param, m, v, grad, lr)
    assert extract_test_shape(p_new) == (2,)
    assert extract_test_shape(m_new) == (2,)
    assert extract_test_shape(v_new) == (2,)


def test_apply_adagrad() -> None:
    """Test function test_apply_adagrad."""
    backend = MockBackend()
    param = np.array([1.0, 2.0])
    accum = np.array([0.1, 0.2])
    grad = np.array([0.5, 0.5])
    lr = 0.01

    p_new, accum_new = apply_adagrad(backend, param, accum, grad, lr)
    assert extract_test_shape(p_new) == (2,)
    assert extract_test_shape(accum_new) == (2,)


def test_apply_ftrl() -> None:
    """Test function test_apply_ftrl."""
    backend = MockBackend()
    param = np.array([1.0, 2.0])
    accum = np.array([0.1, 0.2])
    linear = np.array([0.05, 0.1])
    grad = np.array([0.5, 0.5])
    lr = 0.01

    p_new, accum_new, linear_new = apply_ftrl(backend, param, accum, linear, grad, lr)
    assert extract_test_shape(p_new) == (2,)
    assert extract_test_shape(accum_new) == (2,)
    assert extract_test_shape(linear_new) == (2,)


def test_apply_rmsprop() -> None:
    """Test function test_apply_rmsprop."""
    backend = MockBackend()
    param = np.array([1.0, 2.0])
    ms = np.array([0.1, 0.2])
    mom = np.array([0.05, 0.1])
    grad = np.array([0.5, 0.5])
    lr = 0.01

    p_new, ms_new, mom_new = apply_rmsprop(backend, param, ms, mom, grad, lr)
    assert extract_test_shape(p_new) == (2,)
    assert extract_test_shape(ms_new) == (2,)
    assert extract_test_shape(mom_new) == (2,)
