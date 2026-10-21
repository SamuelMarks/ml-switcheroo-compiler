"""Tests for eager optimizers."""

from __future__ import annotations

import typing

import numpy as np

from ml_switcheroo_compiler.backends.eager.optimizers import (
    apply_adagrad,
    apply_adam,
    apply_ftrl,
    apply_rmsprop,
)
from tests.testing_utils import as_callable, as_tensor

if typing.TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import EagerValue


class MockBackend:
    """Mock backend for optimizers."""

    def add(self, a: EagerValue, b: EagerValue) -> EagerValue:
        """Test function add.

        Args:
            a: Test parameter `a`.
            b: Test parameter `b`.
        """
        return as_tensor(a) + as_tensor(b)

    def subtract(self, a: EagerValue, b: EagerValue) -> EagerValue:
        """Test function subtract.

        Args:
            a: Test parameter `a`.
            b: Test parameter `b`.
        """
        return as_tensor(a) - as_tensor(b)

    def multiply(self, a: EagerValue, b: EagerValue) -> EagerValue:
        """Test function multiply.

        Args:
            a: Test parameter `a`.
            b: Test parameter `b`.
        """
        return as_tensor(a) * as_tensor(b)

    def divide(self, a: EagerValue, b: EagerValue) -> EagerValue:
        """Test function divide.

        Args:
            a: Test parameter `a`.
            b: Test parameter `b`.
        """
        return as_tensor(a) / as_tensor(b)

    def sqrt(self, a: EagerValue) -> EagerValue:
        """Test function sqrt.

        Args:
            a: Test parameter `a`.
        """
        return as_tensor(as_callable(np.sqrt)(a))


def test_apply_adam() -> None:
    """Test function test_apply_adam."""
    backend = MockBackend()
    param = np.array([1.0, 2.0])
    m = np.array([0.1, 0.2])
    v = np.array([0.01, 0.04])
    grad = np.array([0.5, 0.5])
    lr = 0.01

    p_new, m_new, v_new = as_callable(apply_adam)(backend, param, m, v, grad, lr)
    assert as_tensor(getattr(p_new, "shape")) == (2,)
    assert as_tensor(getattr(m_new, "shape")) == (2,)
    assert as_tensor(getattr(v_new, "shape")) == (2,)


def test_apply_adagrad() -> None:
    """Test function test_apply_adagrad."""
    backend = MockBackend()
    param = np.array([1.0, 2.0])
    accum = np.array([0.1, 0.2])
    grad = np.array([0.5, 0.5])
    lr = 0.01

    p_new, accum_new = as_callable(apply_adagrad)(backend, param, accum, grad, lr)
    assert as_tensor(getattr(p_new, "shape")) == (2,)
    assert as_tensor(getattr(accum_new, "shape")) == (2,)


def test_apply_ftrl() -> None:
    """Test function test_apply_ftrl."""
    backend = MockBackend()
    param = np.array([1.0, 2.0])
    accum = np.array([0.1, 0.2])
    linear = np.array([0.05, 0.1])
    grad = np.array([0.5, 0.5])
    lr = 0.01

    p_new, accum_new, linear_new = as_callable(apply_ftrl)(backend, param, accum, linear, grad, lr)
    assert as_tensor(getattr(p_new, "shape")) == (2,)
    assert as_tensor(getattr(accum_new, "shape")) == (2,)
    assert as_tensor(getattr(linear_new, "shape")) == (2,)


def test_apply_rmsprop() -> None:
    """Test function test_apply_rmsprop."""
    backend = MockBackend()
    param = np.array([1.0, 2.0])
    ms = np.array([0.1, 0.2])
    mom = np.array([0.05, 0.1])
    grad = np.array([0.5, 0.5])
    lr = 0.01

    p_new, ms_new, mom_new = as_callable(apply_rmsprop)(backend, param, ms, mom, grad, lr)
    assert as_tensor(getattr(p_new, "shape")) == (2,)
    assert as_tensor(getattr(ms_new, "shape")) == (2,)
    assert as_tensor(getattr(mom_new, "shape")) == (2,)
