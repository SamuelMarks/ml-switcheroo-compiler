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

if typing.TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import EagerCallableProtocol, EagerTensorProtocol, EagerValue


class MockBackend:
    """Mock backend for optimizers."""

    def add(self, a: EagerValue, b: EagerValue) -> EagerValue:
        """Test function add.

        Args:
            a: Test parameter `a`.
            b: Test parameter `b`.
        """
        return typing.cast("EagerTensorProtocol", a) + typing.cast("EagerTensorProtocol", b)

    def subtract(self, a: EagerValue, b: EagerValue) -> EagerValue:
        """Test function subtract.

        Args:
            a: Test parameter `a`.
            b: Test parameter `b`.
        """
        return typing.cast("EagerTensorProtocol", a) - typing.cast("EagerTensorProtocol", b)

    def multiply(self, a: EagerValue, b: EagerValue) -> EagerValue:
        """Test function multiply.

        Args:
            a: Test parameter `a`.
            b: Test parameter `b`.
        """
        return typing.cast("EagerTensorProtocol", a) * typing.cast("EagerTensorProtocol", b)

    def divide(self, a: EagerValue, b: EagerValue) -> EagerValue:
        """Test function divide.

        Args:
            a: Test parameter `a`.
            b: Test parameter `b`.
        """
        return typing.cast("EagerTensorProtocol", a) / typing.cast("EagerTensorProtocol", b)

    def sqrt(self, a: EagerValue) -> EagerValue:
        """Test function sqrt.

        Args:
            a: Test parameter `a`.
        """
        return typing.cast("EagerValue", typing.cast("EagerCallableProtocol", np.sqrt)(a))


def test_apply_adam() -> None:
    """Test function test_apply_adam."""
    backend = MockBackend()
    param = np.array([1.0, 2.0])
    m = np.array([0.1, 0.2])
    v = np.array([0.01, 0.04])
    grad = np.array([0.5, 0.5])
    lr = 0.01

    p_new, m_new, v_new = typing.cast("EagerCallableProtocol", apply_adam)(backend, param, m, v, grad, lr)
    assert typing.cast("EagerTensorProtocol", getattr(p_new, "shape")) == (2,)
    assert typing.cast("EagerTensorProtocol", getattr(m_new, "shape")) == (2,)
    assert typing.cast("EagerTensorProtocol", getattr(v_new, "shape")) == (2,)


def test_apply_adagrad() -> None:
    """Test function test_apply_adagrad."""
    backend = MockBackend()
    param = np.array([1.0, 2.0])
    accum = np.array([0.1, 0.2])
    grad = np.array([0.5, 0.5])
    lr = 0.01

    p_new, accum_new = typing.cast("EagerCallableProtocol", apply_adagrad)(backend, param, accum, grad, lr)
    assert typing.cast("EagerTensorProtocol", getattr(p_new, "shape")) == (2,)
    assert typing.cast("EagerTensorProtocol", getattr(accum_new, "shape")) == (2,)


def test_apply_ftrl() -> None:
    """Test function test_apply_ftrl."""
    backend = MockBackend()
    param = np.array([1.0, 2.0])
    accum = np.array([0.1, 0.2])
    linear = np.array([0.05, 0.1])
    grad = np.array([0.5, 0.5])
    lr = 0.01

    p_new, accum_new, linear_new = typing.cast("EagerCallableProtocol", apply_ftrl)(backend, param, accum, linear, grad, lr)
    assert typing.cast("EagerTensorProtocol", getattr(p_new, "shape")) == (2,)
    assert typing.cast("EagerTensorProtocol", getattr(accum_new, "shape")) == (2,)
    assert typing.cast("EagerTensorProtocol", getattr(linear_new, "shape")) == (2,)


def test_apply_rmsprop() -> None:
    """Test function test_apply_rmsprop."""
    backend = MockBackend()
    param = np.array([1.0, 2.0])
    ms = np.array([0.1, 0.2])
    mom = np.array([0.05, 0.1])
    grad = np.array([0.5, 0.5])
    lr = 0.01

    p_new, ms_new, mom_new = typing.cast("EagerCallableProtocol", apply_rmsprop)(backend, param, ms, mom, grad, lr)
    assert typing.cast("EagerTensorProtocol", getattr(p_new, "shape")) == (2,)
    assert typing.cast("EagerTensorProtocol", getattr(ms_new, "shape")) == (2,)
    assert typing.cast("EagerTensorProtocol", getattr(mom_new, "shape")) == (2,)
