"""Tests for eager optimizers."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np

from ml_switcheroo_compiler.backends.eager.optimizers import (
    apply_adagrad,
    apply_adam,
    apply_ftrl,
    apply_rmsprop,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from ml_switcheroo_compiler.backends.eager_registry import EagerTensorProtocol, EagerValue


class MockBackend:
    """Mock backend for optimizers."""

    def add(self, a: EagerValue, b: EagerValue) -> EagerValue:
        """Test function add."""
        return cast("EagerTensorProtocol", a) + cast("EagerTensorProtocol", b)

    def subtract(self, a: EagerValue, b: EagerValue) -> EagerValue:
        """Test function subtract."""
        return cast("EagerTensorProtocol", a) - cast("EagerTensorProtocol", b)

    def multiply(self, a: EagerValue, b: EagerValue) -> EagerValue:
        """Test function multiply."""
        return cast("EagerTensorProtocol", a) * cast("EagerTensorProtocol", b)

    def divide(self, a: EagerValue, b: EagerValue) -> EagerValue:
        """Test function divide."""
        return cast("EagerTensorProtocol", a) / cast("EagerTensorProtocol", b)

    def sqrt(self, a: EagerValue) -> EagerValue:
        """Test function sqrt."""
        return cast("EagerValue", cast("Callable[..., EagerTensorProtocol]", np.sqrt)(a))


def test_apply_adam() -> None:
    """Test function test_apply_adam."""
    backend = MockBackend()
    param = np.array([1.0, 2.0])
    m = np.array([0.1, 0.2])
    v = np.array([0.01, 0.04])
    grad = np.array([0.5, 0.5])
    lr = 0.01

    p_new, m_new, v_new = cast("Callable[..., EagerTensorProtocol]", apply_adam)(backend, param, m, v, grad, lr)
    assert cast("EagerTensorProtocol", getattr(p_new, "shape")) == (2,)
    assert cast("EagerTensorProtocol", getattr(m_new, "shape")) == (2,)
    assert cast("EagerTensorProtocol", getattr(v_new, "shape")) == (2,)


def test_apply_adagrad() -> None:
    """Test function test_apply_adagrad."""
    backend = MockBackend()
    param = np.array([1.0, 2.0])
    accum = np.array([0.1, 0.2])
    grad = np.array([0.5, 0.5])
    lr = 0.01

    p_new, accum_new = cast("Callable[..., EagerTensorProtocol]", apply_adagrad)(backend, param, accum, grad, lr)
    assert cast("EagerTensorProtocol", getattr(p_new, "shape")) == (2,)
    assert cast("EagerTensorProtocol", getattr(accum_new, "shape")) == (2,)


def test_apply_ftrl() -> None:
    """Test function test_apply_ftrl."""
    backend = MockBackend()
    param = np.array([1.0, 2.0])
    accum = np.array([0.1, 0.2])
    linear = np.array([0.05, 0.1])
    grad = np.array([0.5, 0.5])
    lr = 0.01

    p_new, accum_new, linear_new = cast("Callable[..., EagerTensorProtocol]", apply_ftrl)(backend, param, accum, linear, grad, lr)
    assert cast("EagerTensorProtocol", getattr(p_new, "shape")) == (2,)
    assert cast("EagerTensorProtocol", getattr(accum_new, "shape")) == (2,)
    assert cast("EagerTensorProtocol", getattr(linear_new, "shape")) == (2,)


def test_apply_rmsprop() -> None:
    """Test function test_apply_rmsprop."""
    backend = MockBackend()
    param = np.array([1.0, 2.0])
    ms = np.array([0.1, 0.2])
    mom = np.array([0.05, 0.1])
    grad = np.array([0.5, 0.5])
    lr = 0.01

    p_new, ms_new, mom_new = cast("Callable[..., EagerTensorProtocol]", apply_rmsprop)(backend, param, ms, mom, grad, lr)
    assert cast("EagerTensorProtocol", getattr(p_new, "shape")) == (2,)
    assert cast("EagerTensorProtocol", getattr(ms_new, "shape")) == (2,)
    assert cast("EagerTensorProtocol", getattr(mom_new, "shape")) == (2,)
