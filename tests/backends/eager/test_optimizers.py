"""Tests for eager optimizers."""

import numpy as np

from ml_switcheroo_compiler.backends.eager.optimizers import (
    apply_adagrad,
    apply_adam,
    apply_ftrl,
    apply_rmsprop,
)


class MockBackend:
    """Mock backend for optimizers."""

    def add(self, a, b):
        return a + b

    def subtract(self, a, b):
        return a - b

    def multiply(self, a, b):
        return a * b

    def divide(self, a, b):
        return a / b

    def sqrt(self, a):
        return np.sqrt(a)


def test_apply_adam():
    backend = MockBackend()
    param = np.array([1.0, 2.0])
    m = np.array([0.1, 0.2])
    v = np.array([0.01, 0.04])
    grad = np.array([0.5, 0.5])
    lr = 0.01

    p_new, m_new, v_new = apply_adam(backend, param, m, v, grad, lr)
    assert p_new.shape == (2,)
    assert m_new.shape == (2,)
    assert v_new.shape == (2,)


def test_apply_adagrad():
    backend = MockBackend()
    param = np.array([1.0, 2.0])
    accum = np.array([0.1, 0.2])
    grad = np.array([0.5, 0.5])
    lr = 0.01

    p_new, accum_new = apply_adagrad(backend, param, accum, grad, lr)
    assert p_new.shape == (2,)
    assert accum_new.shape == (2,)


def test_apply_ftrl():
    backend = MockBackend()
    param = np.array([1.0, 2.0])
    accum = np.array([0.1, 0.2])
    linear = np.array([0.05, 0.1])
    grad = np.array([0.5, 0.5])
    lr = 0.01

    p_new, accum_new, linear_new = apply_ftrl(backend, param, accum, linear, grad, lr)
    assert p_new.shape == (2,)
    assert accum_new.shape == (2,)
    assert linear_new.shape == (2,)


def test_apply_rmsprop():
    backend = MockBackend()
    param = np.array([1.0, 2.0])
    ms = np.array([0.1, 0.2])
    mom = np.array([0.05, 0.1])
    grad = np.array([0.5, 0.5])
    lr = 0.01

    p_new, ms_new, mom_new = apply_rmsprop(backend, param, ms, mom, grad, lr)
    assert p_new.shape == (2,)
    assert ms_new.shape == (2,)
    assert mom_new.shape == (2,)
