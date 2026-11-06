"""Tests for test_optimizers."""

from __future__ import annotations

import typing

import numpy as np

from ml_switcheroo_compiler.backends.numpy.eager.optimizers_ops import _np_apply_adagrad, _np_apply_adam, _np_apply_ftrl, _np_apply_rmsprop
from tests.testing_utils import ensure_callable, ensure_eager_tensor

if typing.TYPE_CHECKING:
    pass


def test_optimizers_missing() -> None:
    """Test function test_optimizers_missing."""
    p, m, v, g = np.array([1.0]), np.array([0.0]), np.array([0.0]), np.array([0.1])
    res = _np_apply_adam(ensure_eager_tensor(np), p, m, v, g)
    assert ensure_callable(len)(res) == 3

    res2 = _np_apply_adagrad(ensure_eager_tensor(np), p, m, g)
    assert ensure_callable(len)(res2) == 2

    res3 = _np_apply_ftrl(ensure_eager_tensor(np), p, m, v, g)
    assert ensure_callable(len)(res3) == 3

    res4 = _np_apply_rmsprop(ensure_eager_tensor(np), p, m, v, g)
    assert ensure_callable(len)(res4) == 3
