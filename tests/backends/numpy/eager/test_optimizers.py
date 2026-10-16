"""Tests for test_optimizers."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np

from ml_switcheroo_compiler.backends.numpy.eager.optimizers_ops import _np_apply_adagrad, _np_apply_adam, _np_apply_ftrl, _np_apply_rmsprop

if TYPE_CHECKING:
    from collections.abc import Callable

    from ml_switcheroo_compiler.backends.eager_registry import EagerTensorProtocol


def test_optimizers_missing() -> None:
    """Test function test_optimizers_missing."""
    p, m, v, g = np.array([1.0]), np.array([0.0]), np.array([0.0]), np.array([0.1])
    res = _np_apply_adam(np, p, m, v, g)
    assert cast("Callable[..., EagerTensorProtocol]", len)(res) == 3

    res2 = _np_apply_adagrad(np, p, m, g)
    assert cast("Callable[..., EagerTensorProtocol]", len)(res2) == 2

    res3 = _np_apply_ftrl(np, p, m, v, g)
    assert cast("Callable[..., EagerTensorProtocol]", len)(res3) == 3

    res4 = _np_apply_rmsprop(np, p, m, v, g)
    assert cast("Callable[..., EagerTensorProtocol]", len)(res4) == 3
