"""Tests for test_random_stateless."""

from __future__ import annotations

import unittest.mock as mock
from typing import TYPE_CHECKING, cast

import numpy as np

from ml_switcheroo_compiler.core.device import Device
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.ops.random_stateless import Generator, stateless_split

if TYPE_CHECKING:
    from collections.abc import Callable

    from ml_switcheroo_compiler.backends.eager_registry import EagerTensorProtocol, EagerValue


def create_eager_tensor(data: EagerValue) -> EagerValue:
    """Test function create_eager_tensor."""
    return cast("Callable[..., EagerTensorProtocol]", Tensor)(data, cast("Callable[..., EagerTensorProtocol]", TensorConfig)(cast("EagerTensorProtocol", getattr(data, "shape")), DType.Float32, cast("Callable[..., EagerTensorProtocol]", Device)("cpu")))


def test_random_stateless_missing() -> None:
    """Test function test_random_stateless_missing."""
    gen = Generator()

    with mock.patch("ml_switcheroo_compiler.tracing.state.global_tracing_state.is_tracing", True):
        with mock.patch("ml_switcheroo_compiler.tracing.state.global_tracing_state.active_graph"):
            res1 = gen.normal((2,))
            res2 = gen.uniform((2,))
            assert isinstance(res1, Tensor)
            assert isinstance(res2, Tensor)

            seed_tensor = create_eager_tensor(np.array([42, 0]))
            res3 = stateless_split(seed_tensor, num=2)
            assert isinstance(res3, Tensor)


def test_random_extras() -> None:
    """Test global generator and stateless rng state creation."""
    from ml_switcheroo_compiler.ops.random_stateless import create_rng_state, get_global_generator

    assert get_global_generator() is not None
    assert create_rng_state(0) is not None
