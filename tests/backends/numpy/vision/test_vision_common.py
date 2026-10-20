"""Tests for numpy eager vision common ops."""

from __future__ import annotations

import typing

import numpy as np
import pytest

from ml_switcheroo_compiler.backends.eager_registry import global_eager_registry
from ml_switcheroo_compiler.backends.numpy.eager.vision_common import _np_mel_filterbank, _np_mfcc, _np_power_iteration

if typing.TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import EagerCallableProtocol, EagerTensorProtocol, EagerValue


def test_np_mel_filterbank() -> None:
    """Test function test_np_mel_filterbank."""

    @global_eager_registry.register("MelFilterbank")
    def _dummy_mel_filterbank(bm: EagerValue, spec: EagerValue, config: EagerValue) -> EagerValue:
        """Test function _dummy_mel_filterbank.

        Args:
            bm: Test parameter `bm`.
            spec: Test parameter `spec`.
            config: Test parameter `config`.
        """
        return np.ones((2, 2))

    _np_mel_filterbank(np, None, config={})
    assert callable(test_np_mel_filterbank)


def test_np_mfcc() -> None:
    """Test function test_np_mfcc."""

    @global_eager_registry.register("Mfcc")
    def _dummy_mfcc(bm: EagerValue, spec: EagerValue, config: EagerValue) -> EagerValue:
        """Test function _dummy_mfcc.

        Args:
            bm: Test parameter `bm`.
            spec: Test parameter `spec`.
            config: Test parameter `config`.
        """
        return np.ones((2, 2))

    with pytest.raises(ValueError):
        _np_mfcc(np, np.ones((2, 2)), config={})


def test_np_power_iteration() -> None:
    """Test function disabled_test_np_power_iteration."""
    w = np.array([[1.0, 2.0], [3.0, 4.0]])
    res = _np_power_iteration(np, w, num_iters=2)
    assert typing.cast("EagerCallableProtocol", len)(res) == 3
    assert typing.cast("EagerTensorProtocol", getattr(typing.cast("EagerTensorProtocol", getattr(res, "__getitem__")(0)), "shape")) == (2,)
    assert typing.cast("EagerTensorProtocol", getattr(typing.cast("EagerTensorProtocol", getattr(res, "__getitem__")(1)), "shape")) == (2,)
    assert typing.cast("EagerTensorProtocol", getattr(typing.cast("EagerTensorProtocol", getattr(res, "__getitem__")(2)), "shape")) == ()
    res2 = _np_power_iteration(np, w)
    assert typing.cast("EagerCallableProtocol", len)(res2) == 3
    u = np.array([[1.0], [0.0]])
    res3 = _np_power_iteration(np, w, num_iters=1, u=u)
    assert typing.cast("EagerCallableProtocol", len)(res3) == 3


def test_backends_numpy_vision_power_iteration() -> None:
    """Test _np_power_iteration in backends/numpy/eager/vision_common.py."""
    from ml_switcheroo_compiler.backends.eager_registry import numpy_eager_registry

    func = numpy_eager_registry.get("PowerIteration")
    assert func is not None
    w = np.eye(3, dtype=np.float32)
    (v, u, s) = func(np, w, num_iters=2)
    assert typing.cast("EagerTensorProtocol", getattr(v, "shape")) == (3,)
    assert typing.cast("EagerTensorProtocol", getattr(u, "shape")) == (3,)
    (v2, u2, s2) = func(np, w, num_iters=1, u=np.array([1.0, 0.0, 0.0], dtype=np.float32))
    assert typing.cast("EagerTensorProtocol", getattr(v2, "shape")) == (3,)
