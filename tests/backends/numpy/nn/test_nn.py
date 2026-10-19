"""Tests for test_nn."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np
import pytest

import ml_switcheroo_compiler.backends.eager as eager_mod
from ml_switcheroo_compiler.backends.numpy.eager.nn import _gelu, _np_activity_regularization, _np_alpha_dropout, _np_dropout, _np_relu, _np_rope, _np_rrelu, _np_time_distributed

if TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import EagerCallableProtocol, EagerTensorProtocol, EagerValue


def test_gelu() -> None:
    """Test function test_gelu."""
    res = _gelu(np.array([-1.0, 1.0]))
    assert res.shape == (2,)


def test_np_relu() -> None:
    """Test function test_np_relu."""
    res = _np_relu(np, np.array([-1.0, 1.0]))
    cast("EagerCallableProtocol", np.testing.assert_allclose)(res, [0.0, 1.0])


def test_np_alpha_dropout() -> None:
    """Test function test_np_alpha_dropout."""
    x = np.ones((2, 2))
    res1 = _np_alpha_dropout(np, x, training=False)
    cast("EagerCallableProtocol", np.testing.assert_allclose)(res1, x)
    res2 = _np_alpha_dropout(np, x, rate=0.0, training=True)
    cast("EagerCallableProtocol", np.testing.assert_allclose)(res2, x)
    res3 = _np_alpha_dropout(np, x, rate=0.5, training=True, seed=42)
    assert cast("EagerTensorProtocol", getattr(res3, "shape")) == (2, 2)
    res4 = _np_alpha_dropout(np, x, rate=0.5, training=True, seed=42, noise_shape=(2, 2))
    assert cast("EagerTensorProtocol", getattr(res4, "shape")) == (2, 2)


def test_np_activity_regularization() -> None:
    """Test function test_np_activity_regularization."""
    res = _np_activity_regularization(np, 5)
    assert res == 5


def test_np_dropout() -> None:
    """Test function test_np_dropout."""
    x = np.ones((2, 2))
    res1 = _np_dropout(np, x, training=False)
    cast("EagerCallableProtocol", np.testing.assert_allclose)(res1, x)
    res2 = _np_dropout(np, x, rate=0.0, training=True)
    cast("EagerCallableProtocol", np.testing.assert_allclose)(res2, x)
    res3 = _np_dropout(np, x, rate=0.5, training=True, seed=42)
    assert cast("EagerTensorProtocol", getattr(res3, "shape")) == (2, 2)
    res4 = _np_dropout(np, x, rate=0.5, training=True, seed=42, noise_shape=(2, 2))
    assert cast("EagerTensorProtocol", getattr(res4, "shape")) == (2, 2)


def test_np_time_distributed() -> None:
    """Test function test_np_time_distributed."""
    x2d = np.ones((2, 2))
    x3d = np.ones((2, 2, 2))

    class MockBackend:
        """Test class MockBackend."""

        def execute_op(self, name: EagerValue, x: EagerValue, **kwargs: EagerValue) -> EagerValue:
            """Test function execute_op.

            Args:
                name: Test parameter `name`.
                x: Test parameter `x`.
                kwargs: Test parameter `kwargs`.
            """
            return cast("EagerTensorProtocol", x) * cast("EagerTensorProtocol", 2)

    import ml_switcheroo_compiler.backends.numpy.eager.nn as nn_mod

    original_get_backend = cast("EagerTensorProtocol", getattr(nn_mod, "get_active_backend"))
    setattr(nn_mod, "get_active_backend", lambda: MockBackend())
    try:
        res2d = _np_time_distributed(np, x2d, wrapped_op_name="MockOp")
        cast("EagerCallableProtocol", np.testing.assert_allclose)(res2d, x2d * 2)
        res3d = _np_time_distributed(np, x3d, wrapped_op_name="MockOp")
        cast("EagerCallableProtocol", np.testing.assert_allclose)(res3d, x3d * 2)
    finally:
        setattr(nn_mod, "get_active_backend", original_get_backend)


def test_np_rope() -> None:
    """Test function test_np_rope."""
    x = np.ones((1, 4))
    res = _np_rope(np, x, dim=4)
    assert cast("EagerTensorProtocol", getattr(res, "shape")) == (1, 4)


def test_np_rrelu() -> None:
    """Test function test_np_rrelu."""
    x = np.array([-1.0, 1.0])
    res1 = _np_rrelu(np, x, training=False)
    alpha = (1.0 / 8.0 + 1.0 / 3.0) / 2.0
    expected1 = np.array([-alpha, 1.0])
    cast("EagerCallableProtocol", np.testing.assert_allclose)(res1, expected1)
    res2 = _np_rrelu(np, x, training=True)
    assert cast("EagerTensorProtocol", getattr(res2, "__getitem__")(1)) == 1.0
    assert -1.0 / 3.0 <= getattr(res2, "__getitem__")(0) <= -1.0 / 8.0


def test_numpy_nn_eager_extra() -> None:
    """Test the numpy nn eager extra behavior.

    Returns:
        TensorOrScalar: The inferred shape or computed result.
    """
    _gelu(np.array([-1.0, 2.0, 5.0]))
    res = _np_relu(np, np.array([-1.0, 2.0, 5.0]))
    assert cast("EagerCallableProtocol", np.allclose)(res, [0.0, 2.0, 5.0])
    res = _np_alpha_dropout(np, np.ones((2, 2)), rate=0.5, seed=42, training=True)
    assert cast("EagerTensorProtocol", getattr(res, "shape")) == (2, 2)
    res = _np_alpha_dropout(np, np.ones((2, 2)), rate=0.0)
    assert cast("EagerCallableProtocol", np.allclose)(res, 1.0)
    res = _np_alpha_dropout(np, np.ones((2, 2)), rate=0.5, seed=42, noise_shape=(2, 1), training=True)
    assert cast("EagerTensorProtocol", getattr(res, "shape")) == (2, 2)
    res = _np_activity_regularization(np, np.ones((2, 2)), l1=0.1, l2=0.2)
    assert cast("EagerTensorProtocol", getattr(res, "shape")) == (2, 2)
    res = _np_dropout(np, np.ones((2, 2)), rate=0.5, seed=42, training=True)
    assert cast("EagerTensorProtocol", getattr(res, "shape")) == (2, 2)
    res = _np_dropout(np, np.ones((2, 2)), rate=0.0)
    assert cast("EagerCallableProtocol", np.allclose)(res, 1.0)
    res = _np_dropout(np, np.ones((2, 2)), rate=0.5, seed=42, noise_shape=(2, 1), training=True)
    assert cast("EagerTensorProtocol", getattr(res, "shape")) == (2, 2)


def test_np_time_distributed_2(monkeypatch: EagerTensorProtocol) -> None:
    """Test the np time distributed behavior.

    Args:
        monkeypatch (EagerTensorProtocol): The monkeypatch parameter.

    Returns:
        TensorOrScalar: The inferred shape or computed result.
    """
    with pytest.raises(AttributeError):

        def mock_exec(backend: EagerValue, op_name: EagerValue, x: EagerValue, **kwargs: EagerValue) -> EagerValue:
            """Evaluate and process the mock exec operation.

            Args:
                backend (EagerValue): Required parameter for backend.
                op_name (EagerValue): Required parameter for op_name.
                x (EagerValue): Required parameter for x.
                **kwargs (EagerValue): Arbitrary keyword arguments.

            Returns:
                EagerValue: The evaluated or processed output.
            """
            return x

        monkeypatch.setattr(eager_mod, "execute_generic_op", mock_exec)
        res = _np_time_distributed(np, np.ones((2, 2)), wrapped_op_name="Dummy")
        assert cast("EagerTensorProtocol", getattr(res, "shape")) == (2, 2)
        res = _np_time_distributed(np, np.ones((2, 3, 4)), wrapped_op_name="Dummy")
        assert cast("EagerTensorProtocol", getattr(res, "shape")) == (2, 3, 4)
