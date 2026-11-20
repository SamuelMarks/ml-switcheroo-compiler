"""Behavioral unit tests for nn."""

from __future__ import annotations

import typing

import numpy as np
import pytest

import ml_switcheroo_compiler.backends.eager as eager_mod
from ml_switcheroo_compiler.backends.eager.nn_ops import layers as nn_mod
from ml_switcheroo_compiler.backends.eager.nn_ops.layers import (
    _gelu,
    _np_activity_regularization,
    _np_alpha_dropout,
    _np_dropout,
    _np_relu,
    _np_rope,
    _np_rrelu,
    _np_time_distributed,
)
from ml_switcheroo_compiler.backends.eager_registry import numpy_eager_registry
from ml_switcheroo_compiler.backends.numpy.eager import ensure_ndarray
from tests.testing_utils import coerce_numeric, require_callable

if typing.TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import BackendArg


def test_gelu() -> None:
    """Test function test_gelu."""
    res = _gelu(np.array([-1.0, 1.0]))
    assert res.shape == (2,)


def test_np_relu() -> None:
    """Test function test_np_relu."""
    res = _np_relu(np, np.array([-1.0, 1.0]))
    np.testing.assert_allclose(res, [0.0, 1.0])


def test_np_alpha_dropout() -> None:
    """Test function test_np_alpha_dropout."""
    x = np.ones((2, 2))
    res1 = _np_alpha_dropout(np, x, training=False)
    np.testing.assert_allclose(res1, x)
    res2 = _np_alpha_dropout(np, x, rate=0.0, training=True)
    np.testing.assert_allclose(res2, x)
    res3 = _np_alpha_dropout(np, x, rate=0.5, training=True, seed=42)
    assert res3.shape == (2, 2)
    res4 = _np_alpha_dropout(np, x, rate=0.5, training=True, seed=42, noise_shape=(2, 2))
    assert res4.shape == (2, 2)


def test_np_activity_regularization() -> None:
    """Test function test_np_activity_regularization."""
    res = _np_activity_regularization(np, ensure_ndarray(5))
    assert res == 5


def test_np_dropout() -> None:
    """Test function test_np_dropout."""
    x = np.ones((2, 2))
    res1 = _np_dropout(np, x, training=False)
    np.testing.assert_allclose(res1, x)
    res2 = _np_dropout(np, x, rate=0.0, training=True)
    np.testing.assert_allclose(res2, x)
    res3 = _np_dropout(np, x, rate=0.5, training=True, seed=42)
    assert res3.shape == (2, 2)
    res4 = _np_dropout(np, x, rate=0.5, training=True, seed=42, noise_shape=(2, 2))
    assert res4.shape == (2, 2)


def test_np_time_distributed(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test function test_np_time_distributed.

    Args:
        monkeypatch: Pytest monkeypatch fixture.
    """
    x2d = np.ones((2, 2))
    x3d = np.ones((2, 2, 2))

    class MockBackend:
        """Test class MockBackend."""

        def execute_op(self, name: BackendArg, x: BackendArg, **kwargs: BackendArg) -> BackendArg:
            """Test function execute_op.

            Args:
                name: Test parameter `name`.
                x: Test parameter `x`.
                kwargs: Test parameter `kwargs`.

            Returns:
                BackendArg: Computed output value from `execute_op`.
            """
            return coerce_numeric(x) * 2

    monkeypatch.setattr(nn_mod, "get_active_backend", lambda: MockBackend(), raising=False)
    res2d = _np_time_distributed(np, x2d, wrapped_op_name="MockOp")
    np.testing.assert_allclose(res2d, x2d * 2)
    res3d = _np_time_distributed(np, x3d, wrapped_op_name="MockOp")
    np.testing.assert_allclose(res3d, x3d * 2)


def test_np_rope() -> None:
    """Test function test_np_rope."""
    x = np.ones((1, 4))
    res = _np_rope(np, x, dim=4)
    assert res.shape == (1, 4)


def test_np_rrelu() -> None:
    """Test function test_np_rrelu."""
    x = np.array([-1.0, 1.0])
    res1 = _np_rrelu(np, x, training=False)
    alpha = (1.0 / 8.0 + 1.0 / 3.0) / 2.0
    expected1 = np.array([-alpha, 1.0])
    np.testing.assert_allclose(res1, expected1)
    res2 = _np_rrelu(np, x, training=True)
    assert res2[1] == 1.0
    assert -1.0 / 3.0 <= res2[0] <= -1.0 / 8.0


def test_numpy_nn_eager_extra() -> None:
    """Test the numpy nn eager extra behavior.

    Returns:
        OpValue: The inferred shape or computed result.
    """
    _gelu(np.array([-1.0, 2.0, 5.0]))
    res = _np_relu(np, np.array([-1.0, 2.0, 5.0]))
    assert np.allclose(res, [0.0, 2.0, 5.0])
    res = _np_alpha_dropout(np, np.ones((2, 2)), rate=0.5, seed=42, training=True)
    assert res.shape == (2, 2)
    res = _np_alpha_dropout(np, np.ones((2, 2)), rate=0.0)
    assert np.allclose(res, 1.0)
    res = _np_alpha_dropout(np, np.ones((2, 2)), rate=0.5, seed=42, noise_shape=(2, 1), training=True)
    assert res.shape == (2, 2)
    res = _np_activity_regularization(np, np.ones((2, 2)), l1=0.1, l2=0.2)
    assert res.shape == (2, 2)
    res = _np_dropout(np, np.ones((2, 2)), rate=0.5, seed=42, training=True)
    assert res.shape == (2, 2)
    res = _np_dropout(np, np.ones((2, 2)), rate=0.0)
    assert np.allclose(res, 1.0)
    res = _np_dropout(np, np.ones((2, 2)), rate=0.5, seed=42, noise_shape=(2, 1), training=True)
    assert res.shape == (2, 2)


def test_np_time_distributed_2(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test the np time distributed behavior.

    Args:
        monkeypatch (pytest.MonkeyPatch): The monkeypatch parameter.

    Returns:
        OpValue: The inferred shape or computed result.
    """
    with pytest.raises(AttributeError):

        def mock_exec(backend: BackendArg, op_name: BackendArg, x: BackendArg, **kwargs: BackendArg) -> BackendArg:
            """Evaluate and process the mock exec operation.

            Args:
                backend (BackendArg): Required parameter for backend.
                op_name (BackendArg): Required parameter for op_name.
                x (BackendArg): Required parameter for x.
                **kwargs (BackendArg): Arbitrary keyword arguments.

            Returns:
                BackendArg: The evaluated or processed output.
            """
            return x

        monkeypatch.setattr(eager_mod, "execute_generic_op", mock_exec)
        res = _np_time_distributed(np, np.ones((2, 2)), wrapped_op_name="Dummy")
        assert res.shape == (2, 2)
        res = _np_time_distributed(np, np.ones((2, 3, 4)), wrapped_op_name="Dummy")
        assert res.shape == (2, 3, 4)


def test_nn_ops() -> None:
    """Test neural network operations in nn.

    Returns:
        None
    """
    # Gelu
    x = np.array([-1.0, 0.0, 1.0], dtype=np.float32)
    gelu_res = nn_mod._gelu(x)
    assert gelu_res.shape == (3,)

    # Relu
    relu_res = nn_mod._np_relu(np, x)
    assert np.array_equal(relu_res, [0.0, 0.0, 1.0])
    assert np.array_equal(require_callable(numpy_eager_registry.get)("Relu")(np, x), [0.0, 0.0, 1.0])

    # AlphaDropout: not training, rate == 0, training with custom noise_shape and default noise_shape
    assert np.array_equal(nn_mod._np_alpha_dropout(np, x, training=False), x)
    assert np.array_equal(nn_mod._np_alpha_dropout(np, x, rate=0.0, training=True), x)
    ad_train1 = nn_mod._np_alpha_dropout(np, x, rate=0.2, training=True, seed=42)
    assert ad_train1.shape == (3,)
    ad_train2 = nn_mod._np_alpha_dropout(np, x, rate=0.2, training=True, seed=42, noise_shape=(3,))
    assert ad_train2.shape == (3,)
    assert require_callable(numpy_eager_registry.get)("AlphaDropout")(np, x, training=False) is not None

    # ActivityRegularization
    assert np.array_equal(nn_mod._np_activity_regularization(np, x), x)
    assert require_callable(numpy_eager_registry.get)("ActivityRegularization")(np, x) is not None

    # Dropout: not training, rate == 0, training with custom noise_shape and default noise_shape
    assert np.array_equal(nn_mod._np_dropout(np, x, training=False), x)
    assert np.array_equal(nn_mod._np_dropout(np, x, rate=0.0, training=True), x)
    do_train1 = nn_mod._np_dropout(np, x, rate=0.2, training=True, seed=42)
    assert do_train1.shape == (3,)
    do_train2 = nn_mod._np_dropout(np, x, rate=0.2, training=True, seed=42, noise_shape=(3,))
    assert do_train2.shape == (3,)
    assert require_callable(numpy_eager_registry.get)("Dropout")(np, x, training=False) is not None

    # TimeDistributed: len(shape) < 3 vs len(shape) >= 3
    x_2d = np.array([[-1.0, 2.0], [3.0, -4.0]])
    td_2d = nn_mod._np_time_distributed(np, x_2d, wrapped_op_name="Relu")
    assert np.array_equal(td_2d, [[0.0, 2.0], [3.0, 0.0]])

    x_3d = np.array([[[-1.0, 2.0]], [[3.0, -4.0]]])
    td_3d = nn_mod._np_time_distributed(np, x_3d, wrapped_op_name="Relu")
    assert np.array_equal(td_3d, [[[0.0, 2.0]], [[3.0, 0.0]]])
    assert require_callable(numpy_eager_registry.get)("TimeDistributed")(np, x_2d, wrapped_op_name="Relu") is not None

    # Rope
    rope_input = np.ones((2, 4, 8), dtype=np.float32)
    rope_res1 = nn_mod._np_rope(np, rope_input)
    assert rope_res1.shape == (2, 4, 8)
    rope_res2 = nn_mod._np_rope(np, rope_input, dim=8, offset=2, base=5000.0)
    assert rope_res2.shape == (2, 4, 8)
    assert require_callable(numpy_eager_registry.get)("Rope")(np, rope_input).shape == (2, 4, 8)

    # Rrelu: not training vs training
    rrelu_notrain = nn_mod._np_rrelu(np, x, training=False)
    assert rrelu_notrain.shape == (3,)
    rrelu_train = nn_mod._np_rrelu(np, x, training=True)
    assert rrelu_train.shape == (3,)
    assert require_callable(numpy_eager_registry.get)("Rrelu")(np, x, training=False).shape == (3,)
