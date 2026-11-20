"""Module test_activations.py."""

from __future__ import annotations

import typing

from pytest_mock import MockerFixture

from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.ops.nn.activations import (
    HardSilu,
    HardSwish,
    LogSoftmax,
    OneHot,
    Rrelu,
    Sigmoid,
    Softmax,
    Squareplus,
    isotonic_regression,
    log_softmax,
    one_hot,
    rrelu,
    sigmoid,
    softmax,
)
from ml_switcheroo_compiler.tracing import global_tracing_state
from tests.testing_utils import MockTensorStub, MockTestStub, extract_test_dtype, extract_test_shape, unpack_pair

if typing.TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import BackendArg


def test_isotonic_regression_tracing() -> None:
    """Test isotonic_regression in tracing mode."""
    config.eager_mode = False
    global_tracing_state.start_tracing()
    try:
        y = Tensor(None, TensorConfig((5,), "float32", "cpu"))
        out1, out2 = unpack_pair(isotonic_regression(y))
        assert extract_test_shape(out1) == (5,)
        assert extract_test_shape(out2) == (5,)
        assert extract_test_dtype(out2) == DType.Int32
    finally:
        global_tracing_state.stop_tracing()
        config.eager_mode = True


def test_activations_dispatcher() -> None:
    """test_activations_dispatcher."""
    from unittest.mock import patch

    from ml_switcheroo_compiler.ops.nn.activations import hard_silu, hard_swish, mish, squareplus

    with patch("ml_switcheroo_compiler.ops.dispatcher.dispatch_op") as mock_dispatch:
        mock_dispatch.return_value = "mock_result"

        assert hard_silu(1, kw=2) == "mock_result"
        mock_dispatch.assert_called_with("HardSilu", 1, kw=2)

        assert hard_swish(1, kw=2) == "mock_result"
        mock_dispatch.assert_called_with("HardSwish", 1, kw=2)

        assert mish(1, kw=2) == "mock_result"
        mock_dispatch.assert_called_with("Mish", 1, kw=2)

        assert squareplus(1, kw=2) == "mock_result"
        mock_dispatch.assert_called_with("Squareplus", 1, kw=2)


def test_isotonic_regression_eager() -> None:
    """Test function test_isotonic_regression_eager."""
    from unittest.mock import MagicMock, patch

    from ml_switcheroo_compiler.core.config import config
    from ml_switcheroo_compiler.ops.nn.activations import isotonic_regression

    config.eager_mode = True
    try:
        with patch("ml_switcheroo_compiler.backends.registry.get_active_backend") as mock_backend:
            backend = MagicMock()
            backend.execute_op.return_value = "eager_iso"
            mock_backend.return_value = backend

            res = isotonic_regression("y", sample_weights="w")
            assert res == "eager_iso"
    finally:
        config.eager_mode = False


def test_activations_eager_mode(mocker: MockerFixture) -> None:
    """Verify the behavior and output invariants of test activations eager mode.

    Args:
        mocker: Test parameter `mocker`.
    """
    config.eager_mode = True

    class DummyBackend:
        """Test class DummyBackend."""

        def execute_op(self, op_name: BackendArg, *args: BackendArg, **kwargs: BackendArg) -> BackendArg:
            """Test function execute_op.

            Args:
                op_name: Test parameter `op_name`.
                args: Test parameter `args`.
                kwargs: Test parameter `kwargs`.

            Returns:
                BackendArg: Computed output value from `execute_op`.
            """
            return op_name

    mocker.patch("ml_switcheroo_compiler.backends.registry.get_active_backend", return_value=DummyBackend())

    assert softmax("x", axis=1) == "Softmax"

    config.eager_mode = False


def test_rrelu_non_eager_mode() -> None:
    """Verify the behavior and output invariants of test rrelu non eager mode."""
    config.eager_mode = False

    class DummyNode(MockTensorStub):
        """Test class DummyNode."""

        id = "n"

    res = rrelu(DummyNode())
    assert str(res).startswith("rrelu")


def test_activations_infer_shape_broadcast() -> None:
    """Verify the behavior and output invariants of test activations infer shape broadcast."""

    class DummyTensor(MockTestStub):
        """Test class DummyTensor."""

        def __init__(self, shape: BackendArg) -> None:
            """Test function __init__.

            Args:
                shape: Test parameter `shape`.
            """
            self.shape = shape

    t1 = DummyTensor((2, 3))
    t2 = DummyTensor((2, 1))

    op = Softmax()
    res = op.infer_shape(t1, t2)
    assert res == (2, 3)

    op2 = LogSoftmax()
    res2 = op2.infer_shape(t1, t2)
    assert res2 == (2, 3)

    op3 = Sigmoid()
    res3 = op3.infer_shape(t1, t2)
    assert res3 == (2, 3)

    op4 = Rrelu()
    res4 = op4.infer_shape(t1, t2)
    assert res4 == (2, 3)


def test_onehot_infer_shape() -> None:
    """Verify the behavior and output invariants of test onehot infer shape."""
    op = OneHot()

    class DummyTensor(MockTestStub):
        """Test class DummyTensor."""

        def __init__(self, shape: BackendArg) -> None:
            """Test function __init__.

            Args:
                shape: Test parameter `shape`.
            """
            self.shape = shape

    t1 = DummyTensor((2, 3))

    res = op.infer_shape(t1, 5, axis=1)
    assert res == (2, 5, 3)


def test_activations_non_eager_mode() -> None:
    """Verify the behavior and output invariants of test activations non eager mode."""
    config.eager_mode = False

    class DummyNode(MockTensorStub):
        """Test class DummyNode."""

        id = "n"

        class DummyShape:
            """Test class DummyShape."""

            shape = (1,)

        shape_metadata = DummyShape()

    res = softmax(DummyNode())
    assert str(res).startswith("softmax")

    res = log_softmax(DummyNode())
    assert str(res).startswith("logsoftmax")

    res = sigmoid(DummyNode())
    assert tuple(res.shape) == ()

    res = one_hot(DummyNode(), 5)
    assert str(res).startswith("onehot")

    res = rrelu(DummyNode())
    assert str(res).startswith("rrelu")


def test_activations_infer_shape_missing() -> None:
    """Verify the behavior and output invariants of test activations infer shape missing."""
    # Rrelu missing shapes
    assert Rrelu().infer_shape() == ()

    class DummyTensor(MockTestStub):
        """Test class DummyTensor."""

        def __init__(self, shape: BackendArg) -> None:
            """Test function __init__.

            Args:
                shape: Test parameter `shape`.
            """
            self.shape = shape

    t = DummyTensor((2, 2))

    assert HardSilu().infer_shape(t) == (2, 2)
    assert HardSilu().infer_shape() == ()

    assert HardSwish().infer_shape(t) == (2, 2)
    assert HardSwish().infer_shape() == ()

    assert Squareplus().infer_shape(t) == (2, 2)
    assert Squareplus().infer_shape() == ()
