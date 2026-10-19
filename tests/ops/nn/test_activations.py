"""Module test_activations.py."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.ops.nn.activations import HardSilu, HardSwish, LogSoftmax, OneHot, Rrelu, Sigmoid, Softmax, Squareplus, isotonic_regression, log_softmax, one_hot, rrelu, sigmoid, softmax
from ml_switcheroo_compiler.tracing import global_tracing_state

if TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import EagerCallableProtocol, EagerTensorProtocol, EagerValue


def test_isotonic_regression_tracing() -> None:
    """Test isotonic_regression in tracing mode."""
    config.eager_mode = False
    global_tracing_state.start_tracing()
    try:
        y = Tensor(None, TensorConfig((5,), "float32", "cpu"))
        out1, out2 = cast("EagerCallableProtocol", isotonic_regression)(y)
        assert cast("EagerTensorProtocol", getattr(out1, "shape")) == (5,)
        assert cast("EagerTensorProtocol", getattr(out2, "shape")) == (5,)
        assert cast("EagerTensorProtocol", getattr(out2, "dtype")) == DType.Int32
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


def test_activations_eager_mode(mocker: EagerTensorProtocol) -> None:
    """Verify the behavior and output invariants of test activations eager mode.

    Args:
        mocker: Test parameter `mocker`.
    """
    config.eager_mode = True

    class DummyBackend:
        """Test class DummyBackend."""

        def execute_op(self, op_name: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
            """Test function execute_op.

            Args:
                op_name: Test parameter `op_name`.
                args: Test parameter `args`.
                kwargs: Test parameter `kwargs`.
            """
            return op_name

    cast("EagerCallableProtocol", mocker.patch)("ml_switcheroo_compiler.backends.registry.get_active_backend", return_value=cast("EagerCallableProtocol", DummyBackend)())

    assert softmax("x", axis=1) == "Softmax"

    config.eager_mode = False


def test_rrelu_non_eager_mode() -> None:
    """Verify the behavior and output invariants of test rrelu non eager mode."""
    config.eager_mode = False

    class DummyNode:
        """Test class DummyNode."""

        id = "n"

    res = cast("EagerCallableProtocol", rrelu)(cast("EagerCallableProtocol", DummyNode)())
    assert res is not None


def test_activations_infer_shape_broadcast() -> None:
    """Verify the behavior and output invariants of test activations infer shape broadcast."""

    class DummyTensor:
        """Test class DummyTensor."""

        def __init__(self, shape: EagerValue) -> None:
            """Test function __init__.

            Args:
                shape: Test parameter `shape`.
            """
            self.shape = shape

    t1 = DummyTensor((2, 3))
    t2 = DummyTensor((2, 1))

    op = Softmax()
    res = cast("EagerCallableProtocol", op.infer_shape)(t1, t2)
    assert res == (2, 3)

    op2 = LogSoftmax()
    res2 = cast("EagerCallableProtocol", op2.infer_shape)(t1, t2)
    assert res2 == (2, 3)

    op3 = Sigmoid()
    res3 = cast("EagerCallableProtocol", op3.infer_shape)(t1, t2)
    assert res3 == (2, 3)

    op4 = Rrelu()
    res4 = cast("EagerCallableProtocol", op4.infer_shape)(t1, t2)
    assert res4 == (2, 3)


def test_onehot_infer_shape() -> None:
    """Verify the behavior and output invariants of test onehot infer shape."""
    op = OneHot()

    class DummyTensor:
        """Test class DummyTensor."""

        def __init__(self, shape: EagerValue) -> None:
            """Test function __init__.

            Args:
                shape: Test parameter `shape`.
            """
            self.shape = shape

    t1 = DummyTensor((2, 3))

    res = cast("EagerCallableProtocol", op.infer_shape)(t1, 5, axis=1)
    assert res == (2, 5, 3)


def test_activations_non_eager_mode() -> None:
    """Verify the behavior and output invariants of test activations non eager mode."""
    config.eager_mode = False

    class DummyNode:
        """Test class DummyNode."""

        id = "n"

        class DummyShape:
            """Test class DummyShape."""

            shape = (1,)

        shape_metadata = DummyShape()

    res = cast("EagerCallableProtocol", softmax)(cast("EagerCallableProtocol", DummyNode)())
    assert res is not None

    res = cast("EagerCallableProtocol", log_softmax)(cast("EagerCallableProtocol", DummyNode)())
    assert res is not None

    res = cast("EagerCallableProtocol", sigmoid)(cast("EagerCallableProtocol", DummyNode)())
    assert res is not None

    res = cast("EagerCallableProtocol", one_hot)(cast("EagerCallableProtocol", DummyNode)(), 5)
    assert res is not None

    res = cast("EagerCallableProtocol", rrelu)(cast("EagerCallableProtocol", DummyNode)())
    assert res is not None


def test_activations_infer_shape_missing() -> None:
    """Verify the behavior and output invariants of test activations infer shape missing."""
    # Rrelu missing shapes
    assert Rrelu().infer_shape() == ()

    class DummyTensor:
        """Test class DummyTensor."""

        def __init__(self, shape: EagerValue) -> None:
            """Test function __init__.

            Args:
                shape: Test parameter `shape`.
            """
            self.shape = shape

    t = DummyTensor((2, 2))

    assert cast("EagerCallableProtocol", cast("EagerCallableProtocol", HardSilu)().infer_shape)(t) == (2, 2)
    assert HardSilu().infer_shape() == ()

    assert cast("EagerCallableProtocol", cast("EagerCallableProtocol", HardSwish)().infer_shape)(t) == (2, 2)
    assert HardSwish().infer_shape() == ()

    assert cast("EagerCallableProtocol", cast("EagerCallableProtocol", Squareplus)().infer_shape)(t) == (2, 2)
    assert Squareplus().infer_shape() == ()
