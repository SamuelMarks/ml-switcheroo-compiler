"""Exhaustive parity tests for Keras 3 stateless operations and control flow."""

from __future__ import annotations

import sys
import typing
from unittest.mock import MagicMock, patch

from ml_switcheroo_compiler.backends.keras.eager import execute_op
from tests.testing_utils import ensure_callable, ensure_eager_tensor

if typing.TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import EagerValue


def _create_mock_keras() -> MagicMock:
    """Create a fully configured mock of Keras 3 with keras.ops.

    Returns:
        MagicMock: Mocked Keras 3 module.
    """
    mock_keras = MagicMock()
    mock_ops = MagicMock()
    mock_keras.ops = mock_ops

    mock_ops.convert_to_tensor.side_effect = lambda x: x
    mock_ops.add.return_value = "keras_add"
    mock_ops.multiply.return_value = "keras_mul"
    mock_ops.sum.return_value = "keras_sum"
    mock_ops.mean.return_value = "keras_mean"
    mock_ops.argmax.return_value = "keras_argmax"
    mock_ops.argmin.return_value = "keras_argmin"
    mock_ops.cumsum.return_value = "keras_cumsum"
    mock_ops.cumprod.return_value = "keras_cumprod"
    mock_ops.cond.return_value = "keras_cond"
    mock_ops.while_loop.return_value = "keras_while_loop"
    mock_ops.matmul.return_value = "keras_matmul"
    mock_ops.cholesky.return_value = "keras_cholesky"
    mock_ops.qr.return_value = ("keras_q", "keras_r")
    mock_ops.svd.return_value = ("keras_u", "keras_s", "keras_v")
    mock_ops.solve.return_value = "keras_solve"
    mock_ops.conv.return_value = "keras_conv2d"
    mock_ops.max_pool.return_value = "keras_maxpool"
    mock_ops.average_pool.return_value = "keras_avgpool"
    mock_ops.norm.return_value = "keras_norm"

    mock_image = MagicMock()
    mock_image.resize.return_value = "keras_resize"
    mock_ops.image = mock_image

    return mock_keras


def test_keras_ops_parity() -> None:
    """Test all Unified IR math, reduction, and linalg ops mapped to keras.ops."""
    mock_keras = _create_mock_keras()
    with patch.dict(sys.modules, {"keras": mock_keras}):
        # Math
        assert ensure_callable(execute_op)(None, "Add", [1.0], [2.0]) == "keras_add"
        assert ensure_callable(execute_op)(None, "Multiply", [2.0], [3.0]) == "keras_mul"

        # Reductions
        assert ensure_callable(execute_op)(None, "Sum", [1.0, 2.0], dim=0) == "keras_sum"
        assert ensure_callable(execute_op)(None, "Mean", [1.0, 2.0], dim=0) == "keras_mean"
        assert ensure_callable(execute_op)(None, "ArgMax", [1.0, 2.0], dim=0) == "keras_argmax"
        assert ensure_callable(execute_op)(None, "ArgMin", [1.0, 2.0], dim=0) == "keras_argmin"
        assert ensure_callable(execute_op)(None, "CumSum", [1.0, 2.0], dim=0) == "keras_cumsum"
        assert ensure_callable(execute_op)(None, "CumProd", [1.0, 2.0], dim=0) == "keras_cumprod"
        assert ensure_callable(execute_op)(None, "ReduceL2", [1.0, 2.0]) == "keras_norm"

        # Linear algebra
        assert ensure_callable(execute_op)(None, "MatMul", [[1.0]], [[2.0]]) == "keras_matmul"
        assert ensure_callable(execute_op)(None, "Cholesky", [[1.0]]) == "keras_cholesky"
        assert ensure_callable(execute_op)(None, "QR", [[1.0]]) == ("keras_q", "keras_r")
        assert ensure_callable(execute_op)(None, "SVD", [[1.0]]) == ("keras_u", "keras_s", "keras_v")
        assert ensure_callable(execute_op)(None, "Solve", [[1.0]], [[2.0]]) == "keras_solve"


def test_keras_control_flow_and_vision_parity() -> None:
    """Test Keras 3 control flow (cond, while_loop) and vision operators."""
    mock_keras = _create_mock_keras()
    with patch.dict(sys.modules, {"keras": mock_keras}):
        # Control Flow
        pred = True

        def true_fn() -> EagerValue:
            """Return true branch constant.

            Returns:
                EagerValue: Computed output value from `true_fn`.
            """
            return 1

        def false_fn() -> EagerValue:
            """Return false branch constant.

            Returns:
                EagerValue: Computed output value from `false_fn`.
            """
            return 0

        assert ensure_callable(execute_op)(None, "Cond", pred, true_fn, false_fn) == "keras_cond"
        assert ensure_callable(execute_op)(None, "If", pred, true_fn, false_fn) == "keras_cond"

        def cond_fn(i: EagerValue) -> EagerValue:
            """Evaluate loop continuation condition.

            Args:
                i: Test parameter `i`.

            Returns:
                EagerValue: Computed output value from `cond_fn`.
            """
            return ensure_eager_tensor(i) < ensure_eager_tensor(10)

        def body_fn(i: EagerValue) -> EagerValue:
            """Increment loop counter.

            Args:
                i: Test parameter `i`.

            Returns:
                EagerValue: Computed output value from `body_fn`.
            """
            return ensure_eager_tensor(i) + ensure_eager_tensor(1)

        assert ensure_callable(execute_op)(None, "WhileLoop", cond_fn, body_fn, [0]) == "keras_while_loop"
        assert ensure_callable(execute_op)(None, "While", cond_fn, body_fn, [0]) == "keras_while_loop"

        # Vision and NN
        assert ensure_callable(execute_op)(None, "Conv2D", [[[[1.0]]]], [[[[1.0]]]]) == "keras_conv2d"
        assert ensure_callable(execute_op)(None, "MaxPool2D", [[[[1.0]]]]) == "keras_maxpool"
        assert ensure_callable(execute_op)(None, "AvgPool2D", [[[[1.0]]]]) == "keras_avgpool"
        assert ensure_callable(execute_op)(None, "ResizeBilinear", [[[[1.0]]]], size=(2, 2)) == "keras_resize"
