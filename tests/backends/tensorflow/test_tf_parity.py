"""Exhaustive parity tests for TensorFlow backend using tf.math, tf.raw_ops, and tf.nn."""

import sys
from unittest.mock import MagicMock, patch

from ml_switcheroo_compiler.backends.tensorflow.eager import execute_op


def _create_mock_tf() -> MagicMock:
    """Create a fully configured mock of TensorFlow with tf.math and tf.nn.

    Returns:
        MagicMock: Mocked TensorFlow module.
    """
    mock_tf = MagicMock()
    mock_tf.convert_to_tensor.side_effect = lambda x: x

    # Math
    mock_tf.add.return_value = "tf_add"
    mock_math = MagicMock()
    mock_tf.math = mock_math
    mock_math.multiply.return_value = "tf_mul"
    mock_math.reduce_sum.return_value = "tf_sum"
    mock_math.reduce_mean.return_value = "tf_mean"
    mock_math.reduce_euclidean_norm.return_value = "tf_l2norm"
    mock_math.cumsum.return_value = "tf_cumsum"
    mock_math.cumprod.return_value = "tf_cumprod"

    # Reductions / args
    mock_tf.reduce_sum.return_value = "tf_sum"
    mock_tf.reduce_mean.return_value = "tf_mean"
    mock_tf.argmax.return_value = "tf_argmax"
    mock_tf.argmin.return_value = "tf_argmin"

    # Control flow
    mock_tf.cond.return_value = "tf_cond"
    mock_tf.while_loop.return_value = "tf_while_loop"

    # Linalg
    mock_linalg = MagicMock()
    mock_tf.linalg = mock_linalg
    mock_tf.matmul.return_value = "tf_matmul"
    mock_linalg.cholesky.return_value = "tf_cholesky"
    mock_linalg.qr.return_value = ("tf_q", "tf_r")
    mock_linalg.svd.return_value = ("tf_s", "tf_u", "tf_v")
    mock_linalg.solve.return_value = "tf_solve"

    # NN and Vision
    mock_nn = MagicMock()
    mock_tf.nn = mock_nn
    mock_nn.conv2d.return_value = "tf_conv2d"
    mock_nn.max_pool2d.return_value = "tf_maxpool"
    mock_nn.avg_pool2d.return_value = "tf_avgpool"

    mock_image = MagicMock()
    mock_image.resize.return_value = "tf_resize"
    mock_tf.image = mock_image

    return mock_tf


def test_tf_ops_parity() -> None:
    """Test all Unified IR math, reduction, and linalg ops mapped to tf.math and tf.raw_ops."""
    mock_tf = _create_mock_tf()
    with patch.dict(sys.modules, {"tensorflow": mock_tf}):
        # Math
        assert execute_op(None, "Add", [1.0], [2.0]) == "tf_add"
        assert execute_op(None, "Multiply", [2.0], [3.0]) == "tf_mul"

        # Reductions
        assert execute_op(None, "ReduceSum", [1.0, 2.0], dim=0) == "tf_sum"
        assert execute_op(None, "ReduceMean", [1.0, 2.0], dim=0) == "tf_mean"
        assert execute_op(None, "ArgMax", [1.0, 2.0], dim=0) == "tf_argmax"
        assert execute_op(None, "ArgMin", [1.0, 2.0], dim=0) == "tf_argmin"
        assert execute_op(None, "ReduceL2", [1.0, 2.0]) == "tf_l2norm"
        assert execute_op(None, "CumSum", [1.0, 2.0], dim=0) == "tf_cumsum"
        assert execute_op(None, "CumProd", [1.0, 2.0], dim=0) == "tf_cumprod"

        # Linear algebra
        assert execute_op(None, "MatMul", [[1.0]], [[2.0]]) == "tf_matmul"
        assert execute_op(None, "Cholesky", [[1.0]]) == "tf_cholesky"
        assert execute_op(None, "QR", [[1.0]]) == ("tf_q", "tf_r")
        assert execute_op(None, "SVD", [[1.0]]) == ("tf_s", "tf_u", "tf_v")
        assert execute_op(None, "Solve", [[1.0]], [[2.0]]) == "tf_solve"


def test_tf_control_flow_and_vision_parity() -> None:
    """Test TensorFlow control flow (cond, while_loop) and vision operators."""
    mock_tf = _create_mock_tf()
    with patch.dict(sys.modules, {"tensorflow": mock_tf}):
        # Control Flow
        pred = True
        true_fn = lambda: 1
        false_fn = lambda: 0
        assert execute_op(None, "Cond", pred, true_fn, false_fn) == "tf_cond"
        assert execute_op(None, "If", pred, true_fn, false_fn) == "tf_cond"

        cond_fn = lambda i: i < 10
        body_fn = lambda i: i + 1
        assert execute_op(None, "WhileLoop", cond_fn, body_fn, [0]) == "tf_while_loop"
        assert execute_op(None, "While", cond_fn, body_fn, [0]) == "tf_while_loop"

        # Vision and NN
        assert execute_op(None, "Conv2D", [[[[1.0]]]], [[[[1.0]]]]) == "tf_conv2d"
        assert execute_op(None, "MaxPool2D", [[[[1.0]]]]) == "tf_maxpool"
        assert execute_op(None, "AvgPool2D", [[[[1.0]]]]) == "tf_avgpool"
        assert execute_op(None, "ResizeBilinear", [[[[1.0]]]], size=(2, 2)) == "tf_resize"
