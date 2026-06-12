"""Unit tests for shape manipulation and state operations in the ml_switcheroo_compiler library."""

import numpy as np

from ml_switcheroo.ops.shape.basic import (
    BroadcastTo,
    Reshape,
    Transpose,
)


def test_reshape_op() -> None:
    """Tests the Reshape operation's shape inference and NumPy evaluation.

    Verifies that the Reshape operation correctly infers the target shape
    and matches the behavior of np.reshape during evaluation

    Returns:
    None
    """
    op = Reshape()
    x = np.array([1, 2, 3, 4])
    newshape = (2, 2)

    assert op.infer_shape(x.shape, newshape) == newshape
    assert np.array_equal(op.numpy_eval(x, newshape), np.reshape(x, newshape))





def test_transpose_op() -> None:
    """Tests the Transpose operation's shape inference and NumPy evaluation.

    Verifies that the Transpose operation correctly infers the transposed shape
    with and without specified axes, and matches np.transpose during evaluation

    Returns:
    None
    """
    op = Transpose()
    x = np.random.randn(2, 3)

    assert op.infer_shape(x.shape) is None
    assert op.infer_shape(x.shape, (1, 0)) == (3, 2)
    assert np.array_equal(op.numpy_eval(x), np.transpose(x))
    assert np.array_equal(op.numpy_eval(x, axes=(1, 0)), np.transpose(x, axes=(1, 0)))







def test_broadcast_to_op() -> None:
    """Tests the BroadcastTo operation's shape inference and NumPy evaluation.

    Verifies that the BroadcastTo operation correctly infers the broadcasted shape
    and matches np.broadcast_to during evaluation

    Returns:
    None
    """
    op = BroadcastTo()
    x = np.array([1, 2])
    shape = (2, 2)

    assert op.infer_shape(x.shape, shape) == shape
    assert np.array_equal(op.numpy_eval(x, shape), np.broadcast_to(x, shape))





def test_state_ops() -> None:
    """Tests the shape inference and error handling of state operations.

    Verifies that ReadVariable and AssignVariable correctly infer shapes,
    and raise CompilationError when attempting NumPy evaluation directly

    Returns:
    None
    """
    import pytest

    from ml_switcheroo.core.errors import CompilationError
    from ml_switcheroo.ops.base import get_op

    r = get_op("ReadVariable")()
    a = get_op("AssignVariable")()

    assert r.infer_shape(shape=(2,)) == (2,)
    assert a.infer_shape((2,)) == (2,)

    with pytest.raises(CompilationError):
        r.numpy_eval()

    with pytest.raises(CompilationError):
        a.numpy_eval(1)
