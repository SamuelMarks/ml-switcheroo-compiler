"""Tests for ReductionOp base class in ops.reductions.core."""

from __future__ import annotations

import typing

from pytest_mock import MockerFixture

from ml_switcheroo_compiler.ops.reductions.core import ReductionOp
from tests.testing_utils import as_tensor

if typing.TYPE_CHECKING:
    pass


def test_reduction_op_call(mocker: MockerFixture) -> None:
    """Test calling ReductionOp dispatches operation to dispatch_op.

    Args:
        mocker (MockerFixture): Pytest mock fixture.

    Returns:
        None.
    """
    op = ReductionOp()
    op.op_type = "MockReduction"

    mock_dispatch = mocker.patch("ml_switcheroo_compiler.ops.dispatcher.dispatch_op", return_value="result")

    # Test passing arguments
    res = op(axis=0)
    assert as_tensor(res) == as_tensor("result")
    mock_dispatch.assert_called_with("MockReduction", axis=0)

    res = op(keepdims=True)
    assert as_tensor(res) == as_tensor("result")
    mock_dispatch.assert_called_with("MockReduction", keepdims=True)

    res = op(axis=1, keepdims=False)
    assert as_tensor(res) == as_tensor("result")
    mock_dispatch.assert_called_with("MockReduction", axis=1, keepdims=False)


def test_reduction_op_infer_shape() -> None:
    """Test infer_shape on ReductionOp across empty, scalar, and multidimensional inputs.

    Returns:
        None.
    """
    from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig

    op = ReductionOp()
    assert op.infer_shape() == ()
    t0 = Tensor(0.0, TensorConfig((), "float32", None))
    assert op.infer_shape(t0) == ()
    t2 = Tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], TensorConfig((2, 3), "float32", None))
    assert op.infer_shape(t2) == ()
    assert op.infer_shape(t2, keepdims=True) == (1, 1)
    assert op.infer_shape(t2, axis=0) == (3,)
    assert op.infer_shape(t2, axis=-1, keepdims=True) == (2, 1)
    assert op.infer_shape(t2, axis=(0, 1)) == ()
    assert op.infer_shape(t2, axis="invalid") == (2, 3)


def test_reduction_op_format_args() -> None:
    """Test _format_args on ReductionOp formats axis and keepdims.

    Returns:
        None.
    """
    op = ReductionOp()

    # Only x
    assert op._format_args("x") == "x"

    # axis
    assert op._format_args("x", axis=0) == "x, axis=0"

    # keepdims
    assert op._format_args("x", keepdims=True) == "x, keepdims=True"

    # axis and keepdims
    assert op._format_args("x", axis=1, keepdims=False) == "x, axis=1"
    assert op._format_args("x", axis=1, keepdims=True) == "x, axis=1, keepdims=True"
