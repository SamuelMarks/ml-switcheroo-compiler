from ml_switcheroo_compiler.ops.reductions.core import ReductionOp


def test_reduction_op_call(mocker):
    op = ReductionOp()
    op.op_type = "MockReduction"

    mock_dispatch = mocker.patch("ml_switcheroo_compiler.ops.dispatcher.dispatch_op", return_value="result")

    # Test dim -> axis conversion
    res = op(dim=0)
    assert res == "result"
    mock_dispatch.assert_called_with("MockReduction", axis=0)

    # Test keepdim -> keepdims conversion
    res = op(keepdim=True)
    assert res == "result"
    mock_dispatch.assert_called_with("MockReduction", keepdims=True)

    # Test both
    res = op(dim=1, keepdim=False)
    assert res == "result"
    mock_dispatch.assert_called_with("MockReduction", axis=1, keepdims=False)

    # Test axis and keepdims existing
    res = op(axis=2, keepdims=True, dim=1, keepdim=False)
    assert res == "result"
    # Wait, if axis is in kwargs, dim is not popped but it's ignored in the override
    # let's see: `if "dim" in kwargs and "axis" not in kwargs:`
    # If "axis" is in kwargs, dim is just passed through!
    mock_dispatch.assert_called_with("MockReduction", axis=2, keepdims=True, dim=1, keepdim=False)


def test_reduction_op_infer_shape():
    op = ReductionOp()
    assert op.infer_shape() == ()


def test_reduction_op_format_args():
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
