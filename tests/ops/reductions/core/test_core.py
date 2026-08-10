from ml_switcheroo_compiler.ops.reductions.core import ReductionOp


def test_reduction_op_call(mocker):
    op = ReductionOp()
    op.op_type = "MockReduction"

    mock_dispatch = mocker.patch("ml_switcheroo_compiler.ops.dispatcher.dispatch_op", return_value="result")

    # Test passing arguments
    res = op(axis=0)
    assert res == "result"
    mock_dispatch.assert_called_with("MockReduction", axis=0)

    res = op(keepdims=True)
    assert res == "result"
    mock_dispatch.assert_called_with("MockReduction", keepdims=True)

    res = op(axis=1, keepdims=False)
    assert res == "result"
    mock_dispatch.assert_called_with("MockReduction", axis=1, keepdims=False)


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
