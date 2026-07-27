from ml_switcheroo_compiler.ops.reductions.aggregations import NaryMathOp


class MockTensor:
    def __init__(self, shape):
        self.shape = shape


def test_nary_math_op_infer_shape():
    op = NaryMathOp()
    t1 = MockTensor((2, 3))
    t2 = MockTensor((2, 3))

    # Empty args
    assert op.infer_shape() == ()

    # Empty list
    assert op.infer_shape([]) == ()
    assert op.infer_shape(inputs=[]) == ()

    # Valid list
    assert op.infer_shape([t1, t2]) == (2, 3)
    assert op.infer_shape(inputs=[t1, t2]) == (2, 3)

    # Not a list
    assert op.infer_shape(t1) == ()
