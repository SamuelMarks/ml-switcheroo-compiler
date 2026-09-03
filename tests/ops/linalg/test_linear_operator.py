def test_linear_operator_shape_inference():
    from ml_switcheroo_compiler.ops.linalg.linear_operator import LinearOperator

    assert LinearOperator().infer_shape() == ()

    class MockOperand:
        shape = (1, 2)

    assert LinearOperator().infer_shape(MockOperand()) == (1, 2)
    assert LinearOperator().infer_shape(operand=MockOperand()) == (1, 2)
