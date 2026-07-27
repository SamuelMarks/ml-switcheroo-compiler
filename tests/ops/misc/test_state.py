from ml_switcheroo_compiler.ops.state import AssignVariable, ReadVariable


def test_state_infer_shape():
    op1 = ReadVariable()
    assert op1.infer_shape(shape=(2, 3)) == (2, 3)
    assert op1.infer_shape() == ()

    op2 = AssignVariable()
    assert op2.infer_shape((2, 3)) == (2, 3)
