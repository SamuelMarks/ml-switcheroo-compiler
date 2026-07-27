from ml_switcheroo_compiler.ops.raw_ops import RawConv2D, RawMatMul, RawMerge, RawOp, RawSwitch


def test_raw_op_infer_shape():
    op = RawOp()
    assert op.infer_shape() == ()
    assert op.infer_shape(1, 2, a=3) == ()


def test_raw_op_registry():
    assert RawSwitch.op_name == "RawSwitch"
    assert RawMerge.op_name == "RawMerge"
    assert RawConv2D.op_name == "RawConv2D"
    assert RawMatMul.op_name == "RawMatMul"
