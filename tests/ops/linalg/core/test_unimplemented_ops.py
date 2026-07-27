import ml_switcheroo_compiler.ops.linalg as mod


def test_linalg_dummy_ops_infer_shape() -> None:
    # Test infer_shape with args
    assert mod.Vecdot().infer_shape("shape1") == "shape1"
    assert mod.CustomLinearSolve().infer_shape("shape2") == "shape2"
    assert mod.CustomRoot().infer_shape("shape3") == "shape3"

    # Test infer_shape without args (the missing branches)
    assert mod.Vecdot().infer_shape() == ()
    assert mod.CustomLinearSolve().infer_shape() == ()
    assert mod.CustomRoot().infer_shape() == ()
