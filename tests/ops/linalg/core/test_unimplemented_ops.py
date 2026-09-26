import ml_switcheroo_compiler.ops.linalg as mod


def test_linalg_dummy_ops_infer_shape() -> None:
    class MockTensor:
        def __init__(self, shape: tuple[int, ...]) -> None:
            self.shape = shape

    t1 = MockTensor((2, 3))
    t2 = MockTensor((2, 3))
    mat = MockTensor((2, 3, 3))
    vec = MockTensor((2, 3))

    # Test infer_shape with args
    assert mod.Vecdot().infer_shape(t1, t2) == (2,)
    assert mod.CustomLinearSolve().infer_shape(mat, vec) == (2, 3)
    assert mod.CustomRoot().infer_shape(None, t1) == (2, 3)

    # Test infer_shape without args (the missing branches)
    assert mod.Vecdot().infer_shape() == ()
    assert mod.CustomLinearSolve().infer_shape() == ()
    assert mod.CustomRoot().infer_shape() == ()
