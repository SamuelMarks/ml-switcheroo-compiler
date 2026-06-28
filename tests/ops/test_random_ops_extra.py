from ml_switcheroo_compiler.ops.random_ops import RngBitGenerator, RngUniform


def test_random_ops_infer_shape_coverage():
    res1 = RngBitGenerator().infer_shape(None, (1, 2), None)
    assert res1 == (1, 2)

    res2 = RngUniform().infer_shape(None, None, (3, 4), None)
    assert res2 == (3, 4)
