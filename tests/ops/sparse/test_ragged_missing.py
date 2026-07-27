from ml_switcheroo_compiler.ops.ragged.core import RaggedDot


def test_ragged_missing():
    assert RaggedDot().infer_shape() == ()
