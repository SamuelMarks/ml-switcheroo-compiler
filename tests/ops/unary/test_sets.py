# ruff: noqa: E501
from ml_switcheroo_compiler.ops.unary.sets import Setdiff1d, Setxor1d, Union1d, UniqueAll, UniqueCounts, UniqueInverse, UniqueValues


def test_sets_infer_shape():
    assert Setdiff1d().infer_shape() == ()
    assert Setxor1d().infer_shape() == ()
    assert Union1d().infer_shape() == ()
    assert Setdiff1d().infer_shape((2, 3)) == (2, 3)
    assert Setxor1d().infer_shape((2, 3)) == (2, 3)
    assert Union1d().infer_shape((2, 3)) == (2, 3)
    assert UniqueAll().infer_shape() == ()
    assert UniqueCounts().infer_shape() == ()
    assert UniqueInverse().infer_shape() == ()
    assert UniqueValues().infer_shape() == ()
