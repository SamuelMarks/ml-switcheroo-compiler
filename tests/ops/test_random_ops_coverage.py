"""Test random ops."""

from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.ops.random_ops import binomial, categorical, choice, dirichlet, permutation, truncated_normal


def test_random_ops_infer_shape():
    """Test shape inference."""
    t = Tensor(None, TensorConfig((2, 2), "float32", "cpu"))
    assert categorical().infer_shape(t) == (2, 2)
    assert dirichlet().infer_shape(t) == (2, 2)
    assert binomial().infer_shape(t) == (2, 2)
    assert truncated_normal().infer_shape(t) == (2, 2)
    assert permutation().infer_shape(t) == (2, 2)
    assert choice().infer_shape(t) == (2, 2)
