"""Test misc ops."""

from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.ops.misc import AxisIndex


def test_axis_index_infer_shape():
    """Test AxisIndex infer_shape fallback."""
    op = AxisIndex()
    # No args
    assert op.infer_shape() == ()
    # Arg without shape
    assert op.infer_shape(1) == ()
    # Arg with shape
    t = Tensor(None, TensorConfig((2, 3), "float32", "cpu"))
    assert op.infer_shape(t) == (2, 3)
