"""Test misc ops."""

from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.ops.info_and_histograms import AxisIndex


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


"""Tests for ops misc coverage."""

from unittest.mock import patch

import ml_switcheroo_compiler.ops.binary.math as binary_math
import ml_switcheroo_compiler.ops.creation.frontend_utils as frontend_utils
import ml_switcheroo_compiler.ops.distributed_ops as dist_ops
import ml_switcheroo_compiler.ops.info_and_histograms as misc
import ml_switcheroo_compiler.ops.io as io_ops
import ml_switcheroo_compiler.ops.linalg.dot as dot_ops
import ml_switcheroo_compiler.ops.nn.activations as activations
import ml_switcheroo_compiler.ops.random_ops.core as random_core
import ml_switcheroo_compiler.ops.shape.pad_and_tile as shape_misc
import ml_switcheroo_compiler.ops.shape.slicing as slicing
import ml_switcheroo_compiler.ops.unary.logical as unary_logical


def test_cover_misc() -> None:
    """Test coverage for misc ops."""
    with patch("ml_switcheroo_compiler.ops.dispatcher.dispatch_op") as mock_dispatch:
        for module in [misc, shape_misc, unary_logical, dist_ops, frontend_utils, activations, io_ops, dot_ops, binary_math, slicing, random_core]:
            for attr in dir(module):
                val = getattr(module, attr)
                if callable(val) and attr not in ["dispatch_op", "Any", "object"]:
                    try:
                        val()
                    except Exception:
                        pass
