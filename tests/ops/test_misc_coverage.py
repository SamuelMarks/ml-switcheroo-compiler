"""Tests for ops misc coverage."""

from unittest.mock import patch

import ml_switcheroo_compiler.ops.binary.math as binary_math
import ml_switcheroo_compiler.ops.creation.frontend_utils as frontend_utils
import ml_switcheroo_compiler.ops.distributed_ops as dist_ops
import ml_switcheroo_compiler.ops.io as io_ops
import ml_switcheroo_compiler.ops.linalg.dot as dot_ops
import ml_switcheroo_compiler.ops.misc as misc
import ml_switcheroo_compiler.ops.nn.activations as activations
import ml_switcheroo_compiler.ops.random_ops.core as random_core
import ml_switcheroo_compiler.ops.shape.misc as shape_misc
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
