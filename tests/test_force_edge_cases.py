"""Test suite verifying mathematical and shape edge-cases."""

import numpy as np

from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig


def test_valid_force_edge_cases() -> None:
    """Verify that edge inputs like zero-length arrays and empty shapes are handled correctly."""
    # Verify that a 0-D empty array handles operations correctly
    t1 = Tensor(np.array(0.0), TensorConfig((), DType.Float32, "cpu"))
    assert t1.shape == ()
    assert t1.data == 0.0

    # Verify that a zero-length array handles shape mappings
    t2 = Tensor(np.zeros((0, 5)), TensorConfig((0, 5), DType.Float32, "cpu"))
    assert t2.shape == (0, 5)
    assert t2.data.size == 0
