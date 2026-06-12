"""Unit tests for basic reduction operations.

This module verifies the correctness of reduction operations such as Sum, Mean, Max, and
Min by comparing their shape inference and evaluation results against their equivalent
NumPy implementations.
"""

import numpy as np

from ml_switcheroo.ops.reductions.basic import (
    Max,
    Mean,
    Min,
    Sum,
)


def test_reduction_ops() -> None:
    """Tests the correctness of basic reduction operations against NumPy equivalents.

    This test validates that the custom reduction operations (Sum, Mean, Max, Min)
    correctly infer output shapes and produce identical numerical results to their
    corresponding NumPy functions (np.sum, np.mean, np.max, np.min) under
    various configurations of axis and keepdims

    Returns:
    None
    """
    x = np.array([[1.0, 2.0], [3.0, 4.0]])

    ops = [
        (Sum(), np.sum),
        (Mean(), np.mean),
        (Max(), np.max),
        (Min(), np.min),
    ]

    for op, np_func in ops:
        assert op.infer_shape(x.shape) == ()
        assert np.allclose(op.numpy_eval(x), np_func(x))
        assert np.allclose(op.numpy_eval(x, axis=0), np_func(x, axis=0))
        assert np.allclose(
            op.numpy_eval(x, axis=1, keepdims=True),
            np_func(x, axis=1, keepdims=True),
        )
