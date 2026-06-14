"""Provides required module functionality."""

import numpy as np

from ml_switcheroo_compiler.ops.unary.special import Bitcast, Cast


def test_unary_special_coverage_brute() -> None:
    """Execute the requested function."""
    c = Cast()
    bc = Bitcast()

    c.eager_eval(np.array([1, 2]), dtype="float32")
    bc.eager_eval(np.array([1, 2], dtype=np.int32), dtype="float32")
