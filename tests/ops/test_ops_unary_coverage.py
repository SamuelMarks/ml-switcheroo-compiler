"""Module docstring."""

from ml_switcheroo_compiler.ops.unary.special import Cast, Bitcast
import numpy as np


def test_unary_special_coverage_brute() -> None:
    """Function docstring."""
    c = Cast()
    bc = Bitcast()

    c.numpy_eval(np.array([1, 2]), dtype="float32")
    bc.numpy_eval(np.array([1, 2], dtype=np.int32), dtype="float32")
