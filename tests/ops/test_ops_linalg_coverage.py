"""Provides required module functionality."""

from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.ir.core import TensorSpec
from ml_switcheroo_compiler.ops.linalg.fft_ops import Rfft


def test_rfft_infer_shape_coverage() -> None:
    """Execute the requested function."""
    r = Rfft()

    spec = TensorSpec(shape=(3, 4), dtype=DType.Float32)
    r.infer_shape(spec, n=5, axis=-1)
