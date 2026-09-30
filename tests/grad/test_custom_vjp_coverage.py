"""Tests for test_custom_vjp_coverage."""

from __future__ import annotations

from ml_switcheroo_compiler.grad.custom_vjp_ops import CustomVJPFunction


def test_custom_vjp_infer_shape_empty_tensor_args() -> None:
    """Verify CustomVJPFunction._resolve_output_metadata with empty tensor_args."""
    cvjp = CustomVJPFunction(lambda: 42)
    shape, dtype, device = cvjp._resolve_output_metadata([])
    assert shape == ()
    assert dtype == "float32"
    assert device == "cpu"
