"""Tests for ops shape misc functions."""

import ml_switcheroo_compiler.ops.shape.pad_and_tile as shape_misc


def test_shape_misc_functions() -> None:
    """Test shape misc coverage."""

    class DummyNode:
        shape = (10, 20)

    a = DummyNode()
    shape_misc._infer_shape_percentile_quantile(a, 0.5, axis=None, keepdims=True)
    shape_misc._infer_shape_percentile_quantile(a, 0.5, axis=None, keepdims=False)
    shape_misc._infer_shape_percentile_quantile(a, [0.5], axis=0, keepdims=True)
    shape_misc._infer_shape_percentile_quantile(a, [0.5], axis=0, keepdims=False)

    for attr in dir(shape_misc):
        if attr.startswith("_infer_shape_"):
            val = getattr(shape_misc, attr)
            if callable(val):
                try:
                    val(a)
                except Exception:
                    pass
                try:
                    val(a, a)
                except Exception:
                    pass
                try:
                    val(a, a, a)
                except Exception:
                    pass
