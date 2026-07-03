"""Tests for image operations alias."""

import ml_switcheroo_compiler.ops.image as image_ops


def test_image_alias() -> object:
    """Test image operations alias."""
    assert hasattr(image_ops, "crop")
    assert hasattr(image_ops, "pad_to_bounding_box")
    assert hasattr(image_ops, "resize_bicubic")
