"""Tests for image operations alias."""

from ml_switcheroo_compiler.ops.image import resize


def test_image_alias():
    """Test image operations alias."""
    from ml_switcheroo_compiler.ops.vision import resize_bilinear

    assert resize is resize_bilinear

    import ml_switcheroo_compiler.ops.image as image_ops

    assert hasattr(image_ops, "crop")
    assert hasattr(image_ops, "pad_to_bounding_box")
    assert hasattr(image_ops, "resize")
    assert hasattr(image_ops, "resize_bicubic")
