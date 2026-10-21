"""Test module."""

from __future__ import annotations

import typing

from ml_switcheroo_compiler.backends.common.mixins.image import ImageASTVisitor
from tests.testing_utils import as_callable

if typing.TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import EagerValue


class DummyGenerator:
    """Test class DummyGenerator."""

    def get_fallback_prefix(self) -> EagerValue:
        """Test function get_fallback_prefix."""
        return "bk"


class DummyVisitor(ImageASTVisitor):
    """Test class DummyVisitor."""

    def __init__(self) -> None:
        """Test function __init__."""
        setattr(self, "_generator", DummyGenerator())


class DummyNode:
    """Test class DummyNode."""

    pass


def test_image_mixin() -> None:
    """Test function test_image_mixin."""
    vis = DummyVisitor()
    node = DummyNode()

    assert as_callable(vis.visit_AdjustBrightness)(node, ["a"], delta=0.5) == "bk_adjust_brightness(a, 0.5)"
    assert as_callable(vis.visit_AdjustBrightness)(node, ["a"]) == "bk_adjust_brightness(a, 0.0)"

    assert as_callable(vis.visit_AdjustContrast)(node, ["a"], contrast_factor=2.0) == "bk_adjust_contrast(a, 2.0)"
    assert as_callable(vis.visit_AdjustContrast)(node, ["a"]) == "bk_adjust_contrast(a, 1.0)"

    assert as_callable(vis.visit_AdjustHue)(node, ["a"], delta=0.2) == "bk_adjust_hue(a, 0.2)"
    assert as_callable(vis.visit_AdjustHue)(node, ["a"]) == "bk_adjust_hue(a, 0.0)"

    assert as_callable(vis.visit_AdjustSaturation)(node, ["a"], saturation_factor=1.5) == "bk_adjust_saturation(a, 1.5)"
    assert as_callable(vis.visit_AdjustSaturation)(node, ["a"]) == "bk_adjust_saturation(a, 1.0)"

    assert as_callable(vis.visit_AffineGenerator)(node, ["a"], batch_size=2) == "bk_affine_generator(2, a)"
    assert as_callable(vis.visit_AffineGenerator)(node, ["a"]) == "bk_affine_generator(1, a)"

    assert as_callable(vis.visit_AffineGrid)(node, ["a"], size=(1, 2), align_corners=True) == "bk_affine_grid(a, size=(1, 2), align_corners=True)"
    assert as_callable(vis.visit_AffineGrid)(node, ["a"]) == "bk_affine_grid(a, size=(), align_corners=False)"

    assert as_callable(vis.visit_AffineTransform)(node, ["a", "b"], interpolation="bilinear") == "bk_affine_transform(a, b, interpolation='bilinear')"
    assert as_callable(vis.visit_AffineTransform)(node, ["a", "b"]) == "bk_affine_transform(a, b, interpolation='nearest')"

    assert as_callable(vis.visit_AugMix)(node, ["a"]) == "bk_augmix(a)"
    assert as_callable(vis.visit_AutoContrast)(node, ["a"]) == "bk_auto_contrast(a)"
