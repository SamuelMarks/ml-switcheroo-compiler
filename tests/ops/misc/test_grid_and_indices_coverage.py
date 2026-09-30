"""Tests for test_grid_and_indices_coverage."""

from __future__ import annotations

from unittest.mock import patch

import ml_switcheroo_compiler.ops.info_and_histograms.grid_and_indices as grid_mod


class DummyWithShape:
    """Mock operand providing a shape attribute."""

    def __init__(self, shape: tuple[int, ...]) -> None:
        """Initialize mock with specific shape.

        Args:
            shape (tuple[int, ...]): Target shape.
        """
        self.shape: tuple[int, ...] = shape


class DummyWithoutShape:
    """Mock operand lacking a shape attribute."""

    def __init__(self) -> None:
        """Initialize mock without shape."""
        self.val: int = 42


def test_grid_and_indices_coverage() -> None:
    """Verify 100% line and branch coverage for grid_and_indices operations."""
    # Indices infer_shape
    indices_op = grid_mod.Indices()
    assert indices_op.infer_shape() == ()
    assert indices_op.infer_shape((2, 3)) == (2, 2, 3)
    assert indices_op.infer_shape(42) == ()

    # Ix infer_shape
    ix_op = grid_mod.Ix()
    assert ix_op.infer_shape() == ()
    assert ix_op.infer_shape(DummyWithShape((5,)), DummyWithShape((3,))) == (5, 1)
    assert ix_op.infer_shape(DummyWithoutShape(), DummyWithShape((3,))) == (1, 1)

    # MaskIndices infer_shape
    assert grid_mod.MaskIndices().infer_shape() == (None,)

    # Mgrid and Ogrid infer_shape
    assert grid_mod.Mgrid().infer_shape(shape=(3, 3)) == (3, 3)
    assert grid_mod.Ogrid().infer_shape() == ()

    # R infer_shape
    assert grid_mod.R().infer_shape() == (None,)

    # Module wrappers dispatch
    with patch("ml_switcheroo_compiler.ops.dispatcher.dispatch_op", side_effect=lambda op, *args, **kwargs: f"dispatched_{op}"):
        assert grid_mod.mgrid() == "dispatched_Mgrid"
        assert grid_mod.ogrid() == "dispatched_Ogrid"
        assert grid_mod.r_() == "dispatched_R"
        assert grid_mod.indices((2, 2)) == "dispatched_Indices"
        assert grid_mod.ix_([1], [2]) == "dispatched_Ix"
        assert grid_mod.mask_indices(3) == "dispatched_MaskIndices"
