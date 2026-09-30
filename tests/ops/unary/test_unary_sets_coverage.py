"""Tests for test_unary_sets_coverage."""

from __future__ import annotations

import ml_switcheroo_compiler.ops.unary.sets as unary_sets


class _MockDimWithId:
    """Mock dimension exposing an id attribute."""

    def __init__(self, dim_id: str) -> None:
        """Initialize mock dimension.

        Args:
            dim_id (str): Identifier for dimension.
        """
        self.id = dim_id


class _MockShapeContainer:
    """Mock container exposing a shape attribute."""

    def __init__(self, shape: tuple[int, ...]) -> None:
        """Initialize mock shape container.

        Args:
            shape (tuple[int, ...]): Shape tuple.
        """
        self.shape = shape


class _MockShapeMetaContainer:
    """Mock container exposing a shape_metadata attribute."""

    def __init__(self, shape_metadata: tuple[int, ...]) -> None:
        """Initialize mock shape metadata container.

        Args:
            shape_metadata (tuple[int, ...]): Shape metadata tuple.
        """
        self.shape_metadata = shape_metadata


class _MockGenerator:
    """Mock generator for variable AST visitor."""

    def get_fallback_prefix(self) -> str:
        """Get fallback prefix for AST emissions.

        Returns:
            str: Fallback prefix string.
        """
        return "mock_backend"


def test_unary_sets_exhaustive() -> None:
    """Verify shape inference across all 7 set operators in ops/unary/sets.py."""
    set_ops = [
        unary_sets.Setdiff1d,
        unary_sets.Setxor1d,
        unary_sets.Union1d,
        unary_sets.UniqueAll,
        unary_sets.UniqueCounts,
        unary_sets.UniqueInverse,
        unary_sets.UniqueValues,
    ]

    for op_cls in set_ops:
        inst = op_cls()
        # 1. Empty args
        assert inst.infer_shape() == ()

        # 2. Tuple and list inputs
        assert inst.infer_shape((10, 20)) == (10, 20)
        assert inst.infer_shape([30, 40]) == (30, 40)

        # 3. Shape container objects
        assert inst.infer_shape(_MockShapeContainer((5,))) == (5,)
        assert inst.infer_shape(_MockShapeMetaContainer((7, 8))) == (7, 8)

        # 4. Unmatched object returns empty tuple
        assert inst.infer_shape(42) == ()
