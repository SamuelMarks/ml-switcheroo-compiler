"""Tests for test_unary_base_coverage."""

from __future__ import annotations

import ml_switcheroo_compiler.ops.unary.base as unary_base


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


def test_unary_base_exhaustive() -> None:
    """Verify UnaryMathOp shape inference across empty, tuple, list, shape attr, and scalar inputs."""
    op = unary_base.UnaryMathOp()

    # 1. Empty shapes
    assert op.infer_shape() == ()

    # 2. Tuple and list inputs
    assert op.infer_shape((2, 3)) == (2, 3)
    assert op.infer_shape([4, 5]) == (4, 5)

    # 3. Object with shape attribute
    assert op.infer_shape(_MockShapeContainer((6, 7))) == (6, 7)

    # 4. Object with shape_metadata attribute
    assert op.infer_shape(_MockShapeMetaContainer((8, 9))) == (8, 9)

    # 5. Scalar input without shape attributes
    assert op.infer_shape(123) == ()
