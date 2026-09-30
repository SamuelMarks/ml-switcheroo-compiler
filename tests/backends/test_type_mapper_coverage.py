"""Tests for test_type_mapper_coverage."""

from __future__ import annotations

from ml_switcheroo_compiler.backends.type_mapper import TypeMapper


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


def test_type_mapper_exhaustive() -> None:
    """Verify TypeMapper initialization and mapping fallback branches."""
    # 1. Default initialization
    mapper_default = TypeMapper()
    assert mapper_default.type_dict == {}
    assert mapper_default.map_type("float32") == "float32"

    # 2. Custom dictionary mapping
    custom_types = {"float32": "float", "int64": "long long"}
    mapper_custom = TypeMapper(custom_types)
    assert mapper_custom.map_type("float32") == "float"
    assert mapper_custom.map_type("int64") == "long long"
    assert mapper_custom.map_type("bool") == "bool"
