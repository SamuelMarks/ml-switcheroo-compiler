"""Tests for test_backend_utils_coverage."""

from __future__ import annotations

import ml_switcheroo_compiler.backends.backend_utils as backend_utils
from ml_switcheroo_compiler.ir.core import IRNode


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


def test_backend_utils_exhaustive() -> None:
    """Verify input variable resolution and shape metadata formatting across all branches."""
    # 1. resolve_input_vars with both mapped and unmapped variable names
    node = IRNode("node_add", "Add", inputs=["in_0", "in_1", "in_2"])
    mapping = {"in_0": "var_x", "in_1": "var_y"}
    resolved = backend_utils.resolve_input_vars(node, mapping)
    assert resolved == ["var_x", "var_y", "in_2"]

    # 2. format_shape_metadata when shape_metadata is missing or empty
    node_no_meta = IRNode("node_relu", "Relu")
    assert backend_utils.format_shape_metadata(node_no_meta, {}) is None

    node_empty_meta = IRNode("node_relu_empty", "Relu")
    node_empty_meta.shape_metadata = []
    assert backend_utils.format_shape_metadata(node_empty_meta, {}) is None

    # 3. format_shape_metadata with single dimension (trailing comma)
    node_single_dim = IRNode("node_reshape", "Reshape")
    node_single_dim.shape_metadata = [32]
    assert backend_utils.format_shape_metadata(node_single_dim, {}) == "(32,)"

    # 4. format_shape_metadata with multiple dimensions (no trailing comma) and mixed types
    node_multi_dim = IRNode("node_conv", "Conv")
    node_multi_dim.shape_metadata = [
        _MockDimWithId("batch_sym"),
        _MockDimWithId("unmapped_sym"),
        "dynamic_channel",
        64,
    ]
    var_names = {"batch_sym": "N"}
    formatted = backend_utils.format_shape_metadata(node_multi_dim, var_names)
    assert formatted == "(N, unmapped_sym, 'dynamic_channel', 64)"
