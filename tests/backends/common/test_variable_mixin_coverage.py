"""Tests for test_variable_mixin_coverage."""

from __future__ import annotations

from ml_switcheroo_compiler.backends.common.mixins.variable import VariableASTVisitor
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


def test_variable_ast_visitor_exhaustive() -> None:
    """Verify VariableASTVisitor code generation methods."""
    visitor = VariableASTVisitor(generator=_MockGenerator())

    node = IRNode("node_assign", "Assign")
    assign_str = visitor.visit_Assign(node, ["dest_var", "src_var"])
    assert assign_str == "mock_backend_assign(dest_var, src_var)"

    assign_add_str = visitor.visit_AssignAdd(node, ["accum_var", "delta_var"])
    assert assign_add_str == "mock_backend_assign_add(accum_var, delta_var)"

    assign_sub_str = visitor.visit_AssignSub(node, ["accum_var", "delta_var"])
    assert assign_sub_str == "mock_backend_assign_sub(accum_var, delta_var)"
