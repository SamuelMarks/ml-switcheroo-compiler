"""Test module."""

from __future__ import annotations

import pytest
from ml_switcheroo_ir import IRNode

from ml_switcheroo_compiler.backends.common.mixins.variable import VariableASTVisitor
from ml_switcheroo_compiler.backends.eager_registry import BackendArg
from tests.testing_utils import MockNodeStub


class DummyGenerator:
    """Test class DummyGenerator."""

    def get_fallback_prefix(self) -> BackendArg:
        """Test function get_fallback_prefix.

        Returns:
            BackendArg: Computed output value from `get_fallback_prefix`.
        """
        return "bk"


class DummyVisitor(VariableASTVisitor):
    """Test class DummyVisitor."""

    def __init__(self) -> None:
        """Test function __init__."""
        pytest.MonkeyPatch().setattr(self, "_generator", DummyGenerator(), raising=False)


class DummyNode(MockNodeStub):
    """Test class DummyNode."""

    pass


def test_variable_mixin() -> None:
    """Test function test_variable_mixin."""
    vis = DummyVisitor()
    node = DummyNode()

    assert vis.visit_Assign(node, ["a", "b"]) == "bk_assign(a, b)"
    assert vis.visit_AssignAdd(node, ["a", "b"]) == "bk_assign_add(a, b)"
    assert vis.visit_AssignSub(node, ["a", "b"]) == "bk_assign_sub(a, b)"


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

    indent_level: int = 0
    var_names: dict[str, str]
    _cholesky_vjp_imported: bool = False
    _lu_vjp_imported: bool = False
    _qr_vjp_imported: bool = False
    _svd_vjp_imported: bool = False

    def __init__(self, graph: BackendArg = None) -> None:
        """Initialize mock generator.

        Args:
            graph: Optional computation graph.
        """
        del graph
        self.var_names = {}
        self._code: list[str] = []

    @property
    def code(self) -> list[str]:
        """Return generated code lines.

        Returns:
            list[str]: Generated code lines.
        """
        return self._code

    def get_fallback_prefix(self) -> str:
        """Get fallback prefix for AST emissions.

        Returns:
            str: Fallback prefix string.
        """
        return "mock_backend"

    def add_line(self, line: str) -> None:
        """Append a line of generated code.

        Args:
            line: Code line to append.
        """
        self._code.append(line)


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
