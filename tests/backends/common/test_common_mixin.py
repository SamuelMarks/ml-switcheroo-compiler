"""Test module."""

from __future__ import annotations

from unittest.mock import MagicMock

from ml_switcheroo_compiler.backends.common.mixins.common import CommonASTVisitor
from ml_switcheroo_compiler.backends.eager_registry import BackendArg


def test_common_ast_visitor() -> None:
    """Test function test_common_ast_visitor."""
    vis = CommonASTVisitor()
    assert vis.generator is vis
    assert vis.get_fallback_prefix() == ""

    class DummyGenerator:
        """Test class DummyGenerator."""

        indent_level: int = 0
        var_names: dict[str, str]
        _cholesky_vjp_imported: bool = False
        _lu_vjp_imported: bool = False
        _qr_vjp_imported: bool = False
        _svd_vjp_imported: bool = False

        def __init__(self, graph: BackendArg = None) -> None:
            """Initialize dummy generator.

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
            """Return fallback prefix.

            Returns:
                str: Fallback prefix string.
            """
            return ""

        def add_line(self, line: str) -> None:
            """Append a line of generated code.

            Args:
                line: Code line to append.
            """
            self._code.append(line)

    gen = DummyGenerator()
    vis2 = CommonASTVisitor(generator=gen)
    assert id(vis2.generator) == id(gen)


def test_common_ast_visitor_mixin() -> None:
    """Verify CommonASTVisitor delegation and fallback prefix."""
    visitor_self = CommonASTVisitor()
    assert visitor_self.generator is visitor_self
    assert visitor_self.get_fallback_prefix() == ""

    mock_gen = MagicMock()
    visitor_gen = CommonASTVisitor(generator=mock_gen)
    assert visitor_gen.generator is mock_gen
