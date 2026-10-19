"""Test module."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from ml_switcheroo_compiler.backends.common.mixins.common import CommonASTVisitor

if TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import EagerCallableProtocol, EagerTensorProtocol


def test_common_ast_visitor() -> None:
    """Test function test_common_ast_visitor."""
    vis = CommonASTVisitor()
    assert cast("EagerTensorProtocol", vis.generator) is cast("EagerTensorProtocol", vis)
    assert vis.get_fallback_prefix() == ""

    class DummyGenerator:
        """Test class DummyGenerator."""

        pass

    gen = DummyGenerator()
    vis2 = cast("EagerCallableProtocol", CommonASTVisitor)(generator=gen)
    assert id(vis2.generator) == id(gen)
