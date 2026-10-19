"""Test module."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from ml_switcheroo_compiler.backends.common.mixins.variable import VariableASTVisitor

if TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import EagerCallableProtocol, EagerValue


class DummyGenerator:
    """Test class DummyGenerator."""

    def get_fallback_prefix(self) -> EagerValue:
        """Test function get_fallback_prefix."""
        return "bk"


class DummyVisitor(VariableASTVisitor):
    """Test class DummyVisitor."""

    def __init__(self) -> None:
        """Test function __init__."""
        setattr(self, "_generator", DummyGenerator())


class DummyNode:
    """Test class DummyNode."""

    pass


def test_variable_mixin() -> None:
    """Test function test_variable_mixin."""
    vis = DummyVisitor()
    node = DummyNode()

    assert cast("EagerCallableProtocol", vis.visit_Assign)(node, ["a", "b"]) == "bk_assign(a, b)"
    assert cast("EagerCallableProtocol", vis.visit_AssignAdd)(node, ["a", "b"]) == "bk_assign_add(a, b)"
    assert cast("EagerCallableProtocol", vis.visit_AssignSub)(node, ["a", "b"]) == "bk_assign_sub(a, b)"
