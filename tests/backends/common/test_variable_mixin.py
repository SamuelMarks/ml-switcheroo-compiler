"""Test module."""

from __future__ import annotations

import typing

from ml_switcheroo_compiler.backends.common.mixins.variable import VariableASTVisitor
from tests.testing_utils import as_callable

if typing.TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import EagerValue


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

    assert as_callable(vis.visit_Assign)(node, ["a", "b"]) == "bk_assign(a, b)"
    assert as_callable(vis.visit_AssignAdd)(node, ["a", "b"]) == "bk_assign_add(a, b)"
    assert as_callable(vis.visit_AssignSub)(node, ["a", "b"]) == "bk_assign_sub(a, b)"
