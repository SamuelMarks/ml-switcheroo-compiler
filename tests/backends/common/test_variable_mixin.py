"""Test module."""

from ml_switcheroo_compiler.backends.common.mixins.variable import VariableASTVisitor


class DummyGenerator:
    def _get_backend_prefix(self):
        return "bk"


class DummyVisitor(VariableASTVisitor):
    def __init__(self):
        self._generator = DummyGenerator()


class DummyNode:
    pass


def test_variable_mixin():
    vis = DummyVisitor()
    node = DummyNode()

    assert vis.visit_Assign(node, ["a", "b"]) == "bk_assign(a, b)"
    assert vis.visit_AssignAdd(node, ["a", "b"]) == "bk_assign_add(a, b)"
    assert vis.visit_AssignSub(node, ["a", "b"]) == "bk_assign_sub(a, b)"
