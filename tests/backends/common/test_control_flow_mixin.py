"""Test module."""

from ml_switcheroo_compiler.backends.common.mixins.control_flow import ControlFlowASTVisitor


class DummyGenerator:
    def _get_backend_prefix(self):
        return "bk"


class DummyVisitor(ControlFlowASTVisitor):
    def __init__(self):
        self._generator = DummyGenerator()


class DummyNode:
    def __init__(self, attrs=None):
        self.attributes = attrs or {}


def test_control_flow_mixin():
    vis = DummyVisitor()
    node = DummyNode()

    assert vis.visit_Scan(node, ["a", "b"]) == "bk_scan(a, b)"
    assert vis.visit_Switch(node, ["a", "b"]) == "bk_switch(a, b)"

    node_td = DummyNode({"wrapped_op_name": "MyOp"})
    assert vis.visit_TimeDistributed(node_td, ["a"]) == "bk_time_distributed(a, 'MyOp')"

    assert vis.visit_Assert(node, ["cond"], data=["msg"]) == "bk_assert(cond, data=['msg'])"
    assert vis.visit_Assert(node, ["cond"]) == "bk_assert(cond, data=['Assertion failed.'])"

    assert vis.visit_AssociativeScan(node, ["a", "b"]) == "bk_associative_scan(a, b)"
