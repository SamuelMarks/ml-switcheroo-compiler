"""Test module."""

from ml_switcheroo_compiler.backends.common.mixins.control_flow import ControlFlowASTVisitor


class DummyGenerator:
    def get_fallback_prefix(self):
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


def test_control_flow_mixin_advanced():
    from ml_switcheroo_compiler.backends.common.mixins.control_flow import ControlFlowASTVisitor
    from ml_switcheroo_compiler.ir.core import IRGraph, LogicalNode

    class DummyGenerator:
        def __init__(self, graph=None):
            self.graph = graph
            self.code = ["def dummy():", "    return 1"]
            self.indent_level = 0
            self.formatter = type("Formatter", (), {"indent_level": 0})()

        def get_fallback_prefix(self):
            return "bk"

        def add_line(self, line):
            self.code.append(line)

    class DummyVisitor(ControlFlowASTVisitor):
        def __init__(self):
            self._generator = DummyGenerator()
            pass

    vis = DummyVisitor()

    # Create dummy IRGraph with dummy nodes
    def make_graph():
        g = IRGraph()
        n = LogicalNode(id="dummy_out", op_type="Identity", inputs=["x"])
        g.nodes = [n]
        return g

    g_cond = make_graph()
    g_body = make_graph()
    g_true = make_graph()
    g_false = make_graph()
    g_f = make_graph()

    # visit_WhileLoop
    node_while = LogicalNode(id="while1", op_type="WhileLoop")
    node_while.attributes = {"cond": g_cond, "body": g_body}

    # We must patch CodeGeneratorVisitor since it attempts to parse the graph
    from unittest.mock import patch

    with patch("ml_switcheroo_compiler.backends.visitor.CodeGeneratorVisitor") as mock_cgv:
        mock_cgv_instance = mock_cgv.return_value
        mock_cgv_instance.generate_body.return_value = None

        res_while = vis.visit_WhileLoop(node_while, ["init_state"])
        assert res_while == "loop_val_while1"

        # visit_Cond
        node_cond = LogicalNode(id="cond1", op_type="Cond")
        node_cond.attributes = {"true_branch": g_true, "false_branch": g_false}
        res_cond = vis.visit_Cond(node_cond, ["cond_val"])
        assert "if cond_val else" in res_cond

        # visit_ForiLoop
        node_fori = LogicalNode(id="fori1", op_type="ForiLoop")
        node_fori.attributes = {"body": g_body}
        res_fori = vis.visit_ForiLoop(node_fori, ["0", "10", "init_val"])
        assert res_fori == "loop_val_fori1"

        # visit_Map
        node_map = LogicalNode(id="map1", op_type="Map")
        node_map.attributes = {"f": g_f}
        res_map = vis.visit_Map(node_map, ["xs"])
        assert "bk.stack" in res_map

        # visit_Fold
        node_fold = LogicalNode(id="fold1", op_type="Fold")
        node_fold.attributes = {"f": g_f}
        res_fold = vis.visit_Fold(node_fold, ["init", "xs"])
        assert res_fold == "fold_val_fold1"

    # visit_Vmap
    assert vis.visit_Vmap(DummyNode(), ["a", "b"]) == "bk_vmap(a, b)"

    # visit_Pmap
    assert vis.visit_Pmap(DummyNode(), ["a", "b"]) == "bk_pmap(a, b)"
