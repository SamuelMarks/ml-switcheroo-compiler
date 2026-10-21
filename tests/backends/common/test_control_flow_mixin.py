"""Test module."""

from __future__ import annotations

import typing

from ml_switcheroo_compiler.backends.common.mixins.control_flow import ControlFlowASTVisitor
from tests.testing_utils import as_attrs, as_callable

if typing.TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import EagerValue


class DummyGenerator:
    """Test class DummyGenerator."""

    def get_fallback_prefix(self) -> EagerValue:
        """Test function get_fallback_prefix."""
        return "bk"


class DummyVisitor(ControlFlowASTVisitor):
    """Test class DummyVisitor."""

    def __init__(self) -> None:
        """Test function __init__."""
        setattr(self, "_generator", DummyGenerator())


class DummyNode:
    """Test class DummyNode."""

    def __init__(self, attrs: EagerValue = None) -> None:
        """Test function __init__.

        Args:
            attrs: Test parameter `attrs`.
        """
        self.attributes = attrs or {}


def test_control_flow_mixin() -> None:
    """Test function test_control_flow_mixin."""
    vis = DummyVisitor()
    node = DummyNode()

    assert as_callable(vis.visit_Scan)(node, ["a", "b"]) == "bk_scan(a, b)"
    assert as_callable(vis.visit_Switch)(node, ["a", "b"]) == "bk_switch(a, b)"

    node_td = DummyNode({"wrapped_op_name": "MyOp"})
    assert as_callable(vis.visit_TimeDistributed)(node_td, ["a"]) == "bk_time_distributed(a, 'MyOp')"

    assert as_callable(vis.visit_Assert)(node, ["cond"], data=["msg"]) == "bk_assert(cond, data=['msg'])"
    assert as_callable(vis.visit_Assert)(node, ["cond"]) == "bk_assert(cond, data=['Assertion failed.'])"

    assert as_callable(vis.visit_AssociativeScan)(node, ["a", "b"]) == "bk_associative_scan(a, b)"


def test_control_flow_mixin_advanced() -> None:
    """Test function test_control_flow_mixin_advanced."""
    from ml_switcheroo_compiler.backends.common.mixins.control_flow import ControlFlowASTVisitor
    from ml_switcheroo_compiler.ir.core import IRGraph, LogicalNode

    class DummyGenerator:
        """Test class DummyGenerator."""

        def __init__(self, graph: EagerValue = None) -> None:
            """Test function __init__.

            Args:
                graph: Test parameter `graph`.
            """
            self.graph = graph
            self.code = ["def dummy():", "    return 1"]
            self.indent_level = 0
            self.formatter = type("Formatter", (), {"indent_level": 0})()

        def get_fallback_prefix(self) -> EagerValue:
            """Test function get_fallback_prefix."""
            return "bk"

        def add_line(self, line: EagerValue) -> None:
            """Test function add_line.

            Args:
                line: Test parameter `line`.
            """
            as_callable(self.code.append)(line)

    class DummyVisitor(ControlFlowASTVisitor):
        """Test class DummyVisitor."""

        def __init__(self) -> None:
            """Test function __init__."""
            setattr(self, "_generator", DummyGenerator())
            pass

    vis = DummyVisitor()

    # Create dummy IRGraph with dummy nodes
    def make_graph() -> EagerValue:
        """Test function make_graph."""
        g = IRGraph()
        n = LogicalNode(id="dummy_out", op_type="Identity", inputs=["x"])
        object.__setattr__(g, "nodes", [n])
        return g

    g_cond = make_graph()
    g_body = make_graph()
    g_true = make_graph()
    g_false = make_graph()
    g_f = make_graph()

    # visit_WhileLoop
    node_while = LogicalNode(id="while1", op_type="WhileLoop")
    node_while.attributes = as_attrs({"cond": g_cond, "body": g_body})

    # We must patch CodeGeneratorVisitor since it attempts to parse the graph
    from unittest.mock import patch

    with patch("ml_switcheroo_compiler.backends.visitor.CodeGeneratorVisitor") as mock_cgv:
        mock_cgv_instance = mock_cgv.return_value
        mock_cgv_instance.generate_body.return_value = None

        res_while = vis.visit_WhileLoop(node_while, ["init_state"])
        assert res_while == "loop_val_while1"

        # visit_Cond
        node_cond = LogicalNode(id="cond1", op_type="Cond")
        node_cond.attributes = as_attrs({"true_branch": g_true, "false_branch": g_false})
        res_cond = vis.visit_Cond(node_cond, ["cond_val"])
        assert "if cond_val else" in res_cond

        # visit_ForiLoop
        node_fori = LogicalNode(id="fori1", op_type="ForiLoop")
        node_fori.attributes = as_attrs({"body": g_body})
        res_fori = vis.visit_ForiLoop(node_fori, ["0", "10", "init_val"])
        assert res_fori == "loop_val_fori1"

        # visit_Map
        node_map = LogicalNode(id="map1", op_type="Map")
        node_map.attributes = as_attrs({"f": g_f})
        res_map = vis.visit_Map(node_map, ["xs"])
        assert "bk.stack" in res_map

        # visit_Fold
        node_fold = LogicalNode(id="fold1", op_type="Fold")
        node_fold.attributes = as_attrs({"f": g_f})
        res_fold = vis.visit_Fold(node_fold, ["init", "xs"])
        assert res_fold == "fold_val_fold1"

    # visit_Vmap
    assert as_callable(vis.visit_Vmap)(as_callable(DummyNode)(), ["a", "b"]) == "bk_vmap(a, b)"

    # visit_Pmap
    assert as_callable(vis.visit_Pmap)(as_callable(DummyNode)(), ["a", "b"]) == "bk_pmap(a, b)"
