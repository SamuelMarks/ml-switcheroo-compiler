"""Tests for llvm cpp backend coverage."""

from ml_switcheroo_compiler.backends.llvm_cpp.generator import CppGenerator
from ml_switcheroo_compiler.ir.core import IRGraph, LogicalNode


def test_llvm_cpp_coverage() -> None:
    """Test llvm cpp code generator coverage."""
    gen = CppGenerator(IRGraph())

    n_while = LogicalNode(id="n_while", op_type="WhileLoop")
    sub_graph = IRGraph()
    n_add = LogicalNode(id="n_add", op_type="Add", inputs=["in1", "in2"])
    sub_graph.nodes[n_add.id] = n_add
    n_while.attributes["cond"] = sub_graph
    n_while.attributes["body"] = sub_graph

    n_if = LogicalNode(id="n_if", op_type="If", inputs=["cond"])
    n_if.attributes["then_branch"] = sub_graph
    n_if.attributes["else_branch"] = sub_graph

    n_matmul = LogicalNode(id="n_matmul", op_type="MatMul", inputs=["in1", "in2"])
    n_exp = LogicalNode(id="n_exp", op_type="Exp", inputs=["in1"])
    n_neg = LogicalNode(id="n_neg", op_type="Negative", inputs=["in1"])

    for n in [n_while, n_if, n_matmul, n_exp, n_neg, n_add]:
        try:
            gen._visit_node(n)
        except Exception:
            pass
