# ruff: noqa: E501
from ml_switcheroo_compiler.ir.core import IRGraph, IRNode
from ml_switcheroo_compiler.transforms.passes.operator_fusion import apply_operator_fusion

"Extra tests for operator fusion."


def test_operator_fusion_non_str_input() -> None:
    """Test fusing when input is not a string."""
    nodes = {"reshape1": IRNode(id="reshape1", op_type="Reshape", inputs=[5, "shape1"])}
    graph = IRGraph(name="test", nodes=nodes, outputs=["reshape1"])
    graph = apply_operator_fusion(graph)
    assert graph.nodes["reshape1"].inputs == [5, "shape1"]


"Tests for operator fusion pass."


def test_operator_fusion_reshape() -> None:
    """Test fusing consecutive Reshape nodes."""
    nodes = {
        "input": IRNode(id="input", op_type="Input", inputs=[]),
        "shape1": IRNode(id="shape1", op_type="Constant", inputs=[]),
        "reshape1": IRNode(id="reshape1", op_type="Reshape", inputs=["input", "shape1"]),
        "shape2": IRNode(id="shape2", op_type="Constant", inputs=[]),
        "reshape2": IRNode(id="reshape2", op_type="Reshape", inputs=["reshape1", "shape2"]),
        "other": IRNode(id="other", op_type="Add", inputs=["reshape2", "reshape2"]),
    }
    graph = IRGraph(name="test", nodes=nodes, outputs=["other"])
    graph = apply_operator_fusion(graph)
    assert graph.nodes["reshape2"].inputs == ["input", "shape2"]
    assert graph.nodes["other"].inputs == ["reshape2", "reshape2"]


def test_operator_fusion_no_op() -> None:
    """Test when no fusions apply."""
    nodes = {"input": IRNode(id="input", op_type="Input", inputs=[]), "reshape1": IRNode(id="reshape1", op_type="Reshape", inputs=["input", "shape1"])}
    graph = IRGraph(name="test", nodes=nodes, outputs=["reshape1"])
    graph = apply_operator_fusion(graph)
    assert graph.nodes["reshape1"].inputs == ["input", "shape1"]
