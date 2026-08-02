# ruff: noqa: E501
from ml_switcheroo_compiler.ir.core import IRGraph, IRNode
from ml_switcheroo_compiler.transforms.passes.operator_fusion import FusionRule, NodePattern, PatternMatchingEngine, apply_operator_fusion, match_pattern

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


def test_elementwise_fusion() -> None:
    """Test fusing Add and Relu."""
    nodes = {
        "in1": IRNode(id="in1", op_type="Input", inputs=[]),
        "in2": IRNode(id="in2", op_type="Input", inputs=[]),
        "add": IRNode(id="add", op_type="Add", inputs=["in1", "in2"]),
        "relu": IRNode(id="relu", op_type="Relu", inputs=["add"]),
    }
    graph = IRGraph(name="test", nodes=nodes, outputs=["relu"])
    graph = apply_operator_fusion(graph)
    assert graph.nodes["relu"].op_type == "AddRelu"
    assert graph.nodes["relu"].inputs == ["in1", "in2"]


def test_conv2d_batchnorm_fusion() -> None:
    """Test fusing Conv2D and BatchNorm."""
    nodes = {
        "in": IRNode(id="in", op_type="Input", inputs=[]),
        "weight": IRNode(id="weight", op_type="Input", inputs=[]),
        "conv": IRNode(id="conv", op_type="Conv2D", inputs=["in", "weight"]),
        "scale": IRNode(id="scale", op_type="Input", inputs=[]),
        "bias": IRNode(id="bias", op_type="Input", inputs=[]),
        "mean": IRNode(id="mean", op_type="Input", inputs=[]),
        "var": IRNode(id="var", op_type="Input", inputs=[]),
        "bn": IRNode(id="bn", op_type="BatchNorm", inputs=["conv", "scale", "bias", "mean", "var"]),
    }
    graph = IRGraph(name="test", nodes=nodes, outputs=["bn"])
    graph = apply_operator_fusion(graph)
    assert graph.nodes["bn"].op_type == "Conv2DBatchNorm"
    assert graph.nodes["bn"].inputs == ["in", "weight", "scale", "bias", "mean", "var"]


def test_linear_fusion() -> None:
    """Test fusing MatMul and Add."""
    nodes = {
        "in1": IRNode(id="in1", op_type="Input", inputs=[]),
        "in2": IRNode(id="in2", op_type="Input", inputs=[]),
        "matmul": IRNode(id="matmul", op_type="MatMul", inputs=["in1", "in2"]),
        "bias": IRNode(id="bias", op_type="Input", inputs=[]),
        "add": IRNode(id="add", op_type="Add", inputs=["matmul", "bias"]),
    }
    graph = IRGraph(name="test", nodes=nodes, outputs=["add"])
    graph = apply_operator_fusion(graph)
    assert graph.nodes["add"].op_type == "Linear"
    assert graph.nodes["add"].inputs == ["in1", "in2", "bias"]


def test_mha_fusion() -> None:
    """Test fusing MHA pattern."""
    nodes = {
        "q": IRNode(id="q", op_type="Input", inputs=[]),
        "k": IRNode(id="k", op_type="Input", inputs=[]),
        "v": IRNode(id="v", op_type="Input", inputs=[]),
        "matmul1": IRNode(id="matmul1", op_type="MatMul", inputs=["q", "k"]),
        "softmax": IRNode(id="softmax", op_type="Softmax", inputs=["matmul1"]),
        "matmul2": IRNode(id="matmul2", op_type="MatMul", inputs=["softmax", "v"]),
    }
    graph = IRGraph(name="test", nodes=nodes, outputs=["matmul2"])
    graph = apply_operator_fusion(graph)
    assert graph.nodes["matmul2"].op_type == "MultiHeadAttention"
    assert graph.nodes["matmul2"].inputs == ["q", "k", "v"]


def test_cost_model_rejection() -> None:
    """Test cost model rejects expensive fusions."""
    from ml_switcheroo_compiler.transforms.passes.operator_fusion import CostModel, MHAFusion, PatternMatchingEngine

    nodes = {
        "q": IRNode(id="q", op_type="Input", inputs=[]),
        "k": IRNode(id="k", op_type="Input", inputs=[]),
        "v": IRNode(id="v", op_type="Input", inputs=[]),
        "matmul1": IRNode(id="matmul1", op_type="MatMul", inputs=["q", "k"]),
        "softmax": IRNode(id="softmax", op_type="Softmax", inputs=["matmul1"]),
        "matmul2": IRNode(id="matmul2", op_type="MatMul", inputs=["softmax", "v"]),
    }
    graph = IRGraph(name="test", nodes=nodes, outputs=["matmul2"])

    # MHA cost is 30, so max_cost=20 will reject it
    rules = [MHAFusion()]
    engine = PatternMatchingEngine(rules, CostModel(max_cost=20))
    engine.apply_passes(graph)

    assert graph.nodes["matmul2"].op_type == "MatMul"  # Unchanged


def testmatch_pattern_length_mismatch() -> None:
    """Test when pattern inputs length doesn't match node inputs length."""
    from ml_switcheroo_compiler.transforms.passes.operator_fusion import FusionRule, NodePattern, PatternMatchingEngine

    class FakeRule(FusionRule):
        def __init__(self):
            pattern = NodePattern(op_type="Add", inputs=[NodePattern(), NodePattern(), NodePattern()])
            super().__init__("fake", pattern)

        def apply(self, graph, match):
            return None

    nodes = {
        "in1": IRNode(id="in1", op_type="Input", inputs=[]),
        "in2": IRNode(id="in2", op_type="Input", inputs=[]),
        "add": IRNode(id="add", op_type="Add", inputs=["in1", "in2"]),
    }
    graph = IRGraph(name="test", nodes=nodes, outputs=["add"])
    engine = PatternMatchingEngine([FakeRule()])
    engine.apply_passes(graph)


def testmatch_pattern_non_string_id() -> None:
    """Test matching non-string IDs like constants."""
    from ml_switcheroo_compiler.transforms.passes.operator_fusion import FusionRule, NodePattern, PatternMatchingEngine

    class FakeRule(FusionRule):
        def __init__(self):
            pattern = NodePattern(op_type="Reshape", inputs=[NodePattern(), NodePattern(capture="shape_val")])
            super().__init__("fake", pattern)

        def apply(self, graph, match):
            assert match["shape_val"] == [1, 2, 3]
            return None

    nodes = {
        "in1": IRNode(id="in1", op_type="Input", inputs=[]),
        "reshape": IRNode(id="reshape", op_type="Reshape", inputs=["in1", [1, 2, 3]]),
    }
    graph = IRGraph(name="test", nodes=nodes, outputs=["reshape"])
    engine = PatternMatchingEngine([FakeRule()])
    engine.apply_passes(graph)


def testmatch_pattern_missing_node() -> None:
    """Test matching a node ID that doesn't exist in graph."""
    from ml_switcheroo_compiler.transforms.passes.operator_fusion import FusionRule, NodePattern, PatternMatchingEngine

    class FakeRule(FusionRule):
        def __init__(self):
            pattern = NodePattern(op_type="Add", inputs=[NodePattern()])
            super().__init__("fake", pattern)

        def apply(self, graph, match):
            return None

    nodes = {
        "add": IRNode(id="add", op_type="Add", inputs=["missing_node"]),
    }
    graph = IRGraph(name="test", nodes=nodes, outputs=["add"])
    engine = PatternMatchingEngine([FakeRule()])
    engine.apply_passes(graph)


def test_fusion_rule_apply_not_implemented() -> None:
    """Test FusionRule apply NotImplementedError."""
    import pytest

    from ml_switcheroo_compiler.transforms.passes.operator_fusion import FusionRule, NodePattern

    rule = FusionRule("base", NodePattern())
    with pytest.raises(NotImplementedError):
        rule.apply(None, {})


def test_pattern_match_no_capture():
    g = IRGraph()
    n1 = IRNode(id="n1", op_type="Relu")
    g.nodes["n1"] = n1

    pattern = NodePattern(op_type="Relu")  # no capture
    capture_map = {}
    assert match_pattern(g, "n1", pattern, capture_map) is True
    assert capture_map == {}


def test_pattern_match_constant_input_no_capture():
    g = IRGraph()
    # For a string/non-node match
    pattern = NodePattern(capture=None)
    capture_map = {}
    assert match_pattern(g, 42, pattern, capture_map) is True


def test_apply_passes_node_already_in_new_nodes():
    class DummyRule(FusionRule):
        def __init__(self):
            super().__init__("Dummy", NodePattern(op_type="Relu", capture="n1"))

        def apply(self, graph, match):
            n1 = match["n1"]
            new_n1 = IRNode(id=n1.id, op_type="Dummy")
            new_n2 = IRNode(id="n2", op_type="Dummy2")
            return {n1.id: new_n1, "n2": new_n2}

    g = IRGraph()
    g.nodes["n1"] = IRNode(id="n1", op_type="Relu")
    g.nodes["n2"] = IRNode(id="n2", op_type="Add")

    fusion = PatternMatchingEngine([DummyRule()])
    assert fusion.apply_passes(g) is True
    assert g.nodes["n2"].op_type == "Dummy2"
