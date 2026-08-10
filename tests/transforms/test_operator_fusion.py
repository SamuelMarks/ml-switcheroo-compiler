"""Unit tests for Operator Fusion pass."""

from ml_switcheroo_compiler.ir.core import IRGraph, IRNode
from ml_switcheroo_compiler.transforms.passes.operator_fusion import (
    ConsecutiveElementwiseFusion,
    Conv2DBatchNormFusion,
    ElementwiseFusion,
    FMAFusion,
    LinearFusion,
    MHAFusion,
    NodePattern,
    NormalizationFusion,
    PatternMatchingEngine,
    ReshapeReshapeFusion,
    apply_operator_fusion,
    match_pattern,
)


def test_match_pattern():
    """Test pattern matching logic."""
    graph = IRGraph()
    n1 = IRNode(id="n1", op_type="Input")
    n2 = IRNode(id="n2", op_type="Add", inputs=["n1", "n1"])
    graph.nodes = {"n1": n1, "n2": n2}

    pat = NodePattern(op_type="Add", capture="add", inputs=[NodePattern(capture="in1"), NodePattern(capture="in2")])
    cap = {}

    assert match_pattern(graph, "n2", pat, cap) is True
    assert cap["add"] == n2
    assert cap["in1"] == n1
    assert cap["in2"] == n1

    # Should fail matching a raw val
    assert match_pattern(graph, 1.0, pat, {}) is False


def test_reshape_reshape_fusion():
    """Test fusing two consecutive reshapes."""
    graph = IRGraph()
    graph.nodes = {"n1": IRNode(id="n1", op_type="Input"), "shape1": IRNode(id="shape1", op_type="Constant"), "shape2": IRNode(id="shape2", op_type="Constant"), "r1": IRNode(id="r1", op_type="Reshape", inputs=["n1", "shape1"]), "r2": IRNode(id="r2", op_type="Reshape", inputs=["r1", "shape2"])}
    graph.outputs = ["r2"]

    rule = ReshapeReshapeFusion()
    engine = PatternMatchingEngine([rule])
    assert engine.apply_passes(graph) is True

    # Note: DCE cleans up r1, operator fusion leaves it dangling
    assert "r1" in graph.nodes
    assert graph.nodes["r2"].inputs == ["n1", "shape2"]


def test_elementwise_fusion():
    """Test fusing Add and Relu."""
    graph = IRGraph()
    graph.nodes = {"n1": IRNode(id="n1", op_type="Input"), "n2": IRNode(id="n2", op_type="Input"), "add1": IRNode(id="add1", op_type="Add", inputs=["n1", "n2"]), "relu1": IRNode(id="relu1", op_type="Relu", inputs=["add1"])}
    graph.outputs = ["relu1"]

    rule = ElementwiseFusion()
    engine = PatternMatchingEngine([rule])
    assert engine.apply_passes(graph) is True

    assert graph.nodes["relu1"].op_type == "AddRelu"
    assert graph.nodes["relu1"].inputs == ["n1", "n2"]


def test_conv2d_batchnorm_fusion():
    """Test fusing Conv2D and BatchNorm."""
    graph = IRGraph()
    graph.nodes = {
        "in": IRNode(id="in", op_type="Input"),
        "w": IRNode(id="w", op_type="Input"),
        "conv": IRNode(id="conv", op_type="Conv2D", inputs=["in", "w"]),
        "s": IRNode(id="s", op_type="Input"),
        "b": IRNode(id="b", op_type="Input"),
        "m": IRNode(id="m", op_type="Input"),
        "v": IRNode(id="v", op_type="Input"),
        "bn": IRNode(id="bn", op_type="BatchNorm", inputs=["conv", "s", "b", "m", "v"]),
    }
    graph.outputs = ["bn"]

    rule = Conv2DBatchNormFusion()
    engine = PatternMatchingEngine([rule])
    assert engine.apply_passes(graph) is True

    assert graph.nodes["bn"].op_type == "Conv2DBatchNorm"
    assert graph.nodes["bn"].inputs == ["in", "w", "s", "b", "m", "v"]


def test_linear_fusion():
    """Test fusing MatMul and BiasAdd."""
    graph = IRGraph()
    graph.nodes = {
        "n1": IRNode(id="n1", op_type="Input"),
        "w": IRNode(id="w", op_type="Input"),
        "matmul": IRNode(id="matmul", op_type="MatMul", inputs=["n1", "w"]),
        "b": IRNode(id="b", op_type="Input"),
        "add": IRNode(id="add", op_type="Add", inputs=["matmul", "b"]),
    }
    graph.outputs = ["add"]

    rule = LinearFusion()
    engine = PatternMatchingEngine([rule])
    assert engine.apply_passes(graph) is True

    assert graph.nodes["add"].op_type == "Linear"
    assert graph.nodes["add"].inputs == ["n1", "w", "b"]


def test_mha_fusion():
    """Test fusing MHA."""
    graph = IRGraph()
    graph.nodes = {
        "q": IRNode(id="q", op_type="Input"),
        "k": IRNode(id="k", op_type="Input"),
        "v": IRNode(id="v", op_type="Input"),
        "qk": IRNode(id="qk", op_type="MatMul", inputs=["q", "k"]),
        "soft": IRNode(id="soft", op_type="Softmax", inputs=["qk"]),
        "out": IRNode(id="out", op_type="MatMul", inputs=["soft", "v"]),
    }
    graph.outputs = ["out"]

    rule = MHAFusion()
    engine = PatternMatchingEngine([rule])
    assert engine.apply_passes(graph) is True

    assert graph.nodes["out"].op_type == "MultiHeadAttention"
    assert graph.nodes["out"].inputs == ["q", "k", "v"]


def test_apply_operator_fusion():
    """Test the main API wrapper."""
    graph = IRGraph()
    graph.nodes = {"n1": IRNode(id="n1", op_type="Input"), "n2": IRNode(id="n2", op_type="Input"), "add1": IRNode(id="add1", op_type="Add", inputs=["n1", "n2"]), "relu1": IRNode(id="relu1", op_type="Relu", inputs=["add1"])}
    graph.outputs = ["relu1"]

    apply_operator_fusion(graph)
    assert graph.nodes["relu1"].op_type == "AddRelu"


def test_pattern_matching_edge_cases():
    """Test edge cases in pattern matching."""
    graph = IRGraph()
    n1 = IRNode(id="n1", op_type="Input")
    graph.nodes = {"n1": n1}

    # Line 52: len(node.inputs) != len(p_inputs)
    pat1 = NodePattern(op_type="Input", inputs=[NodePattern()])
    assert match_pattern(graph, "n1", pat1, {}) is False

    # Lines 80-82: Raw value matching with capture
    pat2 = NodePattern(capture="val")
    cap = {}
    assert match_pattern(graph, 1.0, pat2, cap) is True
    assert cap["val"] == 1.0

    # Line 117: FusionRule.apply NotImplementedError
    # Line 194 and 472: CostModel rejection
    from ml_switcheroo_compiler.transforms.passes.operator_fusion import CostModel, FusionRule, PatternMatchingEngine

    class MockRule(FusionRule):
        def apply(self, graph, match):
            return {"n1": IRNode(id="n1", op_type="MultiHeadAttention")}  # cost is 30

    rule = MockRule("expensive", NodePattern(op_type="Input"))
    cost_model = CostModel(max_cost=10)  # 30 > 10, so it will be rejected
    engine = PatternMatchingEngine([rule], cost_model)

    # Should be rejected by CostModel, so graph is unmodified
    assert engine.apply_passes(graph) is False


def test_fma_fusion():
    """Test fusing Multiply and Add."""
    graph = IRGraph()
    graph.nodes = {
        "n1": IRNode(id="n1", op_type="Input"),
        "n2": IRNode(id="n2", op_type="Input"),
        "n3": IRNode(id="n3", op_type="Input"),
        "mul": IRNode(id="mul", op_type="Multiply", inputs=["n1", "n2"]),
        "add": IRNode(id="add", op_type="Add", inputs=["mul", "n3"]),
    }
    graph.outputs = ["add"]

    rule = FMAFusion()
    engine = PatternMatchingEngine([rule])
    assert engine.apply_passes(graph) is True

    assert graph.nodes["add"].op_type == "FusedMultiplyAdd"
    assert graph.nodes["add"].inputs == ["n1", "n2", "n3"]


def test_normalization_fusion():
    """Test fusing LayerNorm pattern."""
    graph = IRGraph()
    graph.nodes = {
        "in": IRNode(id="in", op_type="Input"),
        "mean1": IRNode(id="mean1", op_type="ReduceMean", inputs=["in"]),
        "sub": IRNode(id="sub", op_type="Subtract", inputs=["in", "mean1"]),
        "mean2": IRNode(id="mean2", op_type="ReduceMean", inputs=["sub"]),
        "eps": IRNode(id="eps", op_type="Constant"),
        "add": IRNode(id="add", op_type="Add", inputs=["mean2", "eps"]),
        "sqrt": IRNode(id="sqrt", op_type="Sqrt", inputs=["add"]),
        "div": IRNode(id="div", op_type="Divide", inputs=["sub", "sqrt"]),
    }
    graph.outputs = ["div"]

    rule = NormalizationFusion()
    engine = PatternMatchingEngine([rule])
    assert engine.apply_passes(graph) is True

    assert graph.nodes["div"].op_type == "LayerNorm"
    assert graph.nodes["div"].inputs == ["in"]


def test_consecutive_elementwise_fusion():
    """Test fusing consecutive pointwise operations."""
    graph = IRGraph()
    graph.nodes = {
        "in": IRNode(id="in", op_type="Input"),
        "log": IRNode(id="log", op_type="Log", inputs=["in"]),
        "exp": IRNode(id="exp", op_type="Exp", inputs=["log"]),
    }
    graph.outputs = ["exp"]

    rule = ConsecutiveElementwiseFusion()
    engine = PatternMatchingEngine([rule])
    assert engine.apply_passes(graph) is True

    assert graph.nodes["exp"].op_type == "FusedLogExp"
    assert graph.nodes["exp"].inputs == ["in"]


def test_base_rule_apply():
    from ml_switcheroo_compiler.transforms.passes.operator_fusion import FusionRule

    rule = FusionRule("base", NodePattern())
    assert rule.apply(IRGraph(), {}) is None
