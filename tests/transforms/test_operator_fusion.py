"""Unit tests for operator fusion passes and pattern matching transformations."""

import ml_switcheroo_compiler.transforms.passes.operator_fusion as of
from ml_switcheroo_compiler.ir.core import IRGraph, IRNode
from ml_switcheroo_compiler.transforms.passes.config_models import FusionPatternConfig
from ml_switcheroo_compiler.transforms.passes.operator_fusion import (
    NodePattern,
    PatternMatchingEngine,
    YamlFusionRule,
    match_pattern,
)


def test_operator_fusion_extra_coverage() -> None:
    """Test extra coverage paths for operator fusion patterns and graph replacements."""
    g = IRGraph()
    # 1. pattern capture=None for non-string input
    p = NodePattern(op_type=None, capture=None)
    cap = {}
    assert match_pattern(g, 123, p, cap)

    # 2. rule apply returns {}
    class EmptyRule(YamlFusionRule):
        """Rule returning empty dictionary."""

        def __init__(self) -> None:
            """Initialize EmptyRule."""
            super().__init__("EmptyRule", FusionPatternConfig(**{"pattern": {"op_type": "Dummy"}, "replacement": {"op_type": "None", "inputs": [], "capture_to_replace": "foo"}}))

        def apply(self, graph, capture_map):
            """Apply empty replacement."""
            return {}

    n1 = IRNode("n1", "Dummy")
    g.nodes["n1"] = n1
    op_pass = PatternMatchingEngine([EmptyRule()])
    op_pass.apply_passes(g)

    # 3. downstream node replaced
    class DownstreamRule(YamlFusionRule):
        """Rule replacing downstream nodes."""

        def __init__(self) -> None:
            """Initialize DownstreamRule."""
            super().__init__("Downstream", FusionPatternConfig(**{"pattern": {"op_type": "A"}, "replacement": {"op_type": "None", "inputs": [], "capture_to_replace": "foo"}}))

        def apply(self, graph, capture_map):
            """Apply downstream replacements."""
            # when matching A, we replace B as well, putting B in new_nodes
            n = IRNode("b_new", "ReplacedB")
            # give it different id
            n2 = IRNode("c", "ReplacedB2")
            return {"a": IRNode("a", "ReplacedA"), "b": n, "c_old": n2}

    g = IRGraph()
    g.nodes["a"] = IRNode("a", "A", inputs=["b", "c_old"])
    g.nodes["b"] = IRNode("b", "B")
    g.nodes["d"] = IRNode("d", "D", inputs=["b"])
    g.inputs = ["c_old", "b"]
    g.outputs = ["b", "a"]

    op_pass = PatternMatchingEngine([DownstreamRule()])
    op_pass.apply_passes(g)
    assert g.nodes["b_new"].op_type == "ReplacedB"
    assert g.nodes["c"].op_type == "ReplacedB2"
    assert g.inputs[0] == "c"
    assert g.inputs[1] == "b_new"
    assert g.outputs[0] == "b_new"
    assert g.outputs[1] == "a"  # a was not renamed
    assert g.nodes["d"].inputs[0] == "b_new"

    # 4. Fused estimate cost
    class CostRule(YamlFusionRule):
        """Rule with custom cost modeling."""

        def __init__(self) -> None:
            """Initialize CostRule."""
            super().__init__("Cost", FusionPatternConfig(**{"pattern": {"op_type": "Exp"}, "replacement": {"op_type": "None", "inputs": [], "capture_to_replace": "foo"}}))

        def apply(self, graph, capture_map):
            """Apply cost replacement."""
            return {"e": IRNode("e", "NewExp")}

    class RejectCostModel:
        """Cost model rejecting all fusions."""

        def is_fusion_valid(self, replacements):
            """Reject fusion."""
            return False

    g = IRGraph()
    g.nodes["e"] = IRNode("e", "Exp")
    op_pass = PatternMatchingEngine([CostRule()], RejectCostModel())
    assert not op_pass.apply_passes(g)


def test_yaml_rule_apply_returns_none() -> None:
    """Test YamlFusionRule apply edge case returning None on invalid target."""
    rule = YamlFusionRule("Test", FusionPatternConfig(**{"pattern": {"op_type": "Dummy"}, "replacement": {"capture_to_replace": "val", "inputs": ["in_node", "in_val"], "op_type": "NewOp"}}))
    assert rule.apply(IRGraph(), {"val": 1.0}) is None

    n1 = IRNode("n1", "OldOp")
    n_in = IRNode("in", "InOp")
    res = rule.apply(IRGraph(), {"val": n1, "in_node": n_in, "in_val": "raw_str"})
    assert res is not None
    new_n = res["n1"]
    assert new_n.op_type == "NewOp"
    assert new_n.inputs == ["in", "raw_str"]


def test_pass_config_missing() -> None:
    """Test behavior when pass configuration file does not exist."""
    from unittest.mock import patch

    import ml_switcheroo_compiler.transforms.passes.operator_fusion as of

    with patch("os.path.exists", return_value=False):
        cfg = of._load_pass_config()
        assert not cfg.execution_order
        assert not cfg.fusion_patterns


def test_operator_fusion_all_remaining_branches():
    """Test 227->229, 260->273, 266->262, and 380->382 in operator_fusion.py."""
    import tempfile
    from unittest.mock import mock_open, patch

    import yaml

    import ml_switcheroo_compiler.transforms.passes.operator_fusion as of
    from ml_switcheroo_compiler.ir.core import IRNode

    # 1. 227->229: yaml exists but contains a non-dict (e.g. a list)
    with patch("os.path.exists", return_value=True):
        with patch("builtins.open", mock_open(read_data=yaml.dump(["list_item"]))):
            cfg_non_dict = of._load_pass_config()
            assert not cfg_non_dict.execution_order

    # 2. 260->273: patterns_dir is not a directory
    rules_empty = of._discover_fusion_patterns("/nonexistent/directory/path/for/tests")
    assert rules_empty == []

    # 3. 266->262: yaml file in patterns_dir contains a non-dict
    with tempfile.TemporaryDirectory() as tmpdir:
        non_dict_yaml = f"{tmpdir}/invalid.yaml"
        with open(non_dict_yaml, "w") as f:
            yaml.dump(["item1", "item2"], f)
        rules = of._discover_fusion_patterns(tmpdir)
        assert rules == []

    # 4. 380->382: memory_sizes is not a dict in is_fusion_valid
    cm = of.MemoryAwareCostModel(config={"memory_sizes": None})
    n = IRNode(id="n1", op_type="Add", shape_metadata=(2, 2))
    fits = cm.is_fusion_valid({"n1": n})
    assert fits is True


def test_fusion_rule_default_graph_transformation() -> None:
    """Test FusionRule.apply default graph transformation behavior."""
    rule = of.FusionRule("CustomTransform", NodePattern(op_type="Relu"))
    g = IRGraph()

    # Empty match returns None
    assert rule.apply(g, {}) is None

    # Match with non-IRNode values returns None
    assert rule.apply(g, {"val": "some_str"}) is None

    # Match with root IRNode creates standard fused node
    node = IRNode("r1", "Relu", inputs=["in_0"])
    g.nodes["r1"] = node
    res = rule.apply(g, {"root": node})
    assert res is not None
    assert "r1" in res
    assert res["r1"].op_type == "Fused_CustomTransform"
    assert res["r1"].attributes["fused_pattern"] == "CustomTransform"

    # Match without root key falls back to first IRNode in match
    res_fallback = rule.apply(g, {"target_node": node})
    assert res_fallback is not None
    assert "r1" in res_fallback


def test_vertical_fusion_conv_bias_relu() -> None:
    """Test vertical fusion for Conv2D + BiasAdd + Relu patterns."""
    g = IRGraph()
    x = IRNode("x", "Input", shape_metadata=[1, 3, 32, 32])
    w = IRNode("w", "Input", shape_metadata=[16, 3, 3, 3])
    bias = IRNode("b", "Input", shape_metadata=[16])
    conv = IRNode("conv", "Conv2D", inputs=["x", "w"], shape_metadata=[1, 16, 30, 30])
    conv.attributes = {"strides": [1, 1], "padding": "valid"}
    bias_add = IRNode("bias_add", "BiasAdd", inputs=["conv", "b"], shape_metadata=[1, 16, 30, 30])
    relu = IRNode("relu", "Relu", inputs=["bias_add"], shape_metadata=[1, 16, 30, 30])

    g.nodes = {"x": x, "w": w, "b": bias, "conv": conv, "bias_add": bias_add, "relu": relu}
    g.inputs = ["x", "w", "b"]
    g.outputs = ["relu"]

    vpass = of.VerticalFusionPass()
    modified = vpass.run(g)

    assert modified is True
    assert "relu" in g.nodes
    fused = g.nodes["relu"]
    assert fused.op_type == "FusedConv2DBiasRelu"
    assert fused.inputs == ["x", "w", "b"]
    assert fused.attributes["activation"] == "Relu"
    assert fused.attributes["has_bias"] is True
    assert "conv" not in g.nodes
    assert "bias_add" not in g.nodes


def test_vertical_fusion_elementwise_activation() -> None:
    """Test vertical fusion for elementwise-activation patterns."""
    g = IRGraph()
    in1 = IRNode("in1", "Input", shape_metadata=[4, 4])
    in2 = IRNode("in2", "Input", shape_metadata=[4, 4])
    add = IRNode("add", "Add", inputs=["in1", "in2"], shape_metadata=[4, 4])
    sig = IRNode("sig", "Sigmoid", inputs=["add"], shape_metadata=[4, 4])

    g.nodes = {"in1": in1, "in2": in2, "add": add, "sig": sig}
    g.inputs = ["in1", "in2"]
    g.outputs = ["sig"]

    modified = of.fuse_elementwise_activation(g)
    assert modified is True
    assert "sig" in g.nodes
    fused = g.nodes["sig"]
    assert fused.op_type == "AddSigmoid"
    assert fused.inputs == ["in1", "in2"]
    assert "add" not in g.nodes


def test_horizontal_fusion_siblings() -> None:
    """Test horizontal fusion for sibling nodes sharing identical inputs."""
    g = IRGraph()
    x = IRNode("x", "Input", shape_metadata=[8, 8])
    r1 = IRNode("r1", "Relu", inputs=["x"], shape_metadata=[8, 8])
    r2 = IRNode("r2", "Sigmoid", inputs=["x"], shape_metadata=[8, 8])
    out = IRNode("out", "Add", inputs=["r1", "r2"], shape_metadata=[8, 8])

    g.nodes = {"x": x, "r1": r1, "r2": r2, "out": out}
    g.inputs = ["x"]
    g.outputs = ["out"]

    hpass = of.HorizontalFusionPass()
    modified = hpass.run(g)

    assert modified is True
    # Find the horizontal fused node
    h_node = next(n for n in g.nodes.values() if n.op_type == "HorizontalFusedOps")
    assert h_node.inputs == ["x"]
    assert h_node.attributes["fused_ops"] == ["Relu", "Sigmoid"]
    assert g.nodes["r1"].op_type == "TupleGetItem"
    assert g.nodes["r2"].op_type == "TupleGetItem"


def test_fusion_edge_cases() -> None:
    """Test edge cases: empty graphs and non-matching multi-consumer nodes."""
    empty_g = IRGraph()
    assert of.fuse_conv_bias_relu(empty_g) is False
    assert of.fuse_elementwise_activation(empty_g) is False
    assert of.fuse_horizontal_patterns(empty_g) is False

    # Multi-consumer preventing vertical fusion
    g = IRGraph()
    x = IRNode("x", "Input", shape_metadata=[2, 2])
    y = IRNode("y", "Input", shape_metadata=[2, 2])
    add = IRNode("add", "Add", inputs=["x", "y"], shape_metadata=[2, 2])
    r1 = IRNode("r1", "Relu", inputs=["add"], shape_metadata=[2, 2])
    other = IRNode("other", "Mul", inputs=["add", "x"], shape_metadata=[2, 2])
    g.nodes = {"x": x, "y": y, "add": add, "r1": r1, "other": other}
    g.outputs = ["r1", "other"]

    assert of.fuse_elementwise_activation(g) is False
    assert "add" in g.nodes

    # Non-elementwise input to activation
    g2 = IRGraph()
    g2.nodes["in_node"] = IRNode("in_node", "CustomOp")
    g2.nodes["act"] = IRNode("act", "Relu", inputs=["in_node"])
    assert of.fuse_elementwise_activation(g2) is False

    # Missing bias node in conv_bias_relu
    g3 = IRGraph()
    g3.nodes["act"] = IRNode("act", "Relu", inputs=["missing_bias"])
    assert of.fuse_conv_bias_relu(g3) is False

    # Bias node with non-Add op_type
    g4 = IRGraph()
    g4.nodes["non_bias"] = IRNode("non_bias", "Mul", inputs=["a", "b"])
    g4.nodes["act"] = IRNode("act", "Relu", inputs=["non_bias"])
    assert of.fuse_conv_bias_relu(g4) is False

    # Add without Conv input
    g5 = IRGraph()
    g5.nodes["in_a"] = IRNode("in_a", "Input")
    g5.nodes["in_b"] = IRNode("in_b", "Input")
    g5.nodes["add"] = IRNode("add", "Add", inputs=["in_a", "in_b"])
    g5.nodes["act"] = IRNode("act", "Relu", inputs=["add"])
    assert of.fuse_conv_bias_relu(g5) is False

    # Conv with multi-consumers
    g6 = IRGraph()
    g6.nodes["x"] = IRNode("x", "Input")
    g6.nodes["w"] = IRNode("w", "Input")
    g6.nodes["b"] = IRNode("b", "Input")
    g6.nodes["conv"] = IRNode("conv", "Conv2D", inputs=["x", "w"])
    g6.nodes["bias_add"] = IRNode("bias_add", "Add", inputs=["conv", "b"])
    g6.nodes["act"] = IRNode("act", "Relu", inputs=["bias_add"])
    g6.nodes["other_conv_user"] = IRNode("other_conv_user", "Sub", inputs=["conv", "x"])
    assert of.fuse_conv_bias_relu(g6) is False

    # Horizontal siblings with shape mismatch
    g7 = IRGraph()
    g7.nodes["x"] = IRNode("x", "Input")
    s1 = IRNode("s1", "Relu", inputs=["x"], shape_metadata=[4, 4])
    s2 = IRNode("s2", "Sigmoid", inputs=["x"], shape_metadata=[8, 8])
    g7.nodes["s1"] = s1
    g7.nodes["s2"] = s2
    assert of.fuse_horizontal_patterns(g7) is False

    # OperatorFusionPass class run
    pass_runner = of.OperatorFusionPass()
    g8 = IRGraph()
    g8.nodes["x"] = IRNode("x", "Input")
    assert pass_runner.run(g8) is False
