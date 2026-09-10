from ml_switcheroo_compiler.ir.core import IRGraph, IRNode
from ml_switcheroo_compiler.transforms.passes.config_models import FusionPatternConfig
from ml_switcheroo_compiler.transforms.passes.operator_fusion import (
    NodePattern,
    PatternMatchingEngine,
    YamlFusionRule,
    match_pattern,
)


def test_operator_fusion_extra_coverage():
    g = IRGraph()
    # 1. pattern capture=None for non-string input
    p = NodePattern(op_type=None, capture=None)
    cap = {}
    assert match_pattern(g, 123, p, cap)

    # 2. rule apply returns {}
    class EmptyRule(YamlFusionRule):
        def __init__(self):
            super().__init__("EmptyRule", FusionPatternConfig(**{"pattern": {"op_type": "Dummy"}, "replacement": {"op_type": "None", "inputs": [], "capture_to_replace": "foo"}}))

        def apply(self, graph, capture_map):
            return {}

    n1 = IRNode("n1", "Dummy")
    g.nodes["n1"] = n1
    op_pass = PatternMatchingEngine([EmptyRule()])
    op_pass.apply_passes(g)

    # 3. downstream node replaced
    class DownstreamRule(YamlFusionRule):
        def __init__(self):
            super().__init__("Downstream", FusionPatternConfig(**{"pattern": {"op_type": "A"}, "replacement": {"op_type": "None", "inputs": [], "capture_to_replace": "foo"}}))

        def apply(self, graph, capture_map):
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
        def __init__(self):
            super().__init__("Cost", FusionPatternConfig(**{"pattern": {"op_type": "Exp"}, "replacement": {"op_type": "None", "inputs": [], "capture_to_replace": "foo"}}))

        def apply(self, graph, capture_map):
            return {"e": IRNode("e", "NewExp")}

    class RejectCostModel:
        def is_fusion_valid(self, replacements):
            return False

    g = IRGraph()
    g.nodes["e"] = IRNode("e", "Exp")
    op_pass = PatternMatchingEngine([CostRule()], RejectCostModel())
    assert not op_pass.apply_passes(g)


def test_yaml_rule_apply_returns_none():
    rule = YamlFusionRule("Test", FusionPatternConfig(**{"pattern": {"op_type": "Dummy"}, "replacement": {"capture_to_replace": "val", "inputs": ["in_node", "in_val"], "op_type": "NewOp"}}))
    assert rule.apply(IRGraph(), {"val": 1.0}) is None

    n1 = IRNode("n1", "OldOp")
    n_in = IRNode("in", "InOp")
    res = rule.apply(IRGraph(), {"val": n1, "in_node": n_in, "in_val": "raw_str"})
    assert res is not None
    new_n = res["n1"]
    assert new_n.op_type == "NewOp"
    assert new_n.inputs == ["in", "raw_str"]


def test_pass_config_missing():
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
