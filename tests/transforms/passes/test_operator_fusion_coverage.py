from ml_switcheroo_compiler.ir.core import LogicalNode


def test_operator_fusion_cost_fused():
    node = LogicalNode(id="n1", op_type="FusedMultiHeadAttention")
    pass


def test_operator_fusion_extra_coverage():
    from ml_switcheroo_compiler.ir.core import IRGraph, IRNode
    from ml_switcheroo_compiler.transforms.passes.config_models import NodePatternConfig
    from ml_switcheroo_compiler.transforms.passes.operator_fusion import match_pattern

    graph = IRGraph()
    n1 = IRNode(id="n1", op_type="Op1", inputs=["in1", "in2"])
    graph.nodes["n1"] = n1

    # len(inputs) mismatch
    pat = NodePatternConfig(op_type="Op1", inputs=[NodePatternConfig(op_type="Input")])  # 1 input vs 2
    assert match_pattern(graph, "n1", pat, {}) is False

    # not isinstance(node_id, str)
    assert match_pattern(graph, 123, NodePatternConfig(op_type="Input"), {}) is False
    cmap = {}
    assert match_pattern(graph, 123, NodePatternConfig(capture="test"), cmap) is True
    assert cmap["test"] == 123

    # not node or op_type mismatch
    assert match_pattern(graph, "n3", NodePatternConfig(op_type="Input"), {}) is False
    assert match_pattern(graph, "n1", NodePatternConfig(op_type="Input"), {}) is False

    # capture is not None, inputs is None
    assert match_pattern(graph, "n1", NodePatternConfig(capture="test_cap"), cmap) is True
    assert cmap["test_cap"] == n1

    # apply_replacement base class coverage
    from ml_switcheroo_compiler.transforms.passes.config_models import FusionPatternConfig, ReplacementConfig
    from ml_switcheroo_compiler.transforms.passes.operator_fusion import YamlFusionRule

    cfg = FusionPatternConfig(pattern=NodePatternConfig(), replacement=ReplacementConfig(op_type="x", inputs=[], capture_to_replace="x"))
    strat = YamlFusionRule("test", cfg)
    strat.apply(graph, {"x": "n1"})

    # _match_inputs tests
    n2 = IRNode(id="n2", op_type="Input", inputs=[])
    graph.nodes["n2"] = n2
    graph.nodes["n1"].inputs = ["n2", "n2"]
    # recursive match fails
    pat_fail = NodePatternConfig(op_type="Op1", inputs=[NodePatternConfig(op_type="WrongOp"), NodePatternConfig(op_type="WrongOp")])
    assert match_pattern(graph, "n1", pat_fail, {}) is False

    # recursive match pass
    pat_pass = NodePatternConfig(op_type="Op1", inputs=[NodePatternConfig(op_type="Input"), NodePatternConfig(op_type="Input")])
    assert match_pattern(graph, "n1", pat_pass, {}) is True

    from ml_switcheroo_compiler.transforms.passes.operator_fusion import FusionRule

    class DummyFusionRule(FusionRule):
        def __init__(self):
            self.pattern = None
            self.name = "dummy"

    strat2 = DummyFusionRule()
    assert strat2.apply(graph, {}) is None

    # PatternMatchingEngine coverage
    from ml_switcheroo_compiler.transforms.passes.operator_fusion import PatternMatchingEngine

    class FakeCostModel:
        def is_fusion_valid(self, *args, **kwargs):
            return True

    engine = PatternMatchingEngine([strat], cost_model=FakeCostModel())
    # trigger apply_passes
    engine.apply_passes(graph)

    class FakeCostModelFalse:
        def is_fusion_valid(self, *args, **kwargs):
            return False

    engine2 = PatternMatchingEngine([strat], cost_model=FakeCostModelFalse())
    engine2.apply_passes(graph)

    # Add id map test
    from ml_switcheroo_compiler.ir.core import IRGraph, IRNode

    class CustomRule(FusionRule):
        def __init__(self):
            super().__init__("custom", NodePatternConfig(op_type="A"))

        def apply(self, g, match):
            return {"n_a": IRNode(id="n_b", op_type="B", inputs=[])}

    g = IRGraph()
    g.nodes["n_a"] = IRNode(id="n_a", op_type="A", inputs=[])
    g.nodes["n_c"] = IRNode(id="n_c", op_type="C", inputs=["n_a"])
    g.outputs = ["n_a"]
    g.inputs = ["n_a"]

    eng = PatternMatchingEngine([CustomRule()])
    eng.apply_passes(g)
    assert "n_a" not in g.nodes
    assert "n_b" in g.nodes
    assert g.nodes["n_c"].inputs == ["n_b"]
    assert g.outputs == ["n_b"]
    assert g.inputs == ["n_b"]

    # Hit continue line 151
    g2 = IRGraph()
    g2.nodes["n_a"] = IRNode(id="n_a", op_type="A", inputs=[])
    eng2 = PatternMatchingEngine([CustomRule()], cost_model=FakeCostModelFalse())
    eng2.apply_passes(g2)
    assert "n_a" in g2.nodes

    # YamlFusionRule coverage
    cfg = FusionPatternConfig(pattern=NodePatternConfig(), replacement=ReplacementConfig(op_type="B", inputs=["val1", "val2"], capture_to_replace="target"))
    strat3 = YamlFusionRule("test3", cfg)

    match_dict = {"target": IRNode(id="n_a", op_type="A", inputs=[]), "val1": IRNode(id="n_1", op_type="In", inputs=[]), "val2": "val2_id"}
    res = strat3.apply(IRGraph(), match_dict)
    assert res is not None
    assert "n_a" in res
    assert res["n_a"].op_type == "B"
    assert res["n_a"].inputs == ["n_1", "val2_id"]
    assert strat3.apply(IRGraph(), {"target": "not_a_node"}) is None

    # apply_operator_fusion
    from unittest.mock import mock_open, patch

    from ml_switcheroo_compiler.transforms.passes.operator_fusion import _load_pass_config, apply_operator_fusion

    with patch("os.path.exists", return_value=True):
        import yaml

        with patch(
            "builtins.open",
            mock_open(
                read_data=yaml.dump(
                    {
                        "execution_order": [],
                        "cost_model": {"memory_costs": {}, "compute_costs": {}, "default_memory_cost": 0, "default_compute_cost": 0},
                        "fusion_patterns": {"dummy_rule": {"pattern": {"op_type": "Z", "capture": "x"}, "replacement": {"op_type": "Z2", "inputs": [], "capture_to_replace": "x"}}},
                    }
                )
            ),
        ):
            g3 = IRGraph()
            # To hit apply_passes(graph) returning True, we need a matching pattern
            g3.nodes["x"] = IRNode(id="x", op_type="Z", inputs=[])
            apply_operator_fusion(g3)
            cfg = _load_pass_config()
            assert cfg.fusion_patterns is not None

    with patch("os.path.exists", return_value=False):
        cfg2 = _load_pass_config()
        assert cfg2.fusion_patterns == {}
        # Hit 298->302 (config.fusion_patterns is empty)
        # Also hits 303->305 (engine.apply_passes returns False since no rules)
        g_empty = IRGraph()
        g_empty.nodes["x"] = IRNode(id="x", op_type="UnmatchedOp", inputs=[])
        apply_operator_fusion(g_empty)

    # test _build_pattern with inputs
    strat4 = YamlFusionRule("test4", FusionPatternConfig(pattern=NodePatternConfig(op_type="A", inputs=[NodePatternConfig(op_type="B")]), replacement=ReplacementConfig(op_type="x", inputs=[], capture_to_replace="x")))
    assert strat4.pattern.inputs is not None
