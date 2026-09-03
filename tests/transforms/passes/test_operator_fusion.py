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
                        "cost_model": {"memory_sizes": {}, "compute_costs": {"heavy_ops": [], "light_ops": [], "heavy_cost": 1, "light_cost": 1, "default_cost": 1}, "compute_heavy_threshold": 1, "heavy_interleave_penalty": 1, "light_interleave_penalty": 1},
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


def test_operator_fusion_exhaustive_patch():
    from ml_switcheroo_compiler.ir.core import IRGraph, IRNode
    from ml_switcheroo_compiler.transforms.passes.config_models import NodePatternConfig
    from ml_switcheroo_compiler.transforms.passes.operator_fusion import FusionRule, MemoryAwareCostModel, PatternMatchingEngine

    class ReplaceFutureNodeRule(FusionRule):
        def __init__(self):
            super().__init__("replace_future", NodePatternConfig(op_type="OpA"))

        def apply(self, g, match):
            new_node = IRNode(id="n2", op_type="OpB_Fused", inputs=[])
            return {"n2": new_node}

    g = IRGraph()
    g.nodes["n1"] = IRNode(id="n1", op_type="OpA", inputs=[])
    g.nodes["n2"] = IRNode(id="n2", op_type="OpB", inputs=["n1"])

    eng = PatternMatchingEngine([ReplaceFutureNodeRule()])
    eng.apply_passes(g)

    class ReplaceWithSymbolicShapeRule(FusionRule):
        def __init__(self):
            super().__init__("symbolic_shape", NodePatternConfig(op_type="OpC"))

        def apply(self, g, match):
            new_node = IRNode(id="n4", op_type="OpC_Fused", inputs=[])
            new_node.shape_metadata = ["batch", 128]
            return {"n3": new_node}

    g2 = IRGraph()
    g2.nodes["n3"] = IRNode(id="n3", op_type="OpC", inputs=[])

    eng2 = PatternMatchingEngine([ReplaceWithSymbolicShapeRule()], cost_model=MemoryAwareCostModel({"max_fusion_memory_bytes": 1024}))
    eng2.apply_passes(g2)


def test_operator_fusion_symbolic_shape_coverage():
    from ml_switcheroo_compiler.ir.core import IRGraph, IRNode
    from ml_switcheroo_compiler.transforms.passes.config_models import NodePatternConfig
    from ml_switcheroo_compiler.transforms.passes.operator_fusion import FusionRule, MemoryAwareCostModel, PatternMatchingEngine

    class ReplaceWithSymbolicShapeRule(FusionRule):
        def __init__(self):
            super().__init__("symbolic_shape", NodePatternConfig(op_type="OpC"))

        def apply(self, g, match):
            class FakeNode:
                def __init__(self):
                    self.id = "n4"
                    self.shape_metadata = ["batch", 128]
                    self.is_dynamic_shape = False
                    self.attributes = {}
                    self.inputs = []
                    self.op_type = "OpC_Fused"

            new_node = FakeNode()
            return {"n3": new_node}

    g2 = IRGraph()
    g2.nodes["n3"] = IRNode(id="n3", op_type="OpC", inputs=[])

    eng2 = PatternMatchingEngine([ReplaceWithSymbolicShapeRule()], cost_model=MemoryAwareCostModel({"max_fusion_memory_bytes": 1024}))
    eng2.apply_passes(g2)


def test_operator_fusion_extra_coverage2():
    from ml_switcheroo_compiler.ir.core import IRGraph, IRNode
    from ml_switcheroo_compiler.transforms.passes.operator_fusion import FusionRule, MemoryAwareCostModel, NodePattern, PatternMatchingEngine, YamlFusionRule, match_pattern

    # 1. match_pattern branch 81->83
    assert match_pattern(None, None, NodePattern(op_type=None, inputs=None, capture=None), {}) == True

    # 2. match_pattern where node_id is int or None but pattern.op_type is not None (branch 78->79->80)
    assert match_pattern(None, None, NodePattern(op_type="Add"), {}) == False

    # 3. Explicit Edge Rewiring branches: 184->183, 189->188, 193->192
    class MockRule(FusionRule):
        def __init__(self):
            super().__init__("mock", NodePattern(capture="my_node"))

        def apply(self, graph, match):
            node = match.get("my_node")
            if node and node.id == "a_id":
                new_n = IRNode(id="a_fused", op_type="A_fused")
                return {"a_id": new_n}
            return None

    g = IRGraph()
    g.inputs = ["missing_input", "a_id"]
    g.outputs = ["missing_output", "a_id"]

    n_a = IRNode(id="a_id", op_type="A", inputs=["missing_node_input"])
    g.nodes["a_id"] = n_a
    # Put a random node that will not be matched but needs its inputs rewired partially
    n_b = IRNode(id="b_id", op_type="B", inputs=["a_id", "missing_node_input"])
    g.nodes["b_id"] = n_b

    engine = PatternMatchingEngine([MockRule()])
    engine.apply_passes(g)

    # 4. Replacement node input not str or IRNode (YamlFusionRule)
    # create a mock yaml dict that returns invalid capture
    from ml_switcheroo_compiler.transforms.passes.config_models import FusionPatternConfig, NodePatternConfig, ReplacementConfig

    cfg_pydantic = FusionPatternConfig(pattern=NodePatternConfig(op_type="X", capture="x"), replacement=ReplacementConfig(op_type="Y", inputs=["invalid_capture"], capture_to_replace="x"))
    yrule = YamlFusionRule("yaml_mock", cfg_pydantic)
    g2 = IRGraph()
    g2.nodes["x_id"] = IRNode(id="x_id", op_type="X")
    eng2 = PatternMatchingEngine([yrule])
    eng2.apply_passes(g2)

    # 5. MemoryAwareCostModel shape calculations
    cfg = {"max_fusion_memory_bytes": 100, "memory_sizes": {"float32": 4}}
    cost_model = MemoryAwareCostModel(cfg)

    node_static = IRNode(id="static", op_type="Add")
    node_static.shape_metadata = (2, 3)
    node_static.attributes["dtype"] = "float32"

    node_static_big = IRNode(id="big", op_type="Add")
    node_static_big.shape_metadata = (10, 10)  # 100 elements * 4 = 400 bytes
    node_static_big.attributes["dtype"] = "float32"

    assert cost_model.is_fusion_valid({"a": node_static}) == True
    assert cost_model.is_fusion_valid({"a": node_static_big}) == False


def test_operator_fusion_extra_coverage_3():
    from ml_switcheroo_compiler.ir.core import IRGraph, IRNode
    from ml_switcheroo_compiler.transforms.passes.operator_fusion import FusionRule, MemoryAwareCostModel, NodePattern, PatternMatchingEngine

    # test config is None
    cost_model = MemoryAwareCostModel(None)
    assert cost_model.is_fusion_valid({}) == True

    # test False branches for rewiring explicitly
    class MockRule2(FusionRule):
        def __init__(self):
            super().__init__("mock2", NodePattern(capture="my_node"))

        def apply(self, graph, match):
            node = match.get("my_node")
            if node and node.id == "a_id":
                new_n = IRNode(id="a_fused", op_type="A_fused", inputs=["something_else"])
                return {"a_id": new_n}
            return None

    g = IRGraph()
    g.inputs = ["missing_input", "a_id", "another_missing"]
    g.outputs = ["missing_output", "a_id", "another_missing_out"]
    g.nodes["a_id"] = IRNode(id="a_id", op_type="A")
    g.nodes["b_id"] = IRNode(id="b_id", op_type="B", inputs=["a_id", "missing_node_input", "a_id"])

    engine = PatternMatchingEngine([MockRule2()])
    engine.apply_passes(g)
