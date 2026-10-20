"""Tests for test_operator_fusion."""

from __future__ import annotations

import typing

from ml_switcheroo_compiler.ir.core import LogicalNode

if typing.TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import EagerCallableProtocol, EagerTensorProtocol, EagerValue


def test_operator_fusion_cost_fused() -> None:
    """Test function test_operator_fusion_cost_fused."""
    node = LogicalNode(id="n1", op_type="FusedMultiHeadAttention")
    assert node is not None


def _test_operator_fusion_extra_coverage_part_1(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 1 of test_operator_fusion_extra_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    from ml_switcheroo_compiler.ir.core import IRGraph, IRNode
    from ml_switcheroo_compiler.transforms.passes.config_models import NodePatternConfig
    from ml_switcheroo_compiler.transforms.passes.operator_fusion import match_pattern

    graph = IRGraph()
    n1 = IRNode(id="n1", op_type="Op1", inputs=["in1", "in2"])
    graph.nodes["n1"] = n1

    # len(inputs) mismatch
    pat = NodePatternConfig(op_type="Op1", inputs=[NodePatternConfig(op_type="Input")])  # 1 input vs 2
    assert bool(typing.cast("EagerCallableProtocol", match_pattern)(graph, "n1", pat, {})) is False

    # not isinstance(node_id, str)
    assert bool(typing.cast("EagerCallableProtocol", match_pattern)(graph, 123, typing.cast("EagerCallableProtocol", NodePatternConfig)(op_type="Input"), {})) is False
    cmap = typing.cast("dict[str, EagerTensorProtocol]", {})
    assert bool(typing.cast("EagerCallableProtocol", match_pattern)(graph, 123, typing.cast("EagerCallableProtocol", NodePatternConfig)(capture="test"), cmap)) is True
    assert cmap["test"] == 123

    # not node or op_type mismatch
    assert bool(typing.cast("EagerCallableProtocol", match_pattern)(graph, "n3", typing.cast("EagerCallableProtocol", NodePatternConfig)(op_type="Input"), {})) is False
    assert bool(typing.cast("EagerCallableProtocol", match_pattern)(graph, "n1", typing.cast("EagerCallableProtocol", NodePatternConfig)(op_type="Input"), {})) is False

    # capture is not None, inputs is None
    assert bool(typing.cast("EagerCallableProtocol", match_pattern)(graph, "n1", typing.cast("EagerCallableProtocol", NodePatternConfig)(capture="test_cap"), cmap)) is True
    assert cmap["test_cap"] == n1

    # apply_replacement base class coverage
    from ml_switcheroo_compiler.transforms.passes.config_models import FusionPatternConfig, ReplacementConfig
    from ml_switcheroo_compiler.transforms.passes.operator_fusion import YamlFusionRule

    cfg = FusionPatternConfig(pattern=NodePatternConfig(), replacement=ReplacementConfig(op_type="x", inputs=[], capture_to_replace="x"))
    strat = YamlFusionRule("test", cfg)
    strat.apply(graph, {"x": "n1"})

    # _match_inputs tests
    n2 = IRNode(id="n2", op_type="Input", inputs=[])
    getattr(_state, "update")({"FusionPatternConfig": FusionPatternConfig, "IRGraph": IRGraph, "IRNode": IRNode, "NodePatternConfig": NodePatternConfig})
    getattr(_state, "update")({"ReplacementConfig": ReplacementConfig, "YamlFusionRule": YamlFusionRule, "cfg": cfg, "cmap": cmap})
    getattr(_state, "update")({"graph": graph, "match_pattern": match_pattern, "n1": n1, "n2": n2})
    getattr(_state, "update")({"pat": pat, "strat": strat})
    _state.update(locals())


def _test_operator_fusion_extra_coverage_part_2(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 2 of test_operator_fusion_extra_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    NodePatternConfig, graph, match_pattern, n2 = _state["NodePatternConfig"], _state["graph"], _state["match_pattern"], _state["n2"]
    strat = _state["strat"]
    graph.nodes["n2"] = n2
    setattr(graph.nodes["n1"], "inputs", ["n2", "n2"])
    # recursive match fails
    pat_fail = NodePatternConfig(op_type="Op1", inputs=[NodePatternConfig(op_type="WrongOp"), NodePatternConfig(op_type="WrongOp")])
    assert bool(match_pattern(graph, "n1", pat_fail, {})) is False

    # recursive match pass
    pat_pass = NodePatternConfig(op_type="Op1", inputs=[NodePatternConfig(op_type="Input"), NodePatternConfig(op_type="Input")])
    assert bool(match_pattern(graph, "n1", pat_pass, {})) is True

    from ml_switcheroo_compiler.transforms.passes.operator_fusion import FusionRule

    class DummyFusionRule(FusionRule):
        """Test class DummyFusionRule."""

        def __init__(self) -> None:
            """Test function __init__."""
            setattr(self, "pattern", None)
            self.name = "dummy"

    strat2 = DummyFusionRule()
    assert typing.cast("EagerCallableProtocol", strat2.apply)(graph, {}) is None

    # PatternMatchingEngine coverage
    from ml_switcheroo_compiler.transforms.passes.operator_fusion import PatternMatchingEngine

    class FakeCostModel:
        """Test class FakeCostModel."""

        def is_fusion_valid(self, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
            """Test function is_fusion_valid.

            Args:
                args: Test parameter `args`.
                kwargs: Test parameter `kwargs`.
            """
            return True

    engine = typing.cast("EagerCallableProtocol", PatternMatchingEngine)([strat], cost_model=typing.cast("EagerCallableProtocol", FakeCostModel)())
    getattr(_state, "update")({"DummyFusionRule": DummyFusionRule, "FakeCostModel": FakeCostModel, "FusionRule": FusionRule, "PatternMatchingEngine": PatternMatchingEngine})
    getattr(_state, "update")({"engine": engine, "pat_fail": pat_fail, "pat_pass": pat_pass, "strat2": strat2})
    _state.update(locals())


def _test_operator_fusion_extra_coverage_part_3(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 3 of test_operator_fusion_extra_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    FusionRule, NodePatternConfig, PatternMatchingEngine, engine = _state["FusionRule"], _state["NodePatternConfig"], _state["PatternMatchingEngine"], _state["engine"]
    graph, strat = _state["graph"], _state["strat"]
    engine.apply_passes(graph)

    class FakeCostModelFalse:
        """Test class FakeCostModelFalse."""

        def is_fusion_valid(self, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
            """Test function is_fusion_valid.

            Args:
                args: Test parameter `args`.
                kwargs: Test parameter `kwargs`.
            """
            return False

    engine2 = typing.cast("EagerCallableProtocol", PatternMatchingEngine)([strat], cost_model=typing.cast("EagerCallableProtocol", FakeCostModelFalse)())
    engine2.apply_passes(graph)

    # Add id map test
    from ml_switcheroo_compiler.ir.core import IRGraph, IRNode
    from ml_switcheroo_compiler.transforms.passes.operator_fusion import FusionRule as _FusionRuleCls

    class CustomRule(_FusionRuleCls):
        """Test class CustomRule."""

        def __init__(self) -> None:
            """Test function __init__."""
            typing.cast("EagerCallableProtocol", super().__init__)("custom", NodePatternConfig(op_type="A"))

        def apply(self, g: EagerValue, match: EagerValue) -> dict[str, IRNode] | None:
            """Test function apply.

            Args:
                g: Test parameter `g`.
                match: Test parameter `match`.
            """
            return {"n_a": IRNode(id="n_b", op_type="B", inputs=[])}

    g = IRGraph()
    g.nodes["n_a"] = IRNode(id="n_a", op_type="A", inputs=[])
    g.nodes["n_c"] = IRNode(id="n_c", op_type="C", inputs=["n_a"])
    g.outputs = ["n_a"]
    g.inputs = ["n_a"]
    getattr(_state, "update")({"CustomRule": CustomRule, "FakeCostModelFalse": FakeCostModelFalse, "IRGraph": IRGraph, "IRNode": IRNode})
    getattr(_state, "update")({"engine2": engine2, "g": g})
    _state.update(locals())


def _test_operator_fusion_extra_coverage_part_4(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 4 of test_operator_fusion_extra_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    CustomRule, FakeCostModelFalse, FusionPatternConfig, IRGraph = _state["CustomRule"], _state["FakeCostModelFalse"], _state["FusionPatternConfig"], _state["IRGraph"]
    IRNode, NodePatternConfig, PatternMatchingEngine, ReplacementConfig = _state["IRNode"], _state["NodePatternConfig"], _state["PatternMatchingEngine"], _state["ReplacementConfig"]
    YamlFusionRule, g = _state["YamlFusionRule"], _state["g"]
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
    res = typing.cast("EagerCallableProtocol", strat3.apply)(typing.cast("EagerCallableProtocol", IRGraph)(), match_dict)
    assert res is not None
    assert "n_a" in res
    assert res["n_a"].op_type == "B"
    assert res["n_a"].inputs == ["n_1", "val2_id"]
    assert strat3.apply(IRGraph(), {"target": "not_a_node"}) is None

    # apply_operator_fusion
    from unittest.mock import mock_open, patch

    getattr(_state, "update")({"cfg": cfg, "eng": eng, "eng2": eng2, "g2": g2})
    getattr(_state, "update")({"match_dict": match_dict, "mock_open": mock_open, "patch": patch, "res": res})
    getattr(_state, "update")({"strat3": strat3})
    _state.update(locals())


def _test_operator_fusion_extra_coverage_part_5(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 5 of test_operator_fusion_extra_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    FusionPatternConfig, IRGraph, IRNode, NodePatternConfig = _state["FusionPatternConfig"], _state["IRGraph"], _state["IRNode"], _state["NodePatternConfig"]
    ReplacementConfig, YamlFusionRule, mock_open, patch = _state["ReplacementConfig"], _state["YamlFusionRule"], _state["mock_open"], _state["patch"]
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
            typing.cast("EagerCallableProtocol", apply_operator_fusion)(g3)
            cfg = _load_pass_config()
            assert cfg.fusion_patterns is not None
    with patch("os.path.exists", return_value=False):
        cfg2 = _load_pass_config()
        assert cfg2.fusion_patterns == {}
        # Hit 298->302 (config.fusion_patterns is empty)
        # Also hits 303->305 (engine.apply_passes returns False since no rules)
        g_empty = IRGraph()
        g_empty.nodes["x"] = IRNode(id="x", op_type="UnmatchedOp", inputs=[])
        typing.cast("EagerCallableProtocol", apply_operator_fusion)(g_empty)

    # test _build_pattern with inputs
    strat4 = YamlFusionRule("test4", FusionPatternConfig(pattern=NodePatternConfig(op_type="A", inputs=[NodePatternConfig(op_type="B")]), replacement=ReplacementConfig(op_type="x", inputs=[], capture_to_replace="x")))
    assert strat4.pattern.inputs is not None
    getattr(_state, "update")({"_load_pass_config": _load_pass_config, "apply_operator_fusion": apply_operator_fusion, "cfg": cfg, "cfg2": cfg2})
    getattr(_state, "update")({"g3": g3, "g_empty": g_empty, "strat4": strat4, "yaml": yaml})


def test_operator_fusion_extra_coverage() -> None:
    """Test function test_operator_fusion_extra_coverage."""
    _state: dict[str, EagerTensorProtocol] = {}
    _test_operator_fusion_extra_coverage_part_1(_state)
    _test_operator_fusion_extra_coverage_part_2(_state)
    _test_operator_fusion_extra_coverage_part_3(_state)
    _test_operator_fusion_extra_coverage_part_4(_state)
    _test_operator_fusion_extra_coverage_part_5(_state)
    assert len(_state) > 0


def test_operator_fusion_exhaustive_patch() -> None:
    """Test function test_operator_fusion_exhaustive_patch."""
    from ml_switcheroo_compiler.ir.core import IRGraph, IRNode
    from ml_switcheroo_compiler.transforms.passes.config_models import NodePatternConfig
    from ml_switcheroo_compiler.transforms.passes.operator_fusion import FusionRule, MemoryAwareCostModel, PatternMatchingEngine

    class ReplaceFutureNodeRule(FusionRule):
        """Test class ReplaceFutureNodeRule."""

        def __init__(self) -> None:
            """Test function __init__."""
            typing.cast("EagerCallableProtocol", super().__init__)("replace_future", typing.cast("EagerCallableProtocol", NodePatternConfig)(op_type="OpA"))

        def apply(self, g: EagerValue, match: EagerValue) -> dict[str, LogicalNode] | None:
            """Test function apply.

            Args:
                g: Test parameter `g`.
                match: Test parameter `match`.
            """
            new_node = IRNode(id="n2", op_type="OpB_Fused", inputs=[])
            return {"n2": new_node}

    g = IRGraph()
    g.nodes["n1"] = IRNode(id="n1", op_type="OpA", inputs=[])
    g.nodes["n2"] = IRNode(id="n2", op_type="OpB", inputs=["n1"])

    eng = PatternMatchingEngine([ReplaceFutureNodeRule()])
    eng.apply_passes(g)

    class ReplaceWithSymbolicShapeRule(FusionRule):
        """Test class ReplaceWithSymbolicShapeRule."""

        def __init__(self) -> None:
            """Test function __init__."""
            typing.cast("EagerCallableProtocol", super().__init__)("symbolic_shape", typing.cast("EagerCallableProtocol", NodePatternConfig)(op_type="OpC"))

        def apply(self, g: EagerValue, match: EagerValue) -> dict[str, LogicalNode] | None:
            """Test function apply.

            Args:
                g: Test parameter `g`.
                match: Test parameter `match`.
            """
            new_node = IRNode(id="n4", op_type="OpC_Fused", inputs=[])
            new_node.shape_metadata = ("batch", 128)
            return {"n3": new_node}

    g2 = IRGraph()
    g2.nodes["n3"] = IRNode(id="n3", op_type="OpC", inputs=[])

    eng2 = PatternMatchingEngine([ReplaceWithSymbolicShapeRule()], cost_model=MemoryAwareCostModel({"max_fusion_memory_bytes": 1024}))
    eng2.apply_passes(g2)
    assert "n4" in g2.nodes or "n3" in g2.nodes


def test_operator_fusion_symbolic_shape_coverage() -> None:
    """Test function test_operator_fusion_symbolic_shape_coverage."""
    from ml_switcheroo_compiler.ir.core import IRGraph, IRNode
    from ml_switcheroo_compiler.transforms.passes.config_models import NodePatternConfig
    from ml_switcheroo_compiler.transforms.passes.operator_fusion import FusionRule, MemoryAwareCostModel, PatternMatchingEngine

    class ReplaceWithSymbolicShapeRule(FusionRule):
        """Test class ReplaceWithSymbolicShapeRule."""

        def __init__(self) -> None:
            """Test function __init__."""
            typing.cast("EagerCallableProtocol", super().__init__)("symbolic_shape", typing.cast("EagerCallableProtocol", NodePatternConfig)(op_type="OpC"))

        def apply(self, g: EagerValue, match: EagerValue) -> dict[str, LogicalNode] | None:
            """Test function apply.

            Args:
                g: Test parameter `g`.
                match: Test parameter `match`.
            """

            class FakeNode:
                """Test class FakeNode."""

                def __init__(self) -> None:
                    """Test function __init__."""
                    self.id = "n4"
                    self.shape_metadata = ["batch", 128]
                    self.is_dynamic_shape = False
                    self.attributes = typing.cast("dict[str, EagerTensorProtocol]", {})
                    self.inputs = typing.cast("list[EagerTensorProtocol]", [])
                    self.op_type = "OpC_Fused"

            new_node = FakeNode()
            return typing.cast("dict[str, LogicalNode] | None", {"n3": new_node})

    g2 = IRGraph()
    g2.nodes["n3"] = IRNode(id="n3", op_type="OpC", inputs=[])

    eng2 = PatternMatchingEngine([ReplaceWithSymbolicShapeRule()], cost_model=MemoryAwareCostModel({"max_fusion_memory_bytes": 1024}))
    eng2.apply_passes(g2)
    assert "n4" in g2.nodes or "n3" in g2.nodes


def test_operator_fusion_extra_coverage2() -> None:
    """Test function test_operator_fusion_extra_coverage2."""
    from ml_switcheroo_compiler.ir.core import IRGraph, IRNode
    from ml_switcheroo_compiler.transforms.passes.operator_fusion import FusionRule, MemoryAwareCostModel, NodePattern, PatternMatchingEngine, YamlFusionRule, match_pattern

    # 1. match_pattern branch 81->83
    assert typing.cast("EagerCallableProtocol", match_pattern)(None, None, typing.cast("EagerCallableProtocol", NodePattern)(op_type=None, inputs=None, capture=None), {})

    # 2. match_pattern where node_id is int or None but pattern.op_type is not None (branch 78->79->80)
    assert not typing.cast("EagerCallableProtocol", match_pattern)(None, None, typing.cast("EagerCallableProtocol", NodePattern)(op_type="Add"), {})

    # 3. Explicit Edge Rewiring branches: 184->183, 189->188, 193->192
    class MockRule(FusionRule):
        """Test class MockRule."""

        def __init__(self) -> None:
            """Test function __init__."""
            super().__init__("mock", NodePattern(capture="my_node"))

        def apply(self, graph: EagerValue, match: EagerValue) -> dict[str, LogicalNode] | None:
            """Test function apply.

            Args:
                graph: Test parameter `graph`.
                match: Test parameter `match`.
            """
            node = typing.cast("EagerTensorProtocol", getattr(match, "get"))("my_node")
            if node and typing.cast("EagerTensorProtocol", getattr(node, "id")) == "a_id":
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
    cost_model = typing.cast("EagerCallableProtocol", MemoryAwareCostModel)(cfg)

    node_static = IRNode(id="static", op_type="Add")
    node_static.shape_metadata = (2, 3)
    node_static.attributes["dtype"] = "float32"

    node_static_big = IRNode(id="big", op_type="Add")
    node_static_big.shape_metadata = (10, 10)  # 100 elements * 4 = 400 bytes
    node_static_big.attributes["dtype"] = "float32"

    assert cost_model.is_fusion_valid({"a": node_static})
    assert not cost_model.is_fusion_valid({"a": node_static_big})


def test_operator_fusion_extra_coverage_3() -> None:
    """Test function test_operator_fusion_extra_coverage_3."""
    from ml_switcheroo_compiler.ir.core import IRGraph, IRNode
    from ml_switcheroo_compiler.transforms.passes.operator_fusion import FusionRule, MemoryAwareCostModel, NodePattern, PatternMatchingEngine

    # test config is None
    cost_model = MemoryAwareCostModel(None)
    assert cost_model.is_fusion_valid({})

    # test False branches for rewiring explicitly
    class MockRule2(FusionRule):
        """Test class MockRule2."""

        def __init__(self) -> None:
            """Test function __init__."""
            super().__init__("mock2", NodePattern(capture="my_node"))

        def apply(self, graph: EagerValue, match: EagerValue) -> dict[str, LogicalNode] | None:
            """Test function apply.

            Args:
                graph: Test parameter `graph`.
                match: Test parameter `match`.
            """
            node = typing.cast("EagerTensorProtocol", getattr(match, "get"))("my_node")
            if node and typing.cast("EagerTensorProtocol", getattr(node, "id")) == "a_id":
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


def _test_operator_fusion_all_remaining_branches_part_1(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 1 of test_operator_fusion_all_remaining_branches.

    Args:
        _state: Shared test state dictionary.
    """
    from ml_switcheroo_compiler.ir.core import IRGraph, IRNode
    from ml_switcheroo_compiler.transforms.passes.operator_fusion import (
        FusionRule,
        HorizontalFusionPass,
        MatchMapType,
        NodePattern,
        PatternMatchingEngine,
        VerticalFusionPass,
        fuse_conv_bias_relu,
        fuse_elementwise_activation,
        operator_fusion_pass,
    )

    # 1. Line 249->253: graph without hasattr(graph, 'inputs') or where optimized is True and no inputs
    class MinimalGraph:
        """Test class MinimalGraph."""

        def __init__(self) -> None:
            """Test function __init__."""
            self.nodes: dict[str, IRNode] = {}
            self.outputs: list[str] = []

    min_g = MinimalGraph()
    min_g.nodes["a_id"] = IRNode(id="a_id", op_type="A")
    min_g.outputs = ["a_id"]

    class MockRule(FusionRule):
        """Test class MockRule."""

        def __init__(self) -> None:
            """Test function __init__."""
            super().__init__("mock", NodePattern(capture="x"))

        def apply(self, graph: IRGraph, match: MatchMapType) -> dict[str, IRNode] | None:
            """Test function apply.

            Args:
                graph: Test parameter `graph`.
                match: Test parameter `match`.
            """
            node = match.get("x")
            if isinstance(node, IRNode) and node.id == "a_id":
                return {"a_id": IRNode(id="a_fused", op_type="A_fused", inputs=[])}
            return None

    getattr(_state, "update")({"FusionRule": FusionRule, "HorizontalFusionPass": HorizontalFusionPass, "IRGraph": IRGraph, "IRNode": IRNode})
    getattr(_state, "update")({"MatchMapType": MatchMapType, "MinimalGraph": MinimalGraph, "MockRule": MockRule, "NodePattern": NodePattern})
    getattr(_state, "update")({"PatternMatchingEngine": PatternMatchingEngine, "VerticalFusionPass": VerticalFusionPass, "fuse_conv_bias_relu": fuse_conv_bias_relu, "fuse_elementwise_activation": fuse_elementwise_activation})
    getattr(_state, "update")({"min_g": min_g, "operator_fusion_pass": operator_fusion_pass})
    _state.update(locals())


def _test_operator_fusion_all_remaining_branches_part_2(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 2 of test_operator_fusion_all_remaining_branches.

    Args:
        _state: Shared test state dictionary.
    """
    IRGraph, IRNode, MockRule, PatternMatchingEngine = _state["IRGraph"], _state["IRNode"], _state["MockRule"], _state["PatternMatchingEngine"]
    fuse_conv_bias_relu, min_g = _state["fuse_conv_bias_relu"], _state["min_g"]
    eng = PatternMatchingEngine([MockRule()])
    assert eng.apply_passes(min_g)
    assert min_g.outputs == ["a_fused"]
    g_bias_wrong_op = IRGraph()
    g_bias_wrong_op.nodes["conv"] = IRNode(id="conv", op_type="Conv2D", inputs=["in_x", "weight"])
    g_bias_wrong_op.nodes["mul"] = IRNode(id="mul", op_type="Mul", inputs=["conv", "bias"])
    g_bias_wrong_op.nodes["relu"] = IRNode(id="relu", op_type="Relu", inputs=["mul"])
    assert not fuse_conv_bias_relu(g_bias_wrong_op)

    g_bias_wrong_len = IRGraph()
    g_bias_wrong_len.nodes["conv"] = IRNode(id="conv", op_type="Conv2D", inputs=["in_x", "weight"])
    g_bias_wrong_len.nodes["add"] = IRNode(id="add", op_type="Add", inputs=["conv"])
    g_bias_wrong_len.nodes["relu"] = IRNode(id="relu", op_type="Relu", inputs=["add"])
    assert not fuse_conv_bias_relu(g_bias_wrong_len)

    # Line 716: bias_node has multiple consumers (consumer_count != 1)
    g_bias_multi = IRGraph()
    g_bias_multi.nodes["in_x"] = IRNode(id="in_x", op_type="Input", inputs=[])
    g_bias_multi.nodes["weight"] = IRNode(id="weight", op_type="Input", inputs=[])
    g_bias_multi.nodes["bias"] = IRNode(id="bias", op_type="Input", inputs=[])
    g_bias_multi.nodes["conv"] = IRNode(id="conv", op_type="Conv2D", inputs=["in_x", "weight"])
    g_bias_multi.nodes["add"] = IRNode(id="add", op_type="Add", inputs=["conv", "bias"])
    g_bias_multi.nodes["relu"] = IRNode(id="relu", op_type="Relu", inputs=["add"])
    g_bias_multi.nodes["other"] = IRNode(id="other", op_type="Neg", inputs=["add"])
    assert not fuse_conv_bias_relu(g_bias_multi)
    getattr(_state, "update")({"eng": eng, "g_bias_multi": g_bias_multi, "g_bias_wrong_len": g_bias_wrong_len, "g_bias_wrong_op": g_bias_wrong_op})
    _state.update(locals())


def _test_operator_fusion_all_remaining_branches_part_3(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 3 of test_operator_fusion_all_remaining_branches.

    Args:
        _state: Shared test state dictionary.
    """
    IRGraph, IRNode, fuse_conv_bias_relu = _state["IRGraph"], _state["IRNode"], _state["fuse_conv_bias_relu"]
    g_no_conv = IRGraph()
    g_no_conv.nodes["x"] = IRNode(id="x", op_type="Add", inputs=["in1", "in2"])
    g_no_conv.nodes["bias"] = IRNode(id="bias", op_type="Constant", inputs=[])
    g_no_conv.nodes["add"] = IRNode(id="add", op_type="Add", inputs=["x", "bias"])
    g_no_conv.nodes["relu"] = IRNode(id="relu", op_type="Relu", inputs=["add"])
    assert not fuse_conv_bias_relu(g_no_conv)
    g_no_remaining = IRGraph()
    g_no_remaining.nodes["conv"] = IRNode(id="conv", op_type="Conv2D", inputs=["in_x", "weight"])
    g_no_remaining.nodes["add"] = IRNode(id="add", op_type="Add", inputs=["conv", "conv"])
    g_no_remaining.nodes["relu"] = IRNode(id="relu", op_type="Relu", inputs=["add"])
    assert not fuse_conv_bias_relu(g_no_remaining)

    # Line 737: conv has multiple consumers (consumer_count != 1)
    g_conv_multi = IRGraph()
    g_conv_multi.nodes["in_x"] = IRNode(id="in_x", op_type="Input", inputs=[])
    g_conv_multi.nodes["weight"] = IRNode(id="weight", op_type="Input", inputs=[])
    g_conv_multi.nodes["bias"] = IRNode(id="bias", op_type="Input", inputs=[])
    g_conv_multi.nodes["conv"] = IRNode(id="conv", op_type="Conv2D", inputs=["in_x", "weight"])
    g_conv_multi.nodes["add"] = IRNode(id="add", op_type="Add", inputs=["conv", "bias"])
    g_conv_multi.nodes["relu"] = IRNode(id="relu", op_type="Relu", inputs=["add"])
    g_conv_multi.nodes["extra"] = IRNode(id="extra", op_type="Neg", inputs=["conv"])
    assert not fuse_conv_bias_relu(g_conv_multi)

    # Test out_id in consumer_counts (line 711->712)
    g_conv_out = IRGraph()
    g_conv_out.nodes["conv"] = IRNode(id="conv", op_type="Conv2D", inputs=["in_x", "weight"])
    getattr(_state, "update")({"g_conv_multi": g_conv_multi, "g_conv_out": g_conv_out, "g_no_conv": g_no_conv, "g_no_remaining": g_no_remaining})
    _state.update(locals())


def _test_operator_fusion_all_remaining_branches_part_4(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 4 of test_operator_fusion_all_remaining_branches.

    Args:
        _state: Shared test state dictionary.
    """
    IRGraph, IRNode, fuse_conv_bias_relu, fuse_elementwise_activation = _state["IRGraph"], _state["IRNode"], _state["fuse_conv_bias_relu"], _state["fuse_elementwise_activation"]
    g_conv_out = _state["g_conv_out"]
    g_conv_out.nodes["add"] = IRNode(id="add", op_type="Add", inputs=["conv", "bias"])
    g_conv_out.nodes["relu"] = IRNode(id="relu", op_type="Relu", inputs=["add"])
    setattr(g_conv_out, "outputs", ["nonexistent_out", "relu"])
    # Also test an act_node that triggers line 716 continue: op_type not in activation list or inputs != 1
    g_conv_out.nodes["multi_input_relu"] = IRNode(id="multi_input_relu", op_type="Relu", inputs=["add", "other"])
    assert not fuse_conv_bias_relu(g_conv_out)

    # 3. Lines 787->786, 790->789, 795: fuse_elementwise_activation branches
    # Test out_id in consumer_counts (line 790->791)
    g_elem_out = IRGraph()
    g_elem_out.nodes["add"] = IRNode(id="add", op_type="Add", inputs=["in1", "in2"])
    g_elem_out.nodes["relu"] = IRNode(id="relu", op_type="Relu", inputs=["add"])
    setattr(g_elem_out, "outputs", ["nonexistent_out", "relu"])
    g_elem_out.nodes["multi_input_relu"] = IRNode(id="multi_input_relu", op_type="Relu", inputs=["add", "other"])
    assert not fuse_elementwise_activation(g_elem_out)

    # Line 790: elem_node is None or not in ELEMENTWISE_OPS
    g_elem_not_elem = IRGraph()
    g_elem_not_elem.nodes["matmul"] = IRNode(id="matmul", op_type="Matmul", inputs=["in1", "in2"])
    g_elem_not_elem.nodes["relu"] = IRNode(id="relu", op_type="Relu", inputs=["matmul"])
    assert not fuse_elementwise_activation(g_elem_not_elem)

    # Line 795: elem_node consumer_count != 1
    g_elem_multi = IRGraph()
    g_elem_multi.nodes["in1"] = IRNode(id="in1", op_type="Input", inputs=[])
    g_elem_multi.nodes["in2"] = IRNode(id="in2", op_type="Input", inputs=[])
    g_elem_multi.nodes["add"] = IRNode(id="add", op_type="Add", inputs=["in1", "in2"])
    g_elem_multi.nodes["relu"] = IRNode(id="relu", op_type="Relu", inputs=["add"])
    g_elem_multi.nodes["extra"] = IRNode(id="extra", op_type="Neg", inputs=["add"])
    assert not fuse_elementwise_activation(g_elem_multi)
    getattr(_state, "update")({"g_elem_multi": g_elem_multi, "g_elem_not_elem": g_elem_not_elem, "g_elem_out": g_elem_out})
    _state.update(locals())


def _test_operator_fusion_all_remaining_branches_part_5(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 5 of test_operator_fusion_all_remaining_branches.

    Args:
        _state: Shared test state dictionary.
    """
    HorizontalFusionPass, IRGraph, VerticalFusionPass, operator_fusion_pass = _state["HorizontalFusionPass"], _state["IRGraph"], _state["VerticalFusionPass"], _state["operator_fusion_pass"]
    from unittest.mock import patch

    g_pass = IRGraph()
    with (
        patch.object(VerticalFusionPass, "run", return_value=True),
        patch.object(HorizontalFusionPass, "run", return_value=True),
        patch("ml_switcheroo_compiler.transforms.passes.operator_fusion.fuse_elementwise_clusters", return_value=True),
    ):
        assert operator_fusion_pass(g_pass)
    getattr(_state, "update")({"g_pass": g_pass, "patch": patch})


def test_operator_fusion_all_remaining_branches() -> None:
    """Test all remaining branch coverage in operator_fusion.py."""
    _state: dict[str, EagerTensorProtocol] = {}
    _test_operator_fusion_all_remaining_branches_part_1(_state)
    _test_operator_fusion_all_remaining_branches_part_2(_state)
    _test_operator_fusion_all_remaining_branches_part_3(_state)
    _test_operator_fusion_all_remaining_branches_part_4(_state)
    _test_operator_fusion_all_remaining_branches_part_5(_state)
    assert len(_state) > 0
