from ml_switcheroo_compiler.ir.core import IRGraph, IRNode
from ml_switcheroo_compiler.transforms.passes.operator_fusion import (
    FusionRule,
    NodePattern,
    PatternMatchingEngine,
    estimate_node_cost,
    match_pattern,
)


def test_operator_fusion_extra_coverage():
    g = IRGraph()
    # 1. pattern capture=None for non-string input
    p = NodePattern(op_type=None, capture=None)
    cap = {}
    assert match_pattern(g, 123, p, cap)

    # 2. rule apply returns {}
    class EmptyRule(FusionRule):
        def __init__(self):
            super().__init__("EmptyRule", NodePattern(op_type="Dummy"))

        def apply(self, graph, capture_map):
            return {}

    n1 = IRNode("n1", "Dummy")
    g.nodes["n1"] = n1
    op_pass = PatternMatchingEngine([EmptyRule()])
    op_pass.apply_passes(g)

    # 3. downstream node replaced
    class DownstreamRule(FusionRule):
        def __init__(self):
            super().__init__("Downstream", NodePattern(op_type="A"))

        def apply(self, graph, capture_map):
            # when matching A, we replace B as well, putting B in new_nodes
            n = IRNode("b", "ReplacedB")
            return {"a": IRNode("a", "ReplacedA"), "b": n}

    g = IRGraph()
    g.nodes["a"] = IRNode("a", "A")
    g.nodes["b"] = IRNode("b", "B")

    op_pass = PatternMatchingEngine([DownstreamRule()])
    op_pass.apply_passes(g)
    assert g.nodes["b"].op_type == "ReplacedB"

    # 4. Fused estimate cost
    n_fused = IRNode("fused", "FusedOp")
    assert estimate_node_cost(n_fused) == 2
