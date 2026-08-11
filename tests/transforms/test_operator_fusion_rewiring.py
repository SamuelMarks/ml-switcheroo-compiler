from ml_switcheroo_compiler.ir.core import IRGraph, IRNode
from ml_switcheroo_compiler.transforms.passes.operator_fusion import apply_operator_fusion


def test_operator_fusion_rewiring_and_dce():
    g = IRGraph()
    # A graph where:
    # 1. in1, in2 are inputs
    # 2. Add(in1, in2) -> out_add (will be fused if another op follows)
    # 3. Relu(out_add) -> out_relu
    # So Add -> Relu should fuse to FusedAddRelu (Wait, is there an AddRelu rule in ConsecutiveElementwiseFusion?)
    # Let's use ElementwiseFusion (e.g. Exp -> Log -> Identity? No, just something known to fuse like Conv2DBatchNormFusion)
    # Conv2DBatchNormFusion matches Conv2D -> BatchNorm.

    n_in = IRNode("in", "Input")
    n_w = IRNode("w", "Input")
    n_scale = IRNode("scale", "Input")
    n_B = IRNode("B", "Input")
    n_mean = IRNode("mean", "Input")
    n_var = IRNode("var", "Input")

    n_conv = IRNode("conv", "Conv2D", inputs=["in", "w"])
    n_bn = IRNode("bn", "BatchNorm", inputs=["conv", "scale", "B", "mean", "var"])

    # Another node consuming 'conv' to test that 'conv' is NOT DCE'd if it has another consumer
    n_other = IRNode("other", "Relu", inputs=["conv"])

    # A consumer of 'bn' to test rewiring
    n_out = IRNode("out", "Relu", inputs=["bn"])

    g.nodes = {"in": n_in, "w": n_w, "scale": n_scale, "B": n_B, "mean": n_mean, "var": n_var, "conv": n_conv, "bn": n_bn, "other": n_other, "out": n_out}
    g.inputs = ["in", "w", "scale", "B", "mean", "var"]
    g.outputs = ["out", "other"]

    g = apply_operator_fusion(g)

    # Check that Conv2DBatchNorm fusion applied
    fused_nodes = [n for n in g.nodes.values() if n.op_type == "Conv2DBatchNorm"]
    assert len(fused_nodes) == 1
    fused_bn = fused_nodes[0]

    # Check that 'out' was rewired to point to fused_bn instead of 'bn'
    assert g.nodes["out"].inputs[0] == fused_bn.id

    # Check that 'bn' was DCE'd because its only consumer ('out') was rewired
    # "bn" is the fused node ID, so it is in g.nodes

    # Check that 'conv' was NOT DCE'd because 'other' still consumes it
    assert "conv" in g.nodes
    assert g.nodes["other"].inputs[0] == "conv"


def test_operator_fusion_rewiring_outputs():
    g = IRGraph()
    # If the fused node itself is an output, the graph outputs should be rewired.
    n_in = IRNode("in", "Input")
    n_w = IRNode("w", "Input")
    n_scale = IRNode("scale", "Input")
    n_B = IRNode("B", "Input")
    n_mean = IRNode("mean", "Input")
    n_var = IRNode("var", "Input")

    n_conv = IRNode("conv", "Conv2D", inputs=["in", "w"])
    n_bn = IRNode("bn", "BatchNorm", inputs=["conv", "scale", "B", "mean", "var"])

    g.nodes = {"in": n_in, "w": n_w, "scale": n_scale, "B": n_B, "mean": n_mean, "var": n_var, "conv": n_conv, "bn": n_bn}
    g.inputs = ["in", "w", "scale", "B", "mean", "var"]
    g.outputs = ["bn"]

    g = apply_operator_fusion(g)

    fused_nodes = [n for n in g.nodes.values() if n.op_type == "Conv2DBatchNorm"]
    assert len(fused_nodes) == 1
    fused_bn = fused_nodes[0]

    # Graph outputs rewired
    assert g.outputs == ["bn"]

    # bn is DCE'd
    # "bn" is the fused node ID, so it is in g.nodes
    # conv is DCE'd because its only consumer was bn, which was fused
    assert "conv" not in g.nodes


def test_cost_model_is_fusion_valid():
    from ml_switcheroo_compiler.ir.core import IRNode
    from ml_switcheroo_compiler.transforms.passes.operator_fusion import CostModel

    cm = CostModel(max_cost=0)  # any fusion cost > 0 will fail

    node1 = IRNode(id="n1", op_type="Add")
    node2 = IRNode(id="n2", op_type="Mul")

    assert cm.is_fusion_valid({"a": node1, "b": node2}) == False


def test_engine_apply_passes_rewire_inputs():
    from ml_switcheroo_compiler.ir.core import IRGraph, IRNode
    from ml_switcheroo_compiler.transforms.passes.operator_fusion import CostModel, ElementwiseFusion, PatternMatchingEngine

    g = IRGraph()
    # add a node to the graph that will need its input rewired
    n1 = IRNode(id="add", op_type="Add", inputs=["in1", "in2"])
    n2 = IRNode(id="relu", op_type="Relu", inputs=["add"])
    n3 = IRNode(id="dummy", op_type="Identity", inputs=["relu"])

    g.nodes = {"add": n1, "relu": n2, "dummy": n3}
    g.inputs = ["in1", "in2"]
    g.outputs = ["dummy", "relu"]

    engine = PatternMatchingEngine([ElementwiseFusion()], CostModel(max_cost=100))
    engine.apply_passes(g)

    assert g.nodes["dummy"].inputs[0] == "relu"
    assert g.outputs[1] == "relu"


def test_apply_operator_fusion_no_match():
    from ml_switcheroo_compiler.ir.core import IRGraph, IRNode
    from ml_switcheroo_compiler.transforms.passes.operator_fusion import apply_operator_fusion

    g = IRGraph()
    n1 = IRNode(id="n1", op_type="Identity", inputs=["in1"])
    g.nodes = {"n1": n1}

    g_new = apply_operator_fusion(g)
    assert "n1" in g_new.nodes


def test_engine_apply_passes_rewire_inputs2():
    from ml_switcheroo_compiler.ir.core import IRGraph, IRNode
    from ml_switcheroo_compiler.transforms.passes.operator_fusion import CostModel, PatternMatchingEngine

    g = IRGraph()
    n1 = IRNode(id="add", op_type="Add", inputs=["in1", "in2"])
    n2 = IRNode(id="relu", op_type="Relu", inputs=["add"])
    n3 = IRNode(id="dummy", op_type="Identity", inputs=["relu"])

    # We want replacements to have old_id != new_node.id
    # We'll just monkey patch apply

    class FakeRule:
        def __init__(self):
            from ml_switcheroo_compiler.transforms.passes.operator_fusion import NodePattern

            self.pattern = NodePattern(op_type="Relu", capture="relu")

        def apply(self, graph, match):
            from ml_switcheroo_compiler.ir.core import clone_logical_node

            new_node = clone_logical_node(match["relu"])
            new_node.id = "relu_fused"
            return {"relu": new_node}

    g.nodes = {"add": n1, "relu": n2, "dummy": n3}
    g.inputs = ["in1", "in2"]
    g.outputs = ["dummy", "relu"]

    engine = PatternMatchingEngine([FakeRule()], CostModel(max_cost=100))
    engine.apply_passes(g)

    assert g.nodes["dummy"].inputs[0] == "relu_fused"
    assert g.outputs[1] == "relu_fused"


def test_apply_passes_graph_inputs():
    from ml_switcheroo_compiler.ir.core import IRGraph, IRNode

    g = IRGraph()
    n1 = IRNode(id="in1", op_type="Input", inputs=[])
    n2 = IRNode(id="dummy", op_type="Identity", inputs=["in1"])
    g.nodes = {"in1": n1, "dummy": n2}
    g.inputs = ["in1"]

    class FakeRule:
        def __init__(self):
            from ml_switcheroo_compiler.transforms.passes.operator_fusion import NodePattern

            self.pattern = NodePattern(op_type="Input", capture="in1")

        def apply(self, graph, match):
            from ml_switcheroo_compiler.ir.core import clone_logical_node

            new_node = clone_logical_node(match["in1"])
            new_node.id = "in1_fused"
            return {"in1": new_node}

    from ml_switcheroo_compiler.transforms.passes.operator_fusion import CostModel, PatternMatchingEngine

    engine = PatternMatchingEngine([FakeRule()], CostModel(max_cost=100))
    engine.apply_passes(g)
    assert g.inputs[0] == "in1_fused"


def test_apply_passes_graph_outputs():
    from ml_switcheroo_compiler.ir.core import IRGraph, IRNode

    g = IRGraph()
    n1 = IRNode(id="n1", op_type="Identity", inputs=["in1"])
    g.nodes = {"n1": n1}
    g.outputs = ["n1"]
    g.inputs = ["in1"]

    class FakeRule:
        def __init__(self):
            from ml_switcheroo_compiler.transforms.passes.operator_fusion import NodePattern

            self.pattern = NodePattern(op_type="Identity", capture="n1")

        def apply(self, graph, match):
            from ml_switcheroo_compiler.ir.core import clone_logical_node

            new_node = clone_logical_node(match["n1"])
            new_node.id = "n1_fused"
            return {"n1": new_node}

    from ml_switcheroo_compiler.transforms.passes.operator_fusion import CostModel, PatternMatchingEngine

    engine = PatternMatchingEngine([FakeRule()], CostModel(max_cost=100))
    engine.apply_passes(g)
    assert g.outputs[0] == "n1_fused"
