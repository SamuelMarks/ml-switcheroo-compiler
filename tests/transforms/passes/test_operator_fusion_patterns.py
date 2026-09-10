def test_operator_fusion_extra():
    from ml_switcheroo_compiler.ir.core import IRNode
    from ml_switcheroo_compiler.transforms.passes.operator_fusion import MemoryAwareCostModel

    # Missing 295
    model = MemoryAwareCostModel(None)
    assert model.is_fusion_valid({}) is True

    # Missing 307-313
    model = MemoryAwareCostModel({"max_fusion_memory_bytes": 100, "memory_sizes": {"float32": 4}})
    node = IRNode("n", "Add", shape_metadata=(10, 10))
    # 10 * 10 * 4 = 400 > 100 -> False
    assert model.is_fusion_valid({"n": node}) is False

    node2 = IRNode("n2", "Add", shape_metadata=(2, 2))
    # 2 * 2 * 4 = 16 <= 100 -> True
    assert model.is_fusion_valid({"n2": node2}) is True


def test_discover_and_apply_new_fusion_patterns():
    from ml_switcheroo_compiler.ir.core import IRGraph, IRNode
    from ml_switcheroo_compiler.transforms.passes.operator_fusion import (
        _discover_fusion_patterns,
        apply_operator_fusion,
    )

    rules = _discover_fusion_patterns()
    rule_names = [r.name for r in rules]
    assert "Conv2DBiasReluFusion" in rule_names
    assert "MultiHeadAttentionFusion" in rule_names
    assert "LayerNormBasicFusion" in rule_names

    # Test Conv2DBiasReluFusion
    g_conv = IRGraph()
    x = IRNode(id="x", op_type="Input")
    w = IRNode(id="w", op_type="Input")
    bias = IRNode(id="bias", op_type="Input")
    conv = IRNode(id="conv", op_type="Conv2D", inputs=["x", "w"])
    bias_add = IRNode(id="bias_add", op_type="BiasAdd", inputs=["conv", "bias"])
    act = IRNode(id="act", op_type="Relu", inputs=["bias_add"])

    for n in [x, w, bias, conv, bias_add, act]:
        n.shape_metadata = (1, 16, 16, 32)
        g_conv.nodes[n.id] = n
    g_conv.outputs = ["act"]

    apply_operator_fusion(g_conv)
    assert "act" in g_conv.nodes
    assert g_conv.nodes["act"].op_type == "FusedConv2DBiasRelu"
    assert g_conv.nodes["act"].inputs == ["x", "w", "bias"]


def test_layer_norm_basic_fusion():
    from ml_switcheroo_compiler.ir.core import IRGraph, IRNode
    from ml_switcheroo_compiler.transforms.passes.operator_fusion import apply_operator_fusion

    g = IRGraph()
    x = IRNode(id="x", op_type="Input")
    eps = IRNode(id="eps", op_type="Input")
    mean = IRNode(id="mean", op_type="ReduceMean", inputs=["x"])
    sub = IRNode(id="sub", op_type="Subtract", inputs=["x", "mean"])
    var = IRNode(id="var", op_type="ReduceMean", inputs=["x"])
    add_eps = IRNode(id="add_eps", op_type="Add", inputs=["var", "eps"])
    sqrt = IRNode(id="sqrt", op_type="Sqrt", inputs=["add_eps"])
    div = IRNode(id="div", op_type="Divide", inputs=["sub", "sqrt"])

    for n in [x, eps, mean, sub, var, add_eps, sqrt, div]:
        n.shape_metadata = (4, 32)
        g.nodes[n.id] = n
    g.outputs = ["div"]

    apply_operator_fusion(g)
    assert "div" in g.nodes
    assert g.nodes["div"].op_type == "FusedLayerNorm"
    assert g.nodes["div"].inputs == ["x", "eps"]
