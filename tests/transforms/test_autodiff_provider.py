def test_autodiff_provider_coverage(mocker):
    """Test function."""
    from ml_switcheroo_compiler.transforms.autodiff_rules.autodiff_provider import _fallback_finite_difference_jvp, _parse_expression

    class DummyGraph:
        def __init__(self):
            self.nodes = {}

    class DummyNode:
        def __init__(self):
            self.inputs = ["in0", "in1"]
            self.op_type = "MyOp"
            self.attributes = {"attr": 1}

    graph = DummyGraph()
    node = DummyNode()

    mocker.patch("ml_switcheroo_compiler.transforms.autodiff_rules.autodiff_provider.emit_ir_node", return_value="dummy")

    _fallback_finite_difference_jvp(graph, node, ["tangent0"])  # 1 tangent, 2 inputs. Covers line 88-89

    node.op_type = "NotSetItem"
    _parse_expression(graph, "SetItem()", node, "t0")  # Covers line 56
