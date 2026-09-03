from unittest.mock import MagicMock, patch


def test_autodiff_provider_parse_expression():
    from ml_switcheroo_compiler.transforms.autodiff_rules.autodiff_provider import _parse_expression

    class DummyGraph:
        def __init__(self):
            self.nodes = {}

    class DummyNode:
        def __init__(self):
            self.inputs = ["in0", "in1"]
            self.op_type = "MyOp"
            self.attributes = {"attr": 1, "float_attr": 2.5}

    graph = DummyGraph()
    node = DummyNode()

    with patch("ml_switcheroo_compiler.transforms.autodiff_rules.autodiff_provider.emit_ir_node") as mock_emit:
        mock_emit.return_value = "res_id"

        # Test cotangent
        res = _parse_expression(graph, "$cotangent", node, cotangent="C")
        assert res == "C"

        res = _parse_expression(graph, "ZerosLike($x)", node, cotangent="C")
        assert res == "res_id"
        assert mock_emit.call_args[0][1] == "ZerosLike"

        res = _parse_expression(graph, "Mul($cotangent, 2)", node, cotangent="C")
        assert res == "res_id"
        assert mock_emit.call_args[0][1] == "Mul"

        res = _parse_expression(graph, "Add(1, $cotangent)", node, cotangent="C")
        assert res == "res_id"
        assert mock_emit.call_args[0][1] == "Add"

        res = _parse_expression(graph, "Sub($cotangent, $in0)", node, cotangent="C")
        assert res == "res_id"
        assert mock_emit.call_args[0][1] == "Sub"

        res = _parse_expression(graph, "Div($in0, $cotangent)", node, cotangent="C")
        assert res == "res_id"
        assert mock_emit.call_args[0][1] == "Div"

        res = _parse_expression(graph, "Neg($cotangent)", node, cotangent="C")
        assert res == "res_id"
        assert mock_emit.call_args[0][1] == "Neg"

        res = _parse_expression(graph, "Pow($cotangent, 2)", node, cotangent="C")
        assert res == "res_id"
        assert mock_emit.call_args[0][1] == "Pow"

        # Test tangents
        res = _parse_expression(graph, "$tangent[0]", node, tangents=["T0", "T1"])
        assert res == "T0"

        # Input
        res = _parse_expression(graph, "$input[1]", node)
        assert res == "in1"

        # Nested Parens
        res = _parse_expression(graph, "Func(Arg(1))", node)
        assert mock_emit.call_args[0][1] == "Func"
        res = _parse_expression(graph, "ReduceSum($in0)", node)
        assert res == "res_id"
        assert mock_emit.call_args[0][1] == "ReduceSum"

        res = _parse_expression(graph, "Exp($in0)", node)
        assert res == "res_id"
        assert mock_emit.call_args[0][1] == "Exp"

        # Attributes
        res = _parse_expression(graph, "$attr", node)
        assert res == "$attr"

        # Constants
        res = _parse_expression(graph, "Constant(1.5)", node)
        assert res.startswith("cst_ad")


def test_get_vjp_from_data():
    from ml_switcheroo_compiler.ir.core import IRGraph, IRNode
    from ml_switcheroo_compiler.transforms.autodiff_rules.autodiff_provider import get_vjp_from_data

    with patch("ml_switcheroo_compiler.ops.generated_registry.OPS_REGISTRY", {"MyOp": {"autodiff": {"vjp": ["in0 * cotangent", "in1"]}}}):
        with patch("os.path.exists", return_value=True):
            with patch("builtins.open", MagicMock()):
                with patch("yaml.safe_load", return_value={"MyOp": {"vjp": ["mock"]}}):
                    vjp_fn = get_vjp_from_data("MyOp")
                    assert callable(vjp_fn)

        graph = IRGraph()
        node = IRNode("n1", "MyOp", ["in0", "in1"])

        with patch("ml_switcheroo_compiler.transforms.autodiff_rules.autodiff_provider._parse_expression", return_value="parsed"):
            res = vjp_fn(graph, node, "C")
            assert len(res) == 2
            assert list(res) == ["parsed", "parsed"]

    with patch("ml_switcheroo_compiler.ops.generated_registry.OPS_REGISTRY", {"MyOp": {}}):
        with patch("os.path.exists", side_effect=[True, False, False]):
            with patch("builtins.open", MagicMock()) as mock_open:
                mock_open.return_value.__enter__.return_value = "mock_file"
                with patch("yaml.safe_load", return_value={"MyOp": {"vjp": ["mock"]}}):
                    vjp_fn = get_vjp_from_data("MyOp")
                    assert callable(vjp_fn)

    with patch("ml_switcheroo_compiler.ops.generated_registry.OPS_REGISTRY", {"MyOp": {}}):
        with patch("os.path.exists", side_effect=[False, True, False]):
            with patch("builtins.open", MagicMock()) as mock_open:
                mock_open.return_value.__enter__.return_value = "mock_file"
                with patch("yaml.safe_load", return_value={"MyOp": {"vjp": ["mock"]}}):
                    vjp_fn = get_vjp_from_data("MyOp")
                    assert callable(vjp_fn)

    with patch("ml_switcheroo_compiler.ops.generated_registry.OPS_REGISTRY", {"MyOp2": {}}):
        vjp_fn = get_vjp_from_data("MyOp2")
        assert vjp_fn is None


def test_get_jvp_from_data():
    from ml_switcheroo_compiler.ir.core import IRGraph, IRNode
    from ml_switcheroo_compiler.transforms.autodiff_rules.autodiff_provider import get_jvp_from_data

    with patch("ml_switcheroo_compiler.ops.generated_registry.OPS_REGISTRY", {"MyOp": {"autodiff": {"jvp": "in0 * tangents[0]"}}}):
        jvp_fn = get_jvp_from_data("MyOp")
        assert callable(jvp_fn)

        graph = IRGraph()
        node = IRNode("n1", "MyOp", ["in0"])

        with patch("ml_switcheroo_compiler.transforms.autodiff_rules.autodiff_provider._parse_expression", return_value="parsed"):
            res = jvp_fn(graph, node, ["T0"])
            assert res == "parsed"

    with patch("ml_switcheroo_compiler.ops.generated_registry.OPS_REGISTRY", {"MyOp": {}}):
        with patch("os.path.exists", side_effect=[True, False, False]):
            with patch("builtins.open", MagicMock()) as mock_open:
                mock_open.return_value.__enter__.return_value = "mock_file"
                with patch("yaml.safe_load", return_value={"MyOp": {"jvp": ["mock"]}}):
                    jvp_fn = get_jvp_from_data("MyOp")
                    assert callable(jvp_fn)

    with patch("ml_switcheroo_compiler.ops.generated_registry.OPS_REGISTRY", {"MyOp": {}}):
        with patch("os.path.exists", side_effect=[False, True, False]):
            with patch("builtins.open", MagicMock()) as mock_open:
                mock_open.return_value.__enter__.return_value = "mock_file"
                with patch("yaml.safe_load", return_value={"MyOp": {"jvp": ["mock"]}}):
                    jvp_fn = get_jvp_from_data("MyOp")
                    assert callable(jvp_fn)

    with patch("ml_switcheroo_compiler.ops.generated_registry.OPS_REGISTRY", {"MyOp2": {}}):
        jvp_fn = get_jvp_from_data("MyOp2")
        assert getattr(jvp_fn, "__name__", "") == "_fallback_finite_difference_jvp"
