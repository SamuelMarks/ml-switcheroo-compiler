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

    # Test Constant caching (branch 47->52)
    cst_id1 = _parse_expression(graph, "Constant(1.0)", node, "t0")
    cst_id2 = _parse_expression(graph, "Constant(1.0)", node, "t0")
    assert cst_id1 == cst_id2


def test_autodiff_provider_rule_loading(mocker):
    """Test _load_autodiff_rule fallback branches."""
    from unittest.mock import mock_open

    import yaml

    from ml_switcheroo_compiler.transforms.autodiff_rules.autodiff_provider import _load_autodiff_rule

    # 1. yaml_path exists but op_type not in data (branch 130->133)
    def fake_exists_no_op(path):
        return path.endswith("FakeOp.yaml")

    mocker.patch("os.path.exists", side_effect=fake_exists_no_op)
    mocker.patch("builtins.open", mock_open(read_data=yaml.dump({"OtherOp": {"jvp": "x"}})))
    assert _load_autodiff_rule("FakeOp", "nonexistent.yaml") is None

    # 2. legacy_path exists and has op_type (lines 135-138)
    def fake_exists_legacy(path):
        return path.endswith("autodiff_rules.yaml")

    mocker.patch("os.path.exists", side_effect=fake_exists_legacy)
    mocker.patch("builtins.open", mock_open(read_data=yaml.dump({"LegacyOp": {"jvp": "leg_jvp"}})))
    res = _load_autodiff_rule("LegacyOp", "nonexistent.yaml")
    assert res == {"jvp": "leg_jvp"}

    # 3. legacy_path exists but op_type not in data (branch 137->140)
    mocker.patch("builtins.open", mock_open(read_data=yaml.dump({"OtherOp": {}})))
    assert _load_autodiff_rule("MissingLegacyOp", "nonexistent.yaml") is None
