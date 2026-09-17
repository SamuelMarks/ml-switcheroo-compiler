"""Comprehensive unit and branch coverage tests for autodiff_provider."""

from unittest.mock import mock_open, patch

import yaml

from ml_switcheroo_compiler.ir.core import IRGraph, IRNode
from ml_switcheroo_compiler.transforms.autodiff_rules import autodiff_provider
from ml_switcheroo_compiler.transforms.autodiff_rules.autodiff_provider import (
    _build_parsed_call,
    _fallback_finite_difference_jvp,
    _get_autodiff_manifest,
    _get_modular_rules,
    _load_autodiff_rule,
    _load_rule_from_yaml_files,
    _parse_args_and_attrs,
    _parse_expression,
    _split_nested_args,
    get_jvp_from_data,
    get_vjp_from_data,
)


def test_split_nested_args() -> None:
    """Test _split_nested_args under various bracket configurations."""
    assert _split_nested_args("") == []
    assert _split_nested_args("a, b, c") == ["a", "b", "c"]
    assert _split_nested_args("f(x, y), [1, 2], {3: 4}") == ["f(x, y)", "[1, 2]", "{3: 4}"]


def test_parse_args_and_attrs() -> None:
    """Test _parse_args_and_attrs parsing literal and fallback keyword arguments."""
    g = IRGraph()
    node = IRNode(id="n0", op_type="MyOp", inputs=["in0"])

    args = ["in0", "dim=1", "name=unquoted_symbol", "(not_kw=2)"]
    inputs, attrs = _parse_args_and_attrs(g, args, node, cotangent="cot")
    assert inputs[0] == "in0"
    assert attrs["dim"] == 1
    assert attrs["name"] == "unquoted_symbol"


def test_build_parsed_call_variants() -> None:
    """Test _build_parsed_call for BroadcastAlign, BroadcastLike, Reshape, Squeeze, ExpandDims."""
    g = IRGraph()
    tgt = IRNode(id="tgt", op_type="Constant", inputs=[], shape_metadata=(2, 3))
    src = IRNode(id="src", op_type="Constant", inputs=[], shape_metadata=(2, 1))
    g.nodes["tgt"] = tgt
    g.nodes["src"] = src

    # Constant with empty inputs
    cst_id = _build_parsed_call(g, "Constant", [], tgt)
    assert cst_id in g.nodes

    # Constant with existing node in graph (branch 107->112)
    cst_id_dup = _build_parsed_call(g, "Constant", ["1.0"], tgt)
    assert cst_id_dup in g.nodes
    cst_id_dup2 = _build_parsed_call(g, "Constant", ["1.0"], tgt)
    assert cst_id_dup == cst_id_dup2

    # BroadcastAlign / BroadcastReduce with tgt in nodes
    br_id = _build_parsed_call(g, "BroadcastAlign", ["src", "tgt"], tgt)
    assert br_id in g.nodes

    # BroadcastAlign with tgt not in nodes
    br_missing = _build_parsed_call(g, "BroadcastAlign", ["src", "unknown_tgt"], tgt)
    assert br_missing in g.nodes

    # BroadcastLike with tgt in nodes and not in nodes
    bl_id = _build_parsed_call(g, "BroadcastLike", ["src", "tgt"], tgt)
    assert bl_id in g.nodes
    bl_missing = _build_parsed_call(g, "BroadcastLike", ["src", "unknown_tgt"], tgt)
    assert bl_missing in g.nodes

    # SetItem attrs copying when node has attributes (line 126)
    sq_node = IRNode(id="sq", op_type="Squeeze", inputs=["in0"], attributes={"dim": 1})
    setitem_call = _build_parsed_call(g, "SetItem", ["in0"], sq_node)
    assert setitem_call in g.nodes

    # Squeeze -> ExpandDims and ExpandDims -> Squeeze attribute forwarding
    exp_call = _build_parsed_call(g, "ExpandDims", ["in0"], sq_node)
    assert exp_call in g.nodes

    exp_node = IRNode(id="exp_node", op_type="ExpandDims", inputs=["in0"], attributes={"dim": 1})
    sq_call = _build_parsed_call(g, "Squeeze", ["in0"], exp_node)
    assert sq_call in g.nodes

    # Reshape with node.inputs[0] in graph.nodes (line 134)
    r_node = IRNode(id="r", op_type="Reshape", inputs=["tgt"], attributes={"shape": (6,)})
    res_call = _build_parsed_call(g, "Reshape", ["tgt"], r_node)
    assert res_call in g.nodes

    r_node_empty = IRNode(id="r_empty", op_type="Reshape", inputs=[], attributes={"shape": (6,)})
    res_call2 = _build_parsed_call(g, "Reshape", ["unknown"], r_node_empty)
    assert res_call2 in g.nodes

    r_node_unknown = IRNode(id="r_unk", op_type="Reshape", inputs=["not_in_graph"], attributes={"shape": (6,)})
    res_call3 = _build_parsed_call(g, "Reshape", ["not_in_graph"], r_node_unknown)
    assert res_call3 in g.nodes


def test_parse_expression_edge_cases() -> None:
    """Test _parse_expression negative prefixes, $output, and float constants."""
    g = IRGraph()
    node = IRNode(id="n_test", op_type="MyOp", inputs=["in0"])
    g.nodes["in_neg"] = IRNode(id="in_neg", op_type="Constant", inputs=[])

    # Float string returns as-is
    assert _parse_expression(g, "42.5", node) == "42.5"

    # Negative prefixes "- " and "-"
    neg1 = _parse_expression(g, "- in_neg", node)
    assert neg1 in g.nodes
    neg2 = _parse_expression(g, "-in_neg", node)
    assert neg2 in g.nodes

    # Output identifier
    assert _parse_expression(g, "$output", node) == "n_test"


def test_fallback_finite_difference_jvp() -> None:
    """Test _fallback_finite_difference_jvp with string and tuple/list tangents."""
    g = IRGraph()
    g.nodes["in0"] = IRNode(id="in0", op_type="Constant", inputs=[], shape_metadata=(2,))
    g.nodes["in1"] = IRNode(id="in1", op_type="Constant", inputs=[], shape_metadata=(2,))
    node = IRNode(id="node_op", op_type="Add", inputs=["in0", "in1"], attributes={"custom": True})

    # tangents as single string (tests else [tangents] branch and missing tangents loop)
    res1 = _fallback_finite_difference_jvp(g, node, "tangent0")
    assert res1 in g.nodes

    # tangents as tuple with matching count
    res2 = _fallback_finite_difference_jvp(g, node, ("tangent0", "tangent1"))
    assert res2 in g.nodes


def test_get_autodiff_manifest_branches() -> None:
    """Test _get_autodiff_manifest caching, missing file, and exception handling."""
    # 1. Cache hit
    autodiff_provider._MANIFEST_MODEL_CACHE = "cached_manifest"
    assert _get_autodiff_manifest() == "cached_manifest"
    autodiff_provider._MANIFEST_MODEL_CACHE = None

    # 2. File not found
    with patch("os.path.exists", return_value=False):
        assert _get_autodiff_manifest() is None

    # 3. Exception during loading
    with patch("os.path.exists", return_value=True):
        with patch("builtins.open", side_effect=OSError("Read error")):
            assert _get_autodiff_manifest() is None


def test_get_modular_rules_branches() -> None:
    """Test _get_modular_rules caching, missing files, non-dict content, exception handling, and dict loading."""
    # 1. Cache hit
    autodiff_provider._MODULAR_RULES_CACHE = {"jvp_rules": {}, "vjp_rules": {}}
    assert _get_modular_rules() == {"jvp_rules": {}, "vjp_rules": {}}
    autodiff_provider._MODULAR_RULES_CACHE = None

    # 2. File exists and raises exception
    with patch("os.path.exists", return_value=True):
        with patch("builtins.open", side_effect=OSError("Read error")):
            res = _get_modular_rules()
            assert res == {"jvp_rules": {}, "vjp_rules": {}}

    autodiff_provider._MODULAR_RULES_CACHE = None

    # 3. File exists for some and not for others (branch 276->269)
    with patch("os.path.exists", side_effect=lambda p: "linalg_" in p):
        with patch("builtins.open", mock_open(read_data=yaml.dump({"jvp_rules": {}}))):
            res_str = _get_modular_rules()
            assert res_str == {"jvp_rules": {}, "vjp_rules": {}}

    autodiff_provider._MODULAR_RULES_CACHE = None

    # 3b. File exists with non-dict content (branch 280->269)
    with patch("os.path.exists", return_value=True):
        with patch("builtins.open", mock_open(read_data=yaml.dump(["not", "a", "dict"]))):
            res_nondict = _get_modular_rules()
            assert res_nondict == {"jvp_rules": {}, "vjp_rules": {}}

    autodiff_provider._MODULAR_RULES_CACHE = None

    # 4. File exists with dict but without vjp_rules (branch 280->269)
    with patch("os.path.exists", return_value=True):
        with patch("builtins.open", mock_open(read_data=yaml.dump(["not", "a", "dict"]))):
            res_nondict = _get_modular_rules()
            assert res_nondict == {"jvp_rules": {}, "vjp_rules": {}}

    autodiff_provider._MODULAR_RULES_CACHE = None

    # 4. File exists with dict but without vjp_rules (branch 280->269)
    with patch("os.path.exists", return_value=True):
        with patch("builtins.open", mock_open(read_data=yaml.dump({"jvp_rules": {"SampleOp": {"jvp": "$tangent[0]"}}}))):
            res_only_jvp = _get_modular_rules()
            assert "SampleOp" in res_only_jvp["jvp_rules"]

    autodiff_provider._MODULAR_RULES_CACHE = None

    # 5. File exists and returns valid jvp and vjp content
    fake_yaml = yaml.dump(
        {
            "jvp_rules": {"SampleOp": {"jvp": "$tangent[0]"}},
            "vjp_rules": {"SampleOp": {"vjp": ["$cotangent"]}},
        }
    )
    with patch("os.path.exists", return_value=True):
        with patch("builtins.open", mock_open(read_data=fake_yaml)):
            res2 = _get_modular_rules()
            assert "SampleOp" in res2["jvp_rules"]
            assert "SampleOp" in res2["vjp_rules"]


def test_load_autodiff_rule_hierarchy() -> None:
    """Test _load_autodiff_rule across all 5 fallback stages."""
    autodiff_provider._MANIFEST_MODEL_CACHE = None
    autodiff_provider._MODULAR_RULES_CACHE = None

    # 1. Stage 1: AutodiffRulesManifestModel with single string VJP and JVP
    class DummyRule:
        """Dummy rule model with VJP and JVP attributes."""

        def __init__(self, vjp=None, jvp=None):
            """Initialize dummy rule with optional vjp and jvp strings."""
            self.vjp = vjp
            self.jvp = jvp

    class DummyManifest:
        """Dummy manifest model with vjp_rules and jvp_rules dictionaries."""

        def __init__(self):
            """Initialize dummy manifest with test rule mappings."""
            self.vjp_rules = {"str_vjp_op": DummyRule(vjp="x_cot"), "empty_vjp": DummyRule(vjp=None)}
            self.jvp_rules = {"sample_jvp_op": DummyRule(jvp="x_tan"), "empty_jvp": DummyRule(vjp=None)}

    with patch("ml_switcheroo_compiler.transforms.autodiff_rules.autodiff_provider._get_autodiff_manifest", return_value=DummyManifest()):
        # VJP string wraps to list
        res_vjp = _load_autodiff_rule("str_vjp_op", "vjp")
        assert res_vjp == {"vjp": ["x_cot"]}

        # JVP string returns dict
        res_jvp = _load_autodiff_rule("sample_jvp_op", "jvp")
        assert res_jvp == {"jvp": "x_tan"}

        # Empty VJP and JVP fall through
        with patch("ml_switcheroo_compiler.transforms.autodiff_rules.autodiff_provider._get_modular_rules", return_value={"jvp_rules": {}, "vjp_rules": {}}):
            assert _load_autodiff_rule("empty_vjp", "vjp") is None
            assert _load_autodiff_rule("empty_jvp", "jvp") is None

    # 2. Stage 2: Modular derivative files fallback (including non-dict entry branch 319->316)
    with patch("ml_switcheroo_compiler.transforms.autodiff_rules.autodiff_provider._get_autodiff_manifest", return_value=None):
        mod_rules = {
            "vjp_rules": {"modular_op": {"vjp": ["cot_mod"]}, "nondict_op": "not_a_dict"},
            "jvp_rules": {"modular_op": {"jvp": "tan_mod"}},
        }
        with patch("ml_switcheroo_compiler.transforms.autodiff_rules.autodiff_provider._get_modular_rules", return_value=mod_rules):
            assert _load_autodiff_rule("modular_op", "vjp") == {"vjp": ["cot_mod"]}
            assert _load_autodiff_rule("modular_op", "jvp") == {"jvp": "tan_mod"}
            # nondict_op in modular falls through loop
            assert _load_autodiff_rule("nondict_op", "vjp") is None

    # 3. Stage 3: Primitive registries fallback (VJP and JVP, and JVP miss branch 329->337)
    with patch("ml_switcheroo_compiler.transforms.autodiff_rules.autodiff_provider._get_autodiff_manifest", return_value=None):
        with patch("ml_switcheroo_compiler.transforms.autodiff_rules.autodiff_provider._get_modular_rules", return_value={"jvp_rules": {}, "vjp_rules": {}}):
            with patch("ml_switcheroo_compiler.transforms.autodiff_rules.vjp_registry.load_primitive_vjp_rules", return_value={"prim_op": DummyRule(vjp=["cot_prim"])}):
                assert _load_autodiff_rule("prim_op", "vjp") == {"vjp": ["cot_prim"]}

            with patch("ml_switcheroo_compiler.transforms.autodiff_rules.jvp_registry.load_primitive_jvp_rules", return_value={"prim_op": DummyRule(jvp="tan_prim")}):
                assert _load_autodiff_rule("prim_op", "jvp") == {"jvp": "tan_prim"}
                # Missing JVP primitive falls through (branch 329->337)
                assert _load_autodiff_rule("non_prim_op", "jvp") == _fallback_finite_difference_jvp or _load_autodiff_rule("non_prim_op", "jvp") is None

    # 4. Stage 4: OPS_REGISTRY fallback (including non-dict and missing autodiff branch 340->345)
    with patch("ml_switcheroo_compiler.transforms.autodiff_rules.autodiff_provider._get_autodiff_manifest", return_value=None):
        with patch("ml_switcheroo_compiler.transforms.autodiff_rules.autodiff_provider._get_modular_rules", return_value={"jvp_rules": {}, "vjp_rules": {}}):
            registry_data = {
                "registry_op": {"autodiff": {"vjp": ["cot_reg"]}},
                "nondict_reg_op": "not_a_dict",
                "no_ad_op": {"autodiff": "not_a_dict"},
                "diff_rule_op": {"autodiff": {"other_rule": 1}},
            }
            with patch("ml_switcheroo_compiler.ops.generated_registry.OPS_REGISTRY", registry_data):
                assert _load_autodiff_rule("registry_op", "vjp") == {"vjp": ["cot_reg"]}
                assert _load_autodiff_rule("nondict_reg_op", "vjp") is None
                assert _load_autodiff_rule("no_ad_op", "vjp") is None
                assert _load_autodiff_rule("diff_rule_op", "vjp") is None

    # 5. Non-vjp and non-jvp rule_type (branch 329->337)
    with patch("ml_switcheroo_compiler.transforms.autodiff_rules.autodiff_provider._get_autodiff_manifest", return_value=None):
        with patch("ml_switcheroo_compiler.transforms.autodiff_rules.autodiff_provider._get_modular_rules", return_value={"jvp_rules": {}, "vjp_rules": {}}):
            assert _load_autodiff_rule("any_op", "other_type") is None


def test_load_rule_from_yaml_files_all_branches() -> None:
    """Test _load_rule_from_yaml_files all filesystem locations and partial branches."""

    # 1. rule_file: {rule_type}_rules.yaml exists, but op not in data (branch 373->376)
    def exists_rule_file(path: str) -> bool:
        """Check if path matches rule file."""
        return path.endswith("vjp_rules.yaml")

    with patch("os.path.exists", side_effect=exists_rule_file):
        with patch("builtins.open", mock_open(read_data=yaml.dump({"OpRuleFile": {"vjp": ["cot_rf"]}}))):
            assert _load_rule_from_yaml_files("OpRuleFile", "vjp") == {"vjp": ["cot_rf"]}
            assert _load_rule_from_yaml_files("MissingInRuleFile", "vjp") is None

    # 2. yaml_path: rules/{op_type}.yaml exists, but op not in data (branch 381->388)
    def exists_rules_op(path: str) -> bool:
        """Check if path matches rules directory file."""
        return path.endswith("rules/OpRulesDir.yaml") or path.endswith("rules/MissingInRulesDir.yaml")

    with patch("os.path.exists", side_effect=exists_rules_op):
        with patch("builtins.open", mock_open(read_data=yaml.dump({"OpRulesDir": {"vjp": ["cot_rd"]}}))):
            assert _load_rule_from_yaml_files("OpRulesDir", "vjp") == {"vjp": ["cot_rd"]}
            assert _load_rule_from_yaml_files("MissingInRulesDir", "vjp") is None

    # 3. manifest_path: ../autodiff_rules.yaml (including non-dict rule_entry branch 385->382)
    def exists_manifest(path: str) -> bool:
        """Check if path matches autodiff manifest."""
        return path.endswith("autodiff_rules.yaml") and "transforms/autodiff_rules.yaml" in path

    manifest_data = {
        "vjp_rules": {
            "OpManifest": {"vjp": ["cot_m"]},
            "non_dict_manifest": "not_a_dict",
        }
    }
    with patch("os.path.exists", side_effect=exists_manifest):
        with patch("builtins.open", mock_open(read_data=yaml.dump(manifest_data))):
            assert _load_rule_from_yaml_files("OpManifest", "vjp") == {"vjp": ["cot_m"]}
            assert _load_rule_from_yaml_files("non_dict_manifest", "vjp") is None
            assert _load_rule_from_yaml_files("OpManifest", "missing_section") is None

    # 4. legacy_path: autodiff_rules.yaml in same dir (including op not in data branch 392->395)
    def exists_legacy(path: str) -> bool:
        """Check if path matches legacy autodiff rules."""
        return path.endswith("autodiff_rules/autodiff_rules.yaml")

    with patch("os.path.exists", side_effect=exists_legacy):
        with patch("builtins.open", mock_open(read_data=yaml.dump({"OpLegacy": {"vjp": ["cot_leg"]}}))):
            assert _load_rule_from_yaml_files("OpLegacy", "vjp") == {"vjp": ["cot_leg"]}
            assert _load_rule_from_yaml_files("MissingInLegacy", "vjp") is None

    # 5. Non-existent anywhere
    with patch("os.path.exists", return_value=False):
        assert _load_rule_from_yaml_files("UnknownOp", "vjp") is None


def test_get_vjp_and_jvp_data_callables() -> None:
    """Test get_vjp_from_data and get_jvp_from_data and their returned callables."""
    g = IRGraph()
    node = IRNode(id="n0", op_type="MyOp", inputs=["in0"])

    # VJP callable
    with patch("ml_switcheroo_compiler.transforms.autodiff_rules.autodiff_provider._load_autodiff_rule") as mock_load:
        mock_load.return_value = {"vjp": ["$cotangent"]}
        vjp_fn = get_vjp_from_data("MyOp")
        assert callable(vjp_fn)
        res = vjp_fn(g, node, "C")
        assert res == ("C",)

        # Non-existent VJP rule
        mock_load.return_value = None
        assert get_vjp_from_data("NoOp") is None

    # JVP callable with string tangent and tuple tangent
    with patch("ml_switcheroo_compiler.transforms.autodiff_rules.autodiff_provider._load_autodiff_rule") as mock_load:
        mock_load.return_value = {"jvp": "$tangent[0]"}
        jvp_fn = get_jvp_from_data("MyOp")
        assert callable(jvp_fn)
        # string tangent (tests else [tangents] branch)
        res_str = jvp_fn(g, node, "T0")
        assert res_str == "T0"
        # list tangent
        res_list = jvp_fn(g, node, ["T0"])
        assert res_list == "T0"

        # Non-existent JVP rule returns fallback
        mock_load.return_value = None
        assert get_jvp_from_data("NoOp") == _fallback_finite_difference_jvp
