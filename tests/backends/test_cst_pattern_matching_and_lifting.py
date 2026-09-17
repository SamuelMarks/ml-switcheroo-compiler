"""Tests for structural CST pattern matching, state lifting, and AST node coverage."""

import libcst as cst

from ml_switcheroo_compiler.backends.ast_to_ir import parse_ast_to_ir
from ml_switcheroo_compiler.backends.cst_transpiler import (
    ASTPatternMatcher,
    StateLiftingTransformer,
)
from ml_switcheroo_compiler.backends.ir_to_ast import emit_ir_to_ast
from ml_switcheroo_compiler.backends.transpiler_config_models import (
    load_argument_rewrites,
    load_cst_rewrite_rules,
)
from ml_switcheroo_compiler.ir.core import IRGraph, IRNode
from ml_switcheroo_compiler.tree_util import tree_flatten, tree_unflatten


def test_argument_rewrites_yaml_loading() -> None:
    """Verify that argument_rewrites.yaml loads and validates against Pydantic schema."""
    config = load_argument_rewrites()
    assert len(config.keyword_transformations) > 0
    assert len(config.argument_reorderings) > 0
    assert len(config.default_insertions) > 0
    assert len(config.keyword_to_positional) > 0

    dim_rule = next(r for r in config.keyword_transformations if r.source_kwarg == "dim")
    assert dim_rule.target_kwarg == "axis"


def test_cst_rewrite_rules_expanded_sections() -> None:
    """Verify expanded sections in cst_rewrite_rules.yaml."""
    rules = load_cst_rewrite_rules()
    assert len(rules.call_rules) > 0
    assert len(rules.attribute_rules) > 0
    assert len(rules.import_rules) > 0
    assert len(rules.function_def_rules) > 0


def test_ast_pattern_matcher_call_and_args() -> None:
    """Verify ASTPatternMatcher matching and argument transformations."""
    matcher = ASTPatternMatcher()

    # Match call: x.view(2, 3) -> x.reshape(2, 3)
    call_view = cst.Call(
        func=cst.Attribute(value=cst.Name("x"), attr=cst.Name("view")),
        args=[cst.Arg(value=cst.Integer("2")), cst.Arg(value=cst.Integer("3"))],
    )
    rewritten_view = matcher.match_call(call_view, "jax")
    assert rewritten_view is not None
    assert isinstance(rewritten_view, cst.Call)
    assert isinstance(rewritten_view.func, cst.Attribute)
    assert rewritten_view.func.attr.value == "reshape"

    # Match attribute: x.T
    attr_t = cst.Attribute(value=cst.Name("x"), attr=cst.Name("T"))
    rewritten_attr = matcher.match_attribute(attr_t, "jax")
    assert rewritten_attr is not None
    assert isinstance(rewritten_attr, cst.Attribute)
    assert rewritten_attr.attr.value == "T"

    # Match import: torch.nn -> flax.linen
    assert matcher.match_import("torch.nn", "jax") == "flax.linen"
    assert matcher.match_import("unknown.module", "jax") is None

    # Match function def name
    assert matcher.match_function_name("forward", "jax") == "__call__"
    assert matcher.match_function_name("forward", "pytorch") is None

    # Test argument transformations: reorder, default insertion, kw to pos
    # tensordot(a, b) -> tensordot(a, b, axes=2)
    args_td = [cst.Arg(value=cst.Name("a")), cst.Arg(value=cst.Name("b"))]
    transformed_td = matcher.transform_args("tensordot", args_td, "jax")
    assert any(a.keyword and a.keyword.value == "axes" for a in transformed_td)

    # clip(x, a_min=0.0, a_max=1.0) -> clip(x, 0.0, 1.0)
    args_clip = [
        cst.Arg(value=cst.Name("x")),
        cst.Arg(keyword=cst.Name("a_min"), value=cst.Float("0.0")),
        cst.Arg(keyword=cst.Name("a_max"), value=cst.Float("1.0")),
    ]
    transformed_clip = matcher.transform_args("clip", args_clip, "jax")
    assert not any(a.keyword for a in transformed_clip)
    assert len(transformed_clip) == 3


def test_state_lifting_transformer_basic() -> None:
    """Verify stateful class lifting to pure functional function and PyTree parameters."""
    oop_code = """class LinearModel:
    def __init__(self):
        self.weight = 2.5
        self.bias = 1.0

    def forward(self, x):
        return x * self.weight + self.bias
"""
    params, functional_code = StateLiftingTransformer.lift_class(oop_code, param_var_name="params")

    # Verify extracted parameters PyTree
    assert params == {"weight": 2.5, "bias": 1.0}

    # Verify PyTree roundtrip serialization
    leaves, tree_def = tree_flatten(params)
    assert leaves == [1.0, 2.5]
    restored = tree_unflatten(tree_def, leaves)
    assert restored == params

    # Verify functionalized code
    assert "def forward(params, x):" in functional_code
    assert 'params["weight"]' in functional_code
    assert 'params["bias"]' in functional_code
    assert "self." not in functional_code


def test_state_lifting_transformer_methods() -> None:
    """Verify state lifting across different compute method names."""
    code = """class ComputeModule:
    def __init__(self):
        self.scale = 10

    def compute(self, x):
        return x * self.scale
"""
    params, functional_code = StateLiftingTransformer.lift_class(code, param_var_name="p")
    assert params == {"scale": 10}
    assert "def compute(p, x):" in functional_code
    assert 'p["scale"]' in functional_code


def test_ast_to_ir_ifexp_ternary() -> None:
    """Verify parsing ternary IfExp (a if cond else b) into Cond IR node."""
    code = "def check(x, cond):\n    return 1.0 if cond else 0.0\n"
    graph = parse_ast_to_ir(code)
    cond_nodes = [n for n in graph.nodes.values() if n.op_type == "Cond"]
    assert len(cond_nodes) >= 1
    assert cond_nodes[0].attributes.get("is_ternary") is True


def test_ast_to_ir_list_comp() -> None:
    """Verify parsing list comprehension into Map IR node."""
    code = "def process(items):\n    return [x for x in items]\n"
    graph = parse_ast_to_ir(code)
    map_nodes = [n for n in graph.nodes.values() if n.op_type == "Map"]
    assert len(map_nodes) >= 1
    assert map_nodes[0].attributes.get("target_var") == "x"


def test_ast_to_ir_subscript_dict_and_getitem() -> None:
    """Verify parsing dictionary indexing and getitem into IR nodes."""
    code = """def lookup(p, arr, i):
    w = p["weights"]
    elem = arr[i]
    first = arr[0]
    return w
"""
    graph = parse_ast_to_ir(code)
    dict_nodes = [n for n in graph.nodes.values() if n.op_type == "DictGet"]
    assert len(dict_nodes) >= 1
    assert dict_nodes[0].attributes.get("key") == "weights"

    item_nodes = [n for n in graph.nodes.values() if n.op_type == "GetItem"]
    assert len(item_nodes) >= 2


def test_ast_to_ir_dynamic_call_signatures() -> None:
    """Verify forwarding *args and **kwargs in call signatures."""
    code = """def dynamic_call(fn, *args, **kwargs):
    return fn(1.0, *args, **kwargs)
"""
    graph = parse_ast_to_ir(code)
    call_nodes = [n for n in graph.nodes.values() if n.op_type == "ForeignCall"]
    assert len(call_nodes) >= 1
    node = call_nodes[0]
    assert node.attributes.get("has_varargs") is True
    assert node.attributes.get("has_varkwargs") is True


def test_ir_to_ast_control_flow_and_data_structures() -> None:
    """Verify emission of Cond, Scan, WhileLoop, DictGet, and GetItem to AST."""
    graph = IRGraph()
    graph.nodes = {
        "x": IRNode(id="x", op_type="Input"),
        "cond": IRNode(id="cond", op_type="Input"),
        "c_node": IRNode(id="c_node", op_type="Cond", inputs=["cond", "x", "x"], attributes={"var_name": "res"}),
        "scan_node": IRNode(id="scan_node", op_type="Scan", inputs=["x"], attributes={"var_name": "scanned"}),
        "dict_get": IRNode(id="dict_get", op_type="DictGet", inputs=["x"], attributes={"key": "weight", "var_name": "w"}),
        "get_item": IRNode(id="get_item", op_type="GetItem", inputs=["x"], attributes={"index": 0, "var_name": "first"}),
        "foreign": IRNode(
            id="foreign",
            op_type="ForeignCall",
            inputs=["x"],
            attributes={"callee": "custom_fn", "has_varargs": True, "has_varkwargs": True, "var_name": "f_out"},
        ),
    }

    ast_mod = emit_ir_to_ast(graph, "jax")
    code = ast_mod.code

    assert "res = x if cond else x" in code
    assert "scanned = scan(x)" in code
    assert 'w = x["weight"]' in code
    assert "first = x[0]" in code
    assert "f_out = custom_fn(x, *args, **kwargs)" in code
