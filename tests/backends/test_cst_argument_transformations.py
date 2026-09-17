"""Unit tests for CST transpiler argument transformations, pattern matcher, and state lifting."""

from __future__ import annotations

import libcst as cst

from ml_switcheroo_compiler.backends.cst_transpiler import (
    _ARG_RULES,
    _CST_RULES,
    ASTPatternMatcher,
    CSTTransformer,
    StateLiftingTransformer,
    transpile_source,
)
from ml_switcheroo_compiler.backends.transpiler_config_models import (
    ArgumentReorderingSpec,
    ASTPatternMatchSpec,
    ASTRewriteReplacementSpec,
    CSTRewriteRuleEntry,
    DefaultInsertionSpec,
    KeywordToPositionalSpec,
    KeywordTransformationSpec,
)


def test_ast_pattern_matcher_argument_transformations() -> None:
    """Verify ASTPatternMatcher matching, argument reordering, defaults, and conversions."""
    cst_rules = _CST_RULES.model_copy(deep=True)
    arg_rules = _ARG_RULES.model_copy(deep=True)
    matcher = ASTPatternMatcher(cst_rules=cst_rules, arg_rules=arg_rules)

    # 1. Non-Attribute function in match_call
    non_attr_call = cst.Call(func=cst.Name("plain_func"))
    assert matcher.match_call(non_attr_call, "jax") is None

    # 2. Keyword transformation
    matcher.arg_rules.keyword_transformations.append(
        KeywordTransformationSpec(
            pattern="sum_kw",
            source_kwarg="dim",
            target_kwarg="axis",
            target_framework="jax",
        )
    )
    args = [cst.Arg(keyword=cst.Name("dim"), value=cst.Integer("0"))]
    transformed_kw = matcher.transform_args("sum", args, "jax")
    assert transformed_kw[0].keyword is not None
    assert transformed_kw[0].keyword.value == "axis"

    # 3. Argument reordering: satisfied and unsatisfied permutation lengths
    matcher.arg_rules.argument_reorderings.append(ArgumentReorderingSpec(op_name="test_reorder", permutation=[1, 0], target_framework="all"))
    # Unsatisfied (1 arg < 2)
    args_short = [cst.Arg(value=cst.Integer("1"))]
    transformed_short = matcher.transform_args("test_reorder", args_short, "all")
    assert len(transformed_short) == 1

    # Satisfied (2 args >= 2)
    args_satisfy = [cst.Arg(value=cst.Name("a")), cst.Arg(value=cst.Name("b"))]
    transformed_satisfy = matcher.transform_args("test_reorder", args_satisfy, "all")
    assert isinstance(transformed_satisfy[0].value, cst.Name)
    assert transformed_satisfy[0].value.value == "b"
    assert isinstance(transformed_satisfy[1].value, cst.Name)
    assert transformed_satisfy[1].value.value == "a"

    # 4. Default insertions with bool, int, float, str, and existing kwargs
    matcher.arg_rules.default_insertions.append(
        DefaultInsertionSpec(
            op_name="def_op",
            default_kwargs={
                "bool_t": True,
                "bool_f": False,
                "int_val": 42,
                "float_val": 3.14,
                "str_val": "fast",
                "already_present": 0,
            },
            target_framework="all",
        )
    )
    args_def = [cst.Arg(keyword=cst.Name("already_present"), value=cst.Integer("99"))]
    transformed_def = matcher.transform_args("def_op", args_def, "all")
    kw_names = {a.keyword.value for a in transformed_def if a.keyword}
    assert {"bool_t", "bool_f", "int_val", "float_val", "str_val", "already_present"}.issubset(kw_names)

    # 5. Keyword to positional conversions
    matcher.arg_rules.keyword_to_positional.append(
        KeywordToPositionalSpec(
            op_name="test_kp",
            keyword_names=["axis"],
            target_positions=[0],
            target_framework="all",
        )
    )
    args_kp = [cst.Arg(value=cst.Name("x")), cst.Arg(keyword=cst.Name("axis"), value=cst.Integer("1"))]
    transformed_kp = matcher.transform_args("test_kp", args_kp, "all")
    assert transformed_kp[0].keyword is None
    assert isinstance(transformed_kp[0].value, cst.Integer)

    # 6. Attribute matching with target framework mismatch
    matcher.cst_rules.attribute_rules.append(
        CSTRewriteRuleEntry(
            match=ASTPatternMatchSpec(pattern="mismatch_attr"),
            replacement=ASTRewriteReplacementSpec(target_name="replaced_attr", target_framework="unmatched_framework"),
        )
    )
    node_attr = cst.Attribute(value=cst.Name("x"), attr=cst.Name("mismatch_attr"))
    assert matcher.match_attribute(node_attr, "jax") is None

    # 7. Import matching via import_rules (match and framework mismatch)
    matcher.cst_rules.import_rules.append(
        CSTRewriteRuleEntry(
            match=ASTPatternMatchSpec(pattern="custom_module"),
            replacement=ASTRewriteReplacementSpec(target_name="target_module", target_framework="jax"),
        )
    )
    matcher.cst_rules.import_rules.append(
        CSTRewriteRuleEntry(
            match=ASTPatternMatchSpec(pattern="mismatch_module"),
            replacement=ASTRewriteReplacementSpec(target_name="target_module", target_framework="pytorch"),
        )
    )
    assert matcher.match_import("custom_module", "jax") == "target_module"
    assert matcher.match_import("mismatch_module", "jax") is None

    # 8. Function def name matching via function_def_rules
    matcher.cst_rules.function_def_rules.append(
        CSTRewriteRuleEntry(
            match=ASTPatternMatchSpec(pattern="custom_fn"),
            replacement=ASTRewriteReplacementSpec(target_name="renamed_fn", target_framework="jax"),
        )
    )
    assert matcher.match_function_name("custom_fn", "jax") == "renamed_fn"


def test_cst_transformer_method_map_fallback() -> None:
    """Verify method_map renaming when function_def_rules are absent."""
    rules_pt = _CST_RULES.model_copy(deep=True)
    rules_pt.function_def_rules = []

    # Case 1: __call__ with self in PyTorch
    code_call = """
class Model:
    def __call__(self, x):
        return x
"""
    tree_call = cst.parse_module(code_call)
    transformer_pt = CSTTransformer(target_framework="pytorch")
    transformer_pt.matcher = ASTPatternMatcher(cst_rules=rules_pt)
    res_pt = tree_call.visit(transformer_pt)
    assert "def forward(self, x):" in res_pt.code

    # Case 2: __call__ without self in PyTorch (521->527)
    code_call_noself = """
def __call__(x):
    return x
"""
    tree_call_noself = cst.parse_module(code_call_noself)
    transformer_pt_noself = CSTTransformer(target_framework="pytorch")
    transformer_pt_noself.matcher = ASTPatternMatcher(cst_rules=rules_pt)
    res_pt_noself = tree_call_noself.visit(transformer_pt_noself)
    assert "def __call__(x):" in res_pt_noself.code

    rules_jax = _CST_RULES.model_copy(deep=True)
    rules_jax.function_def_rules = []

    code_fwd = """
class Model:
    def forward(self, x):
        return x
"""
    tree_fwd = cst.parse_module(code_fwd)
    transformer_jax = CSTTransformer(target_framework="jax")
    transformer_jax.matcher = ASTPatternMatcher(cst_rules=rules_jax)
    res_jax = tree_fwd.visit(transformer_jax)
    assert "def __call__(self, x):" in res_jax.code


def test_transpile_source_method_rewrites_and_renaming() -> None:
    """Verify method rewrites, substitutions, attribute matching, and framework naming conventions."""
    # Method-to-function rewrite via syntactic patterns fallback (when call_rules is empty, line 383)
    rules_no_call = _CST_RULES.model_copy(deep=True)
    rules_no_call.call_rules = []
    transformer_jax = CSTTransformer(target_framework="jax")
    transformer_jax.matcher = ASTPatternMatcher(cst_rules=rules_no_call)
    tree_view = cst.parse_module("x = a.view(10)")
    out_view = tree_view.visit(transformer_jax).code
    assert "reshape(10)" in out_view

    # Method substitution (dim -> ndim for JAX)
    code_dim = "x = a.dim()"
    out_dim = transpile_source(code_dim, target_framework="jax")
    assert "ndim()" in out_dim

    # Framework method map: PyTorch __call__ with self -> forward
    code_call = """
class Net:
    def __call__(self, x):
        return x
"""
    out_call = transpile_source(code_call, target_framework="pytorch")
    assert "def forward(self, x):" in out_call

    # PyTorch __call__ without self is preserved
    code_call_noself = """
def __call__(x):
    return x
"""
    out_call_noself = transpile_source(code_call_noself, target_framework="pytorch")
    assert "def __call__(x):" in out_call_noself

    # JAX forward with self -> __call__
    code_forward = """
class Net:
    def forward(self, x):
        return x
"""
    out_forward = transpile_source(code_forward, target_framework="jax")
    assert "def __call__(self, x):" in out_forward

    # Attribute rewrite in leave_Attribute
    code_attr = "y = a.T"
    out_attr = transpile_source(code_attr, target_framework="jax")
    assert "a.T" in out_attr


def test_state_lifting_transformer_class_state() -> None:
    """Verify non-self attributes, assignment target filtering, and string parameter extraction."""
    class_code = """
class TestModel:
    def __init__(self):
        local_var = 10
        external_target.attr = 20
        self.model_name = "resnet"
        self.float_val = 1.5
        self.int_val = 3
        self.unsupported_val = [1, 2]

    def forward(self, x):
        dim_info = x.shape
        return x + self.float_val
"""
    params, functional_code = StateLiftingTransformer.lift_class(class_code)
    assert params["model_name"] == "resnet"
    assert params["float_val"] == 1.5
    assert params["int_val"] == 3
    assert params["unsupported_val"] is None
    assert "local_var = 10" in functional_code
    assert "external_target.attr = 20" in functional_code
    assert 'params["float_val"]' in functional_code
    assert "x.shape" in functional_code
