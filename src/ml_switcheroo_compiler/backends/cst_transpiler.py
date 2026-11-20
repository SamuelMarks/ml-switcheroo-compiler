"""Syntactic Transpilation Engine (Whitespace/Comment Preserving)."""

from __future__ import annotations

import libcst as cst

from ml_switcheroo_compiler.backends._cst_call_transformer import CSTTransformer as CSTTransformer
from ml_switcheroo_compiler.backends._cst_pattern_matcher import (
    _ARG_RULES as _ARG_RULES,
)
from ml_switcheroo_compiler.backends._cst_pattern_matcher import (
    _ARG_RULES_PATH as _ARG_RULES_PATH,
)
from ml_switcheroo_compiler.backends._cst_pattern_matcher import (
    _CONFIG as _CONFIG,
)
from ml_switcheroo_compiler.backends._cst_pattern_matcher import (
    _CONFIG_PATH as _CONFIG_PATH,
)
from ml_switcheroo_compiler.backends._cst_pattern_matcher import (
    _CST_RULES as _CST_RULES,
)
from ml_switcheroo_compiler.backends._cst_pattern_matcher import (
    _CST_RULES_PATH as _CST_RULES_PATH,
)
from ml_switcheroo_compiler.backends._cst_pattern_matcher import (
    KNOWN_SOURCE_FRAMEWORKS as KNOWN_SOURCE_FRAMEWORKS,
)
from ml_switcheroo_compiler.backends._cst_pattern_matcher import (
    ASTPatternMatcher as ASTPatternMatcher,
)
from ml_switcheroo_compiler.backends._cst_pattern_matcher import (
    _build_attribute_chain as _build_attribute_chain,
)
from ml_switcheroo_compiler.backends._cst_pattern_matcher import (
    _build_import_name_chain as _build_import_name_chain,
)
from ml_switcheroo_compiler.backends._cst_statement_mixin import (
    CSTStatementRewriterMixin as CSTStatementRewriterMixin,
)
from ml_switcheroo_compiler.backends.eager_registry import BackendArg


class StateLiftingTransformer(cst.CSTTransformer):
    """Universal framework-blind transformer lifting stateful OOP Python classes into pure functional functions.

    with parameter PyTrees.
    """

    def __init__(self, param_var_name: str = "params") -> None:
        """Initialize StateLiftingTransformer.

        Args:
            param_var_name (str): Variable name for functional parameter dictionary / PyTree.
        """
        super().__init__()
        self.param_var_name: str = param_var_name
        self.extracted_params: dict[str, BackendArg] = {}
        self.forward_methods: list[str] = ["forward", "_" + "_call__", "compute", "call"]

    def leave_FunctionDef(
        self,
        original_node: cst.FunctionDef,
        updated_node: cst.FunctionDef,
    ) -> cst.FunctionDef:
        """Transform forward(self, ...) into forward(params, ...).

        Args:
            original_node (cst.FunctionDef): Original function node.
            updated_node (cst.FunctionDef): Updated function node.

        Returns:
            cst.FunctionDef: Pure functional function definition.
        """
        if original_node.name.value in self.forward_methods:
            new_params: list[cst.Param] = []
            for p in updated_node.params.params:
                if p.name.value == "self":
                    new_params.append(p.with_changes(name=cst.Name(self.param_var_name)))
                else:
                    new_params.append(p)
            return updated_node.with_changes(params=updated_node.params.with_changes(params=new_params))
        return updated_node

    def leave_Attribute(
        self,
        original_node: cst.Attribute,
        updated_node: cst.Attribute,
    ) -> cst.BaseExpression:
        """Rewrite self.param attribute accesses into params['param'] lookups.

        Args:
            original_node (cst.Attribute): Original attribute node.
            updated_node (cst.Attribute): Updated attribute node.

        Returns:
            cst.BaseExpression: Subscript parameter lookup or original expression.
        """
        if isinstance(updated_node.value, cst.Name) and updated_node.value.value == "self":
            attr_name: str = updated_node.attr.value
            return cst.Subscript(
                value=cst.Name(self.param_var_name),
                slice=[cst.SubscriptElement(slice=cst.Index(value=cst.SimpleString(f'"{attr_name}"')))],
            )
        return updated_node

    @classmethod
    def lift_class(
        cls,
        class_code: str,
        param_var_name: str = "params",
    ) -> tuple[dict[str, BackendArg], str]:
        """Lift an OOP Python class into an extracted parameter dictionary (PyTree) and pure functional function code.

        Args:
            class_code (str): Source code containing the class definition.
            param_var_name (str): Name of the parameters argument.

        Returns:
            tuple[dict[str, BackendArg], str]: Extracted parameter dict and functional code string.
        """
        from ml_switcheroo_compiler.tree_util import tree_flatten, tree_unflatten

        tree = cst.parse_module(class_code)
        extracted_params: dict[str, BackendArg] = {}

        class InitExtractor(cst.CSTVisitor):
            """Visitor to extract initial parameter assignments in __init__."""

            def visit_Assign(self, node: cst.Assign) -> None:
                """Inspect assignments to self.<param>.

                Args:
                    node (cst.Assign): Assignment node.
                """
                for target in node.targets:
                    if isinstance(target.target, cst.Attribute):
                        if isinstance(target.target.value, cst.Name) and target.target.value.value == "self":
                            param_name = target.target.attr.value
                            val: BackendArg = None
                            if isinstance(node.value, cst.Integer):
                                val = int(node.value.value)
                            elif isinstance(node.value, cst.Float):
                                val = float(node.value.value)
                            elif isinstance(node.value, cst.SimpleString):
                                val = str(node.value.evaluated_value)
                            extracted_params[param_name] = val

        tree.visit(InitExtractor())

        leaves, tree_def = tree_flatten(extracted_params)
        unflattened = tree_unflatten(tree_def, leaves)
        reconstructed: dict[str, BackendArg] = (
            {str(k): v for k, v in unflattened.items()} if isinstance(unflattened, dict) else {}
        )

        transformer = cls(param_var_name=param_var_name)
        transformed_tree = tree.visit(transformer)

        return (reconstructed, transformed_tree.code)


class TypeInferenceVisitor(cst.CSTVisitor):
    """Visitor for dry-run type inference."""

    def __init__(self) -> None:
        """Initialize the TypeInferenceVisitor."""
        super().__init__()
        self.inferred_types: dict[str, str] = {}

    def visit_Assign(self, node: cst.Assign) -> None:
        """Infer type of simple assignments.

        Args:
            node (cst.Assign): The assign node to analyze.
        """
        if isinstance(node.value, cst.Float):
            for t in node.targets:
                if isinstance(t.target, cst.Name):
                    self.inferred_types[t.target.value] = "float"
        elif isinstance(node.value, cst.Integer):
            for t in node.targets:
                if isinstance(t.target, cst.Name):
                    self.inferred_types[t.target.value] = "int"


def transpile_source(source_code: str, target_framework: str = "jax") -> str:
    """Transpile Python source code to the target backend while preserving formatting and comments.

    Args:
        source_code (str): Input Python source code string.
        target_framework (str): Target backend framework name.

    Returns:
        str: Transpiled Python source code string.
    """
    tree: cst.Module = cst.parse_module(source_code)
    wrapper: cst.MetadataWrapper = cst.MetadataWrapper(tree)
    transformer: CSTTransformer = CSTTransformer(target_framework=target_framework)
    modified_tree: cst.Module = wrapper.visit(transformer)
    return modified_tree.code


transpile_cst = transpile_source


def validate_diff(source_code: str, transpiled_code: str) -> bool:
    """Validate that `transpiled_code` differs from `source_code` and parses as valid Python syntax.

    Args:
        source_code (str): Original source code string.
        transpiled_code (str): Transpiled source code string.

    Returns:
        bool: True if transpiled code is modified and syntactically valid, else False.
    """
    if source_code == transpiled_code:
        return False
    try:
        cst.parse_module(transpiled_code)
        return True
    except cst.ParserSyntaxError:
        return False


def type_infer_dry_run(source_code: str) -> dict[str, str]:
    """Perform a dry-run CST parse and literal assignment type-inference pass.

    Args:
        source_code (str): Input Python source code string.

    Returns:
        dict[str, str]: Mapping of variable names to inferred type names plus `'dry_run'` status.
    """
    try:
        tree: cst.Module = cst.parse_module(source_code)
        visitor: TypeInferenceVisitor = TypeInferenceVisitor()
        tree.visit(visitor)
        res: dict[str, str] = visitor.inferred_types
        res["dry_run"] = "success"
        return res
    except cst.ParserSyntaxError:
        return {"dry_run": "failed"}
