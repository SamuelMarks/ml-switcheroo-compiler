"""Configuration loading and AST/LibCST structural pattern matching utilities."""

from __future__ import annotations

import os

import libcst as cst

from ml_switcheroo_compiler.backends.transpiler_config_models import (
    ArgumentRewritesConfig,
    CSTRewriteRulesConfig,
    TranspilerConfig,
    load_argument_rewrites,
    load_cst_rewrite_rules,
    load_transpiler_config,
)

_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "transpilation_rules.yaml")
_CONFIG = load_transpiler_config(_CONFIG_PATH)

_CST_RULES_PATH = os.path.join(os.path.dirname(__file__), "cst_rewrite_rules.yaml")
_CST_RULES = load_cst_rewrite_rules(_CST_RULES_PATH)

_ARG_RULES_PATH = os.path.join(os.path.dirname(__file__), "argument_rewrites.yaml")
_ARG_RULES = load_argument_rewrites(_ARG_RULES_PATH)

KNOWN_SOURCE_FRAMEWORKS = {
    "torch",
    "jax",
    "mlx",
    "numpy",
    "cupy",
    "dask",
    "keras",
    "tensorflow",
    "pytorch",
    "numba",
    "sparse",
}


def _build_attribute_chain(names: list[str]) -> cst.BaseExpression:
    """Build a LibCST Attribute chain expression from a sequence of identifier names.

    Args:
        names (list[str]): Ordered list of attribute path components.

    Returns:
        cst.BaseExpression: Constructed LibCST Name or nested Attribute expression.
    """
    return _build_import_name_chain(names)


def _build_import_name_chain(names: list[str]) -> cst.Attribute | cst.Name:
    """Build a LibCST Name or Attribute chain suitable for import aliases.

    Args:
        names (list[str]): Ordered list of attribute path components.

    Returns:
        cst.Attribute | cst.Name: Constructed LibCST Name or nested Attribute expression.
    """
    if not names:
        return cst.Name("empty")
    if len(names) == 1:
        return cst.Name(names[0])
    expr: cst.Attribute | cst.Name = cst.Name(names[0])
    for name in names[1:]:
        expr = cst.Attribute(value=expr, attr=cst.Name(name))
    return expr


class ASTPatternMatcher:
    """Structural pattern-matching engine driven by declarative YAML rules."""

    def __init__(
        self,
        cst_rules: CSTRewriteRulesConfig | None = None,
        arg_rules: ArgumentRewritesConfig | None = None,
        transpiler_config: TranspilerConfig | None = None,
    ) -> None:
        """Initialize the ASTPatternMatcher.

        Args:
            cst_rules (CSTRewriteRulesConfig | None): Declarative CST rewrite rules.
            arg_rules (ArgumentRewritesConfig | None): Declarative argument rewrites.
            transpiler_config (TranspilerConfig | None): Transpiler configuration.
        """
        self.cst_rules: CSTRewriteRulesConfig = cst_rules or _CST_RULES
        self.arg_rules: ArgumentRewritesConfig = arg_rules or _ARG_RULES
        self.transpiler_config: TranspilerConfig = transpiler_config or _CONFIG

    def match_call(self, node: cst.Call, target_framework: str) -> cst.BaseExpression | None:
        """Match and transform Call expressions via declarative call_rules.

        Args:
            node (cst.Call): Input Call AST node.
            target_framework (str): Target framework name.

        Returns:
            cst.BaseExpression | None: Transformed AST node or None.
        """
        if not isinstance(node.func, cst.Attribute):
            return None

        func_name = node.func.attr.value
        for rule in self.cst_rules.call_rules:
            if rule.match.pattern == func_name and rule.replacement:
                if rule.replacement.target_framework in (target_framework, "all", None):
                    target_name = rule.replacement.target_name
                    transformed_args = self.transform_args(func_name, list(node.args), target_framework)
                    new_func = node.func.with_changes(attr=cst.Name(target_name))
                    return node.with_changes(func=new_func, args=transformed_args)
        return None

    def _apply_keyword_transformations(self, args: list[cst.Arg], target_framework: str) -> list[cst.Arg]:
        """Apply declarative keyword argument renaming rules.

        Args:
            args (list[cst.Arg]): Argument list to transform.
            target_framework (str): Target framework name.

        Returns:
            list[cst.Arg]: Argument list with transformed keyword names.
        """
        new_args = list(args)
        for kw_rule in self.arg_rules.keyword_transformations:
            if kw_rule.target_framework in (target_framework, "all", None):
                for i, arg in enumerate(new_args):
                    if arg.keyword and arg.keyword.value == kw_rule.source_kwarg:
                        new_kw = arg.keyword.with_changes(value=kw_rule.target_kwarg)
                        new_args[i] = arg.with_changes(keyword=new_kw)
        return new_args

    def _apply_argument_reorderings(self, op_name: str, args: list[cst.Arg], target_framework: str) -> list[cst.Arg]:
        """Apply declarative positional argument permutation rules.

        Args:
            op_name (str): Operator or method name.
            args (list[cst.Arg]): Argument list to reorder.
            target_framework (str): Target framework name.

        Returns:
            list[cst.Arg]: Reordered argument list.
        """
        new_args = list(args)
        for reorder_rule in self.arg_rules.argument_reorderings:
            if reorder_rule.op_name == op_name and reorder_rule.target_framework in (target_framework, "all", None):
                perm = reorder_rule.permutation
                pos_args = [a for a in new_args if a.keyword is None]
                kw_args = [a for a in new_args if a.keyword is not None]
                if perm and len(pos_args) >= len(perm):
                    reordered = [pos_args[idx] for idx in perm if idx < len(pos_args)]
                    new_args = reordered + kw_args
        return new_args

    def _literal_to_cst_expr(self, val: bool | int | float | str) -> cst.BaseExpression:
        """Convert a primitive Python value into a LibCST literal expression.

        Args:
            val (bool | int | float | str): Primitive default value.

        Returns:
            cst.BaseExpression: Corresponding LibCST expression node.
        """
        if isinstance(val, bool):
            return cst.Name("True" if val else "False")
        if isinstance(val, int):
            return cst.Integer(str(val))
        if isinstance(val, float):
            return cst.Float(str(val))
        return cst.SimpleString(f'"{val}"')

    def _apply_default_insertions(self, op_name: str, args: list[cst.Arg], target_framework: str) -> list[cst.Arg]:
        """Insert missing default keyword arguments according to declarative rules.

        Args:
            op_name (str): Operator or method name.
            args (list[cst.Arg]): Argument list to augment.
            target_framework (str): Target framework name.

        Returns:
            list[cst.Arg]: Argument list with default keyword arguments appended.
        """
        new_args = list(args)
        for def_rule in self.arg_rules.default_insertions:
            if def_rule.op_name == op_name and def_rule.target_framework in (target_framework, "all", None):
                existing_kw = {a.keyword.value for a in new_args if a.keyword}
                for k, v in def_rule.default_kwargs.items():
                    if k not in existing_kw and isinstance(v, (bool, int, float, str)):
                        new_args.append(cst.Arg(keyword=cst.Name(k), value=self._literal_to_cst_expr(v)))
        return new_args

    def _apply_keyword_to_positional(self, op_name: str, args: list[cst.Arg], target_framework: str) -> list[cst.Arg]:
        """Convert specified keyword arguments into positional arguments at target indices.

        Args:
            op_name (str): Operator or method name.
            args (list[cst.Arg]): Argument list to convert.
            target_framework (str): Target framework name.

        Returns:
            list[cst.Arg]: Updated argument list.
        """
        new_args = list(args)
        for kp_rule in self.arg_rules.keyword_to_positional:
            if kp_rule.op_name == op_name and kp_rule.target_framework in (target_framework, "all", None):
                for kw_name, target_pos in zip(kp_rule.keyword_names, kp_rule.target_positions):
                    for i, arg in enumerate(list(new_args)):
                        if arg.keyword and arg.keyword.value == kw_name:
                            new_args.pop(i)
                            pos_arg = cst.Arg(value=arg.value)
                            new_args.insert(min(target_pos, len(new_args)), pos_arg)
        return new_args

    def transform_args(self, op_name: str, args: list[cst.Arg], target_framework: str) -> list[cst.Arg]:
        """Apply argument reordering, default insertion, and keyword transformations.

        Args:
            op_name (str): Operator or method name.
            args (list[cst.Arg]): Original argument list.
            target_framework (str): Target framework name.

        Returns:
            list[cst.Arg]: Transformed arguments list.
        """
        new_args = self._apply_keyword_transformations(args, target_framework)
        new_args = self._apply_argument_reorderings(op_name, new_args, target_framework)
        new_args = self._apply_default_insertions(op_name, new_args, target_framework)
        return self._apply_keyword_to_positional(op_name, new_args, target_framework)

    def match_attribute(self, node: cst.Attribute, target_framework: str) -> cst.BaseExpression | None:
        """Match and transform Attribute expressions via attribute_rules.

        Args:
            node (cst.Attribute): Attribute AST node.
            target_framework (str): Target framework name.

        Returns:
            cst.BaseExpression | None: Transformed AST node or None.
        """
        attr_name = node.attr.value
        for rule in self.cst_rules.attribute_rules:
            if rule.match.pattern == attr_name and rule.replacement:
                if rule.replacement.target_framework in (target_framework, "all", None):
                    return node.with_changes(attr=cst.Name(rule.replacement.target_name))
        return None

    def match_import(self, module_name: str, target_framework: str) -> str | None:
        """Remap module name for target framework.

        Args:
            module_name (str): Original imported module identifier.
            target_framework (str): Target framework name.

        Returns:
            str | None: Remapped module name or None if unchanged.
        """
        remappings = self.transpiler_config.import_remappings.get(target_framework, {})
        if module_name in remappings:
            return remappings[module_name]

        for rule in self.cst_rules.import_rules:
            if rule.match.pattern == module_name and rule.replacement:
                if rule.replacement.target_framework in (target_framework, "all", None):
                    return rule.replacement.target_name
        return None

    def match_function_name(self, func_name: str, target_framework: str) -> str | None:
        """Remap function definition names based on framework conventions.

        Args:
            func_name (str): Original method name.
            target_framework (str): Target framework name.

        Returns:
            str | None: Remapped method name or None.
        """
        for rule in self.cst_rules.function_def_rules:
            if rule.match.pattern == func_name and rule.replacement:
                if rule.replacement.target_framework in (target_framework, "all", None):
                    return rule.replacement.target_name
        return None
