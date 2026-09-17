"""Module cst_transpiler.py."""

from __future__ import annotations

# ruff: noqa: E402, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, E701, E722, F403, E711, E712, PLR0913, PLR0915
import os
from typing import cast

"""Syntactic Transpilation Engine (Whitespace/Comment Preserving)."""

import libcst as cst
import libcst.matchers as m

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

KNOWN_SOURCE_FRAMEWORKS = {"torch", "jax", "mlx", "numpy", "cupy", "dask", "keras", "tensorflow", "pytorch", "numba", "sparse"}


def _build_attribute_chain(names: list[str]) -> cst.BaseExpression:
    """Build a libcst Attribute chain from a list of strings.

    Args:
        names (list[str]): The names parameter.

    Returns:
        cst.BaseExpression: Result.
    """
    if not names:
        return cst.Name("empty")
    if len(names) == 1:
        return cst.Name(names[0])
    expr: cst.BaseExpression = cst.Name(names[0])
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

    def transform_args(self, op_name: str, args: list[cst.Arg], target_framework: str) -> list[cst.Arg]:
        """Apply argument reordering, default insertion, and keyword transformations.

        Args:
            op_name (str): Operator or method name.
            args (list[cst.Arg]): Original argument list.
            target_framework (str): Target framework name.

        Returns:
            list[cst.Arg]: Transformed arguments list.
        """
        new_args = list(args)

        # Keyword transformations
        for kw_rule in self.arg_rules.keyword_transformations:
            if kw_rule.target_framework in (target_framework, "all", None):
                for i, arg in enumerate(new_args):
                    if arg.keyword and arg.keyword.value == kw_rule.source_kwarg:
                        new_kw = arg.keyword.with_changes(value=kw_rule.target_kwarg)
                        new_args[i] = arg.with_changes(keyword=new_kw)

        # Argument reorderings
        for reorder_rule in self.arg_rules.argument_reorderings:
            if reorder_rule.op_name == op_name and reorder_rule.target_framework in (target_framework, "all", None):
                perm = reorder_rule.permutation
                pos_args = [a for a in new_args if a.keyword is None]
                kw_args = [a for a in new_args if a.keyword is not None]
                if perm and len(pos_args) >= len(perm):
                    reordered = [pos_args[idx] for idx in perm if idx < len(pos_args)]
                    new_args = reordered + kw_args

        # Default insertions
        for def_rule in self.arg_rules.default_insertions:
            if def_rule.op_name == op_name and def_rule.target_framework in (target_framework, "all", None):
                existing_kw = {a.keyword.value for a in new_args if a.keyword}
                for k, v in def_rule.default_kwargs.items():
                    if k not in existing_kw:
                        val_expr: cst.BaseExpression
                        if isinstance(v, bool):
                            val_expr = cst.Name("True" if v else "False")
                        elif isinstance(v, int):
                            val_expr = cst.Integer(str(v))
                        elif isinstance(v, float):
                            val_expr = cst.Float(str(v))
                        else:
                            val_expr = cst.SimpleString(f'"{v}"')
                        new_args.append(cst.Arg(keyword=cst.Name(k), value=val_expr))

        # Keyword to positional conversions
        for kp_rule in self.arg_rules.keyword_to_positional:
            if kp_rule.op_name == op_name and kp_rule.target_framework in (target_framework, "all", None):
                for kw_name, target_pos in zip(kp_rule.keyword_names, kp_rule.target_positions):
                    for i, arg in enumerate(list(new_args)):
                        if arg.keyword and arg.keyword.value == kw_name:
                            new_args.pop(i)
                            pos_arg = cst.Arg(value=arg.value)
                            new_args.insert(min(target_pos, len(new_args)), pos_arg)

        return new_args

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


class CSTTransformer(cst.CSTTransformer):
    """Pass registry for targeted pattern matching of native APIs."""

    def __init__(self, target_framework: str = "jax") -> None:
        """Initialize the CSTTransformer.

        Args:
            target_framework (str): The target_framework parameter.
        """
        super().__init__()
        self.target_framework = target_framework
        self.target_config = _CONFIG.frameworks.get(target_framework)
        self.matcher = ASTPatternMatcher()

    def leave_ImportFrom(
        self,
        original_node: cst.ImportFrom,
        updated_node: cst.ImportFrom,
    ) -> cst.ImportFrom:
        """Robust Import Resolution Pass.

        Returns:
            cst.ImportFrom: The inferred shape or computed result

        Args:
            original_node (cst.ImportFrom): The original_node parameter.
            updated_node (cst.ImportFrom): The updated_node parameter.
        """
        if not updated_node.module or not self.target_config:
            return updated_node

        src_module: str = ""
        if isinstance(updated_node.module, cst.Name):
            src_module = updated_node.module.value
        elif isinstance(updated_node.module, cst.Attribute):

            def _get_base_name(node: cst.BaseExpression) -> str:
                """Get the base name of a node."""
                if isinstance(node, cst.Name):
                    return node.value
                elif isinstance(node, cst.Attribute):
                    return _get_base_name(node.value)
                return ""

            src_module = _get_base_name(updated_node.module)
        else:
            return updated_node

        remapped = self.matcher.match_import(src_module, self.target_framework)
        if remapped:
            target_parts = remapped.split(".")
            return updated_node.with_changes(module=_build_attribute_chain(target_parts))

        if src_module in KNOWN_SOURCE_FRAMEWORKS:
            target_module: str = self.target_config.target_module
            if target_module != src_module:
                target_parts = target_module.split(".")
                return updated_node.with_changes(module=_build_attribute_chain(target_parts))

        return updated_node

    def leave_Import(
        self,
        original_node: cst.Import,
        updated_node: cst.Import,
    ) -> cst.Import:
        """Handle standard imports.

        Args:
            original_node (cst.Import): The original import node.
            updated_node (cst.Import): The updated import node.

        Returns:
            cst.Import: The modified import node.
        """
        if not self.target_config:
            return updated_node

        new_names: list[cst.ImportAlias] = []
        mutated: bool = False
        target_module: str = self.target_config.target_module

        def _get_import_name(node: cst.BaseExpression) -> str:
            """Recursively extract dotted import name string.

            Args:
                node (cst.BaseExpression): AST expression node.

            Returns:
                str: Dot-delimited import name string.
            """
            if isinstance(node, cst.Attribute):
                base = _get_import_name(node.value)
                return f"{base}.{node.attr.value}" if base else node.attr.value
            return getattr(node, "value", "")

        for alias in updated_node.names:
            alias_full = _get_import_name(alias.name)
            remapped = self.matcher.match_import(alias_full, self.target_framework)
            if remapped:
                target_parts = remapped.split(".")
                new_alias = alias.with_changes(name=cast(cst.Name, _build_attribute_chain(target_parts)))
                new_names.append(new_alias)
                mutated = True
                continue

            base_mod = alias_full.split(".")[0]
            if base_mod in KNOWN_SOURCE_FRAMEWORKS:
                if alias_full != target_module:
                    target_parts = target_module.split(".")
                    new_alias = alias.with_changes(name=cast(cst.Name, _build_attribute_chain(target_parts)))
                    new_names.append(new_alias)
                    mutated = True
                else:
                    new_names.append(alias)
            else:
                new_names.append(alias)

        if mutated:
            return updated_node.with_changes(names=new_names)
        return updated_node

    def leave_Call(
        self,
        original_node: cst.Call,
        updated_node: cst.Call,
    ) -> cst.BaseExpression:
        """Handle Framework-Specific Quirks & Stateful-to-Functional rewrites.

        Returns:
            cst.BaseExpression: The inferred shape or computed result

        Args:
            original_node (cst.Call): The original_node parameter.
            updated_node (cst.Call): The updated_node parameter.
        """
        if not self.target_config:
            return updated_node

        new_args: list[cst.Arg] = []
        mutated_args: bool = False
        kw_map: dict[str, str] = self.target_config.kwarg_map
        if kw_map:
            for arg in updated_node.args:
                if arg.keyword and arg.keyword.value in kw_map:
                    new_kw: cst.Name = arg.keyword.with_changes(value=kw_map[arg.keyword.value])
                    new_args.append(arg.with_changes(keyword=new_kw))
                    mutated_args = True
                else:
                    new_args.append(arg)
        else:
            new_args = list(updated_node.args)

        final_node: cst.Call = updated_node
        if mutated_args:
            final_node = final_node.with_changes(args=new_args)

        if not isinstance(final_node.func, cst.Attribute):
            return final_node

        matched_call = self.matcher.match_call(final_node, self.target_framework)
        if matched_call is not None:
            return matched_call

        func_attr_value: str = final_node.func.attr.value

        # Declarative method-to-function rewrites from config schema
        for rewrite in _CONFIG.syntactic_patterns.method_to_function_rewrites:
            if func_attr_value == rewrite.pattern:
                if rewrite.replacement == "identity":
                    return final_node.func.value
                elif self.target_framework != "pytorch":
                    return final_node.with_changes(func=final_node.func.with_changes(attr=cst.Name(rewrite.replacement)))

        # Declarative chain removals from cst_rewrite_rules.yaml
        for chain_rule in _CST_RULES.chain_removals:
            if func_attr_value == chain_rule.match.pattern:
                return final_node.func.value

        # Declarative method substitutions from cst_rewrite_rules.yaml
        for sub_rule in _CST_RULES.method_substitutions:
            if func_attr_value == sub_rule.match.pattern and self.target_framework != "pytorch" and sub_rule.replacement:
                return final_node.with_changes(func=final_node.func.with_changes(attr=cst.Name(sub_rule.replacement.target_name)))

        # Declarative in-place mutation rewrites from cst_rewrite_rules.yaml
        for inplace_rule in _CST_RULES.in_place_mutations:
            if func_attr_value == inplace_rule.match.pattern and self.target_framework != "pytorch" and inplace_rule.replacement:
                return final_node.with_changes(func=final_node.func.with_changes(attr=cst.Name(inplace_rule.replacement.target_name)))

        # Handle explicit broadcast translations
        for fw, fw_config in _CONFIG.frameworks.items():
            if fw_config.broadcast_method == func_attr_value and self.target_framework != fw:
                # Transpile broadcast call
                target_base: cst.BaseExpression = _build_attribute_chain(self.target_config.module_path + [self.target_config.broadcast_method])

                # Special cases for expand/broadcast_to args
                if fw_config.broadcast_method == "expand" and self.target_config.broadcast_method == "broadcast_to":
                    return final_node.with_changes(func=target_base, args=[cst.Arg(value=final_node.func.value)] + list(final_node.args))
                elif fw_config.broadcast_method == "broadcast_to" and self.target_config.broadcast_method == "expand":
                    return final_node.with_changes(func=cst.Attribute(value=final_node.args[0].value, attr=cst.Name("expand")), args=list(final_node.args)[1:])
                else:
                    # just change the method name
                    return final_node.with_changes(func=target_base)

        def _get_base_name(node: cst.BaseExpression) -> str:
            """Get the base name of a node."""
            if isinstance(node, cst.Name):
                return node.value
            elif isinstance(node, cst.Attribute):
                return _get_base_name(node.value)
            return ""

        src_call_base: str = _get_base_name(final_node.func)

        if src_call_base in KNOWN_SOURCE_FRAMEWORKS:
            if func_attr_value == "tensor" and self.target_framework != "pytorch":
                target_chain = self.target_config.module_path + ["array"]
                new_func_expr = _build_attribute_chain(target_chain)
                return final_node.with_changes(func=new_func_expr)

            target_chain: list[str] = self.target_config.module_path
            if target_chain != [src_call_base]:
                new_value: cst.BaseExpression = _build_attribute_chain(target_chain)
                new_func: cst.Attribute = final_node.func.with_changes(value=new_value)
                return final_node.with_changes(func=new_func)

        return final_node

    def leave_ClassDef(
        self,
        original_node: cst.ClassDef,
        updated_node: cst.ClassDef,
    ) -> cst.ClassDef:
        """Handle Stateful-to-Functional and Functional-to-Stateful class rewrites.

        Args:
            original_node (cst.ClassDef): The original node.
            updated_node (cst.ClassDef): The updated node.

        Returns:
            cst.ClassDef: The modified class definition.
        """
        if not self.target_config:
            return updated_node

        new_bases: list[cst.Arg] = []
        mutated_bases: bool = False

        def get_attr_chain(node: cst.BaseExpression) -> str:
            """Get the attribute chain."""
            if isinstance(node, cst.Name):
                return node.value
            elif isinstance(node, cst.Attribute):
                return get_attr_chain(node.value) + "." + node.attr.value
            return ""

        for base in updated_node.bases:
            chain: str = get_attr_chain(base.value)
            found: bool = False
            for src_fw, src_config in _CONFIG.frameworks.items():
                if self.target_framework == src_fw:
                    continue
                # If the base matches a class_base from another framework, we translate to target's base if any
                for src_k, src_v in src_config.class_bases.items():
                    # E.g. "nn.Module" or "flax.linen.Module"
                    src_full: str = ".".join(src_v) if src_v else ""
                    if chain == src_full or chain == src_k:
                        # Find target base
                        target_base_parts: list[str] | None = next(iter(self.target_config.class_bases.values()), None)
                        if target_base_parts:
                            new_base: cst.Arg = base.with_changes(value=_build_attribute_chain(target_base_parts))
                            new_bases.append(new_base)
                            mutated_bases = True
                            found = True
                            break
                if found:
                    break

            if not found:
                new_bases.append(base)

        if mutated_bases:
            return updated_node.with_changes(bases=new_bases)
        return updated_node

    def leave_FunctionDef(
        self,
        original_node: cst.FunctionDef,
        updated_node: cst.FunctionDef,
    ) -> cst.FunctionDef:
        """Rewrite method names based on framework conventions.

        Args:
            original_node (cst.FunctionDef): The original node.
            updated_node (cst.FunctionDef): The updated node.

        Returns:
            cst.FunctionDef: The modified function definition.
        """
        if not self.target_config:
            return updated_node

        method_map: dict[str, str] = self.target_config.method_map
        remapped_name = self.matcher.match_function_name(updated_node.name.value, self.target_framework)
        has_self = bool(updated_node.params.params and updated_node.params.params[0].name.value == "self")
        if remapped_name:
            if not (updated_node.name.value == "__call__" and self.target_framework == "pytorch" and not has_self):
                updated_node = updated_node.with_changes(name=cst.Name(remapped_name))
        elif updated_node.name.value in method_map:
            if updated_node.name.value == "__call__" and self.target_framework == "pytorch":
                if has_self:
                    updated_node = updated_node.with_changes(name=cst.Name(method_map[updated_node.name.value]))
            else:
                updated_node = updated_node.with_changes(name=cst.Name(method_map[updated_node.name.value]))

        # Lower early returns: if cond: return a; return b -> return where(cond, a, b)
        body_stmts = list(updated_node.body.body)
        if len(body_stmts) == 2:
            stmt1, stmt2 = body_stmts[0], body_stmts[1]
            if (
                isinstance(stmt1, cst.If)
                and not stmt1.orelse
                and len(stmt1.body.body) == 1
                and isinstance(stmt1.body.body[0], cst.SimpleStatementLine)
                and len(stmt1.body.body[0].body) == 1
                and isinstance(stmt1.body.body[0].body[0], cst.Return)
                and isinstance(stmt2, cst.SimpleStatementLine)
                and len(stmt2.body) == 1
                and isinstance(stmt2.body[0], cst.Return)
            ):
                cond = stmt1.test
                ret1 = stmt1.body.body[0].body[0].value
                ret2 = stmt2.body[0].value
                if ret1 is not None and ret2 is not None:
                    target_module_parts = self.target_config.module_path if self.target_config else ["jnp"]
                    where_func = _build_attribute_chain(target_module_parts + ["where"])
                    where_call = cst.Call(
                        func=where_func,
                        args=[cst.Arg(value=cond), cst.Arg(value=ret1), cst.Arg(value=ret2)],
                    )
                    new_return = cst.SimpleStatementLine(body=[cst.Return(value=where_call)])
                    updated_node = updated_node.with_changes(body=updated_node.body.with_changes(body=[new_return]))

        return updated_node

    def leave_ListComp(
        self,
        original_node: cst.ListComp,
        updated_node: cst.ListComp,
    ) -> cst.BaseExpression:
        """Functionalize list comprehensions into map operations.

        Args:
            original_node (cst.ListComp): Original list comprehension node.
            updated_node (cst.ListComp): Updated list comprehension node.

        Returns:
            cst.BaseExpression: Functionalized Map call.
        """
        for_clause = updated_node.for_in
        elt = updated_node.elt
        target = for_clause.target
        iter_expr = for_clause.iter

        if isinstance(target, cst.Name):
            lambda_func = cst.Lambda(
                params=cst.Parameters(params=[cst.Param(name=target)]),
                body=elt,
            )
            map_call = cst.Call(
                func=cst.Name("map"),
                args=[cst.Arg(value=lambda_func), cst.Arg(value=iter_expr)],
            )
            return cst.Call(
                func=cst.Name("list"),
                args=[cst.Arg(value=map_call)],
            )
        return updated_node

    def leave_GeneratorExp(
        self,
        original_node: cst.GeneratorExp,
        updated_node: cst.GeneratorExp,
    ) -> cst.BaseExpression:
        """Functionalize generator expressions into Scan/Map operations.

        Args:
            original_node (cst.GeneratorExp): Original generator expression.
            updated_node (cst.GeneratorExp): Updated generator expression.

        Returns:
            cst.BaseExpression: Functionalized generator expression.
        """
        for_clause = updated_node.for_in
        elt = updated_node.elt
        target = for_clause.target
        iter_expr = for_clause.iter

        if isinstance(target, cst.Name):
            lambda_func = cst.Lambda(
                params=cst.Parameters(params=[cst.Param(name=target)]),
                body=elt,
            )
            return cst.Call(
                func=cst.Name("map"),
                args=[cst.Arg(value=lambda_func), cst.Arg(value=iter_expr)],
            )
        return updated_node

    def leave_IfExp(
        self,
        original_node: cst.IfExp,
        updated_node: cst.IfExp,
    ) -> cst.BaseExpression:
        """Lower ternary conditional expressions into functional Cond/where calls.

        Args:
            original_node (cst.IfExp): Original ternary if node.
            updated_node (cst.IfExp): Updated ternary if node.

        Returns:
            cst.BaseExpression: Lowered functional where/Cond call.
        """
        test = updated_node.test
        body = updated_node.body
        orelse = updated_node.orelse

        target_module_parts = self.target_config.module_path if self.target_config else ["jnp"]
        where_func = _build_attribute_chain(target_module_parts + ["where"])

        return cst.Call(
            func=where_func,
            args=[
                cst.Arg(value=test),
                cst.Arg(value=body),
                cst.Arg(value=orelse),
            ],
        )

    def leave_Attribute(
        self,
        original_node: cst.Attribute,
        updated_node: cst.Attribute,
    ) -> cst.BaseExpression:
        """Stateful-to-Functional rewrites.

        Args:
            original_node (cst.Attribute): The original attribute node.
            updated_node (cst.Attribute): The updated attribute node.

        Returns:
            cst.BaseExpression: The modified expression.
        """
        matched_attr = self.matcher.match_attribute(updated_node, self.target_framework)
        if matched_attr is not None:
            return matched_attr

        if isinstance(updated_node.value, cst.Name) and updated_node.value.value == "self":
            attr_name: str = updated_node.attr.value
            return cst.Subscript(value=cst.Name("state"), slice=[cst.SubscriptElement(slice=cst.Index(value=cst.SimpleString(f'"{attr_name}"')))])
        return updated_node

    def leave_AugAssign(
        self,
        original_node: cst.AugAssign,
        updated_node: cst.AugAssign,
    ) -> cst.BaseSmallStatement:
        """Functionalize in-place augmented assignments for functional target backends.

        Args:
            original_node (cst.AugAssign): The original augmented assignment node.
            updated_node (cst.AugAssign): The updated augmented assignment node.

        Returns:
            cst.BaseSmallStatement: Functionalized assignment or unchanged node.
        """
        if self.target_framework == "jax":
            op_map: dict[type[cst.BaseAugOp], cst.BaseBinaryOp] = {
                cst.AddAssign: cst.Add(),
                cst.SubtractAssign: cst.Subtract(),
                cst.MultiplyAssign: cst.Multiply(),
                cst.DivideAssign: cst.Divide(),
            }
            op_type = type(updated_node.operator)
            if op_type in op_map:
                bin_op = cst.BinaryOperation(
                    left=updated_node.target,
                    operator=op_map[op_type],
                    right=updated_node.value,
                )
                return cst.Assign(
                    targets=[cst.AssignTarget(target=updated_node.target)],
                    value=bin_op,
                )
        return updated_node


class StateLiftingTransformer(cst.CSTTransformer):
    """Universal framework-blind transformer lifting stateful OOP Python classes into pure functional functions with parameter PyTrees."""

    def __init__(self, param_var_name: str = "params") -> None:
        """Initialize StateLiftingTransformer.

        Args:
            param_var_name (str): Variable name for functional parameter dictionary / PyTree.
        """
        super().__init__()
        self.param_var_name: str = param_var_name
        self.extracted_params: dict[str, object] = {}
        self.forward_methods: list[str] = ["forward", "__call__", "compute", "call"]

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
    ) -> tuple[dict[str, object], str]:
        """Lift an OOP Python class into an extracted parameter dictionary (PyTree) and pure functional function code.

        Args:
            class_code (str): Source code containing the class definition.
            param_var_name (str): Name of the parameters argument.

        Returns:
            tuple[dict[str, object], str]: Extracted parameter dict and functional code string.
        """
        from ml_switcheroo_compiler.tree_util import tree_flatten, tree_unflatten

        tree = cst.parse_module(class_code)
        extracted_params: dict[str, object] = {}

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
                            val: object = None
                            if isinstance(node.value, cst.Integer):
                                val = int(node.value.value)
                            elif isinstance(node.value, cst.Float):
                                val = float(node.value.value)
                            elif isinstance(node.value, cst.SimpleString):
                                val = node.value.evaluated_value
                            extracted_params[param_name] = val

        tree.visit(InitExtractor())

        leaves, tree_def = tree_flatten(extracted_params)
        reconstructed: dict[str, object] = tree_unflatten(tree_def, leaves)

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
    """Parse source files while retaining 100% formatting, whitespace, comments.

    Args:
        source_code (str): The source_code parameter.
        target_framework (str): The target_framework parameter.

    Returns:
        str: Result.
    """
    tree: cst.Module = cst.parse_module(source_code)
    wrapper: cst.MetadataWrapper = cst.MetadataWrapper(tree)
    transformer: CSTTransformer = CSTTransformer(target_framework=target_framework)
    modified_tree: cst.Module = wrapper.visit(transformer)
    return modified_tree.code


transpile_cst = transpile_source


def validate_diff(source_code: str, transpiled_code: str) -> bool:
    """Implement a Source-Level Diff Validator.

    Args:
        source_code (str): The source_code parameter.
        transpiled_code (str): The transpiled_code parameter.

    Returns:
        bool: Result.
    """
    if source_code == transpiled_code:
        return False
    try:
        cst.parse_module(transpiled_code)
        return True
    except cst.ParserSyntaxError:
        return False


def type_infer_dry_run(source_code: str) -> dict[str, str]:
    """Implement Hybrid Type-Inference.

    Args:
        source_code (str): The source_code parameter.

    Returns:
        dict[str, str]: Result.
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
