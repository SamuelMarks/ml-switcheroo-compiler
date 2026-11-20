"""Import and Call expression rewriting transformer for LibCST transpilation."""

from __future__ import annotations

import libcst as cst

from ml_switcheroo_compiler.backends._cst_pattern_matcher import (
    _CONFIG,
    _CST_RULES,
    KNOWN_SOURCE_FRAMEWORKS,
    ASTPatternMatcher,
    _build_attribute_chain,
    _build_import_name_chain,
)
from ml_switcheroo_compiler.backends._cst_statement_mixin import CSTStatementRewriterMixin


class CSTTransformer(CSTStatementRewriterMixin, cst.CSTTransformer):
    """Pass registry for targeted pattern matching of native APIs."""

    def __init__(self, target_framework: str = "jax") -> None:
        """Initialize the CSTTransformer.

        Args:
            target_framework (str): Input `target_framework` argument (target framework).
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
            original_node (cst.ImportFrom): Input `original_node` argument (original node).
            updated_node (cst.ImportFrom): Input `updated_node` argument (updated node).
        """
        if not updated_node.module or not self.target_config:
            return updated_node

        def _get_base_name(node: cst.BaseExpression) -> str:
            """Extract the root module identifier from a nested CST attribute expression.

            Args:
                node (cst.BaseExpression): CST Name or Attribute node.

            Returns:
                str: Root module name string.
            """
            if isinstance(node, cst.Name):
                return node.value
            if isinstance(node, cst.Attribute):
                return _get_base_name(node.value)
            return ""

        src_module: str = _get_base_name(updated_node.module)

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
            return node.value if isinstance(node, cst.Name) else ""

        for alias in updated_node.names:
            alias_full = _get_import_name(alias.name)
            remapped = self.matcher.match_import(alias_full, self.target_framework)
            if remapped:
                target_parts = remapped.split(".")
                new_alias = alias.with_changes(name=_build_import_name_chain(target_parts))
                new_names.append(new_alias)
                mutated = True
                continue

            base_mod = alias_full.split(".")[0]
            if base_mod in KNOWN_SOURCE_FRAMEWORKS:
                if alias_full != target_module:
                    target_parts = target_module.split(".")
                    new_alias = alias.with_changes(name=_build_import_name_chain(target_parts))
                    new_names.append(new_alias)
                    mutated = True
                else:
                    new_names.append(alias)
            else:
                new_names.append(alias)

        if mutated:
            return updated_node.with_changes(names=new_names)
        return updated_node

    def _remap_call_kwargs(self, node: cst.Call) -> cst.Call:
        """Remap keyword arguments in a Call node using target framework kwarg_map.

        Args:
            node (cst.Call): Input Call node.

        Returns:
            cst.Call: Call node with remapped keyword arguments.
        """
        if not self.target_config or not self.target_config.kwarg_map:
            return node
        kw_map: dict[str, str] = self.target_config.kwarg_map
        new_args: list[cst.Arg] = []
        mutated: bool = False
        for arg in node.args:
            if arg.keyword and arg.keyword.value in kw_map:
                new_kw: cst.Name = arg.keyword.with_changes(value=kw_map[arg.keyword.value])
                new_args.append(arg.with_changes(keyword=new_kw))
                mutated = True
            else:
                new_args.append(arg)
        return node.with_changes(args=new_args) if mutated else node

    def _rewrite_declarative_method_call(
        self, final_node: cst.Call, func_attr: cst.Attribute, func_attr_value: str
    ) -> cst.BaseExpression | None:
        """Apply declarative method-to-function, chain removal, substitution, and in-place rules.

        Args:
            final_node (cst.Call): Current Call node.
            func_attr (cst.Attribute): Attribute expression being called.
            func_attr_value (str): Name of the called method attribute.

        Returns:
            cst.BaseExpression | None: Rewritten expression or None if no rule matched.
        """
        for rewrite in _CONFIG.syntactic_patterns.method_to_function_rewrites:
            if func_attr_value == rewrite.pattern:
                if rewrite.replacement == "identity":
                    return func_attr.value
                if self.target_framework != "pytorch":
                    return final_node.with_changes(func=func_attr.with_changes(attr=cst.Name(rewrite.replacement)))

        for chain_rule in _CST_RULES.chain_removals:
            if func_attr_value == chain_rule.match.pattern:
                return func_attr.value

        if self.target_framework != "pytorch":
            for sub_rule in [*_CST_RULES.method_substitutions, *_CST_RULES.in_place_mutations]:
                if func_attr_value == sub_rule.match.pattern and sub_rule.replacement:
                    return final_node.with_changes(
                        func=func_attr.with_changes(attr=cst.Name(sub_rule.replacement.target_name))
                    )

        return None

    def _rewrite_broadcast_call(
        self, final_node: cst.Call, func_attr: cst.Attribute, func_attr_value: str
    ) -> cst.BaseExpression | None:
        """Rewrite framework-specific broadcast/expand method calls.

        Args:
            final_node (cst.Call): Current Call node.
            func_attr (cst.Attribute): Attribute expression being called.
            func_attr_value (str): Name of the called method attribute.

        Returns:
            cst.BaseExpression | None: Rewritten broadcast call or None if not applicable.
        """
        for fw, fw_config in _CONFIG.frameworks.items():
            if fw_config.broadcast_method == func_attr_value and self.target_framework != fw and self.target_config:
                target_base: cst.BaseExpression = _build_attribute_chain(
                    self.target_config.module_path + [self.target_config.broadcast_method]
                )
                if fw_config.broadcast_method == "expand" and self.target_config.broadcast_method == "broadcast_to":
                    return final_node.with_changes(
                        func=target_base, args=[cst.Arg(value=func_attr.value), *list(final_node.args)]
                    )
                if fw_config.broadcast_method == "broadcast_to" and self.target_config.broadcast_method == "expand":
                    return final_node.with_changes(
                        func=cst.Attribute(value=final_node.args[0].value, attr=cst.Name("expand")),
                        args=list(final_node.args)[1:],
                    )
                return final_node.with_changes(func=target_base)
        return None

    def _get_base_name(self, node: cst.BaseExpression) -> str:
        """Recursively extract the root Name identifier from an expression chain.

        Args:
            node (cst.BaseExpression): Expression node to inspect.

        Returns:
            str: Root identifier name or empty string.
        """
        if isinstance(node, cst.Name):
            return node.value
        if isinstance(node, cst.Attribute):
            return self._get_base_name(node.value)
        return ""

    def _rewrite_framework_module_call(
        self, final_node: cst.Call, func_attr: cst.Attribute, func_attr_value: str
    ) -> cst.Call:
        """Rewrite module prefix on calls originating from a known source framework.

        Args:
            final_node (cst.Call): Current Call node.
            func_attr (cst.Attribute): Attribute expression being called.
            func_attr_value (str): Name of the called method attribute.

        Returns:
            cst.Call: Updated Call node.
        """
        src_call_base: str = self._get_base_name(func_attr)
        if not self.target_config or src_call_base not in KNOWN_SOURCE_FRAMEWORKS:
            return final_node

        if func_attr_value == "tensor" and self.target_framework != "pytorch":
            target_chain = self.target_config.module_path + ["array"]
            return final_node.with_changes(func=_build_attribute_chain(target_chain))

        target_chain = self.target_config.module_path
        if target_chain != [src_call_base]:
            new_value: cst.BaseExpression = _build_attribute_chain(target_chain)
            return final_node.with_changes(func=func_attr.with_changes(value=new_value))
        return final_node

    def leave_Call(
        self,
        original_node: cst.Call,
        updated_node: cst.Call,
    ) -> cst.BaseExpression:
        """Handle Framework-Specific Quirks & Stateful-to-Functional rewrites.

        Returns:
            cst.BaseExpression: The inferred shape or computed result

        Args:
            original_node (cst.Call): Input `original_node` argument (original node).
            updated_node (cst.Call): Input `updated_node` argument (updated node).
        """
        if not self.target_config:
            return updated_node

        final_node: cst.Call = self._remap_call_kwargs(updated_node)
        if not isinstance(final_node.func, cst.Attribute):
            return final_node

        matched_call = self.matcher.match_call(final_node, self.target_framework)
        if matched_call is not None:
            return matched_call

        func_attr = final_node.func
        func_attr_value: str = func_attr.attr.value

        rewritten = self._rewrite_declarative_method_call(final_node, func_attr, func_attr_value)
        if rewritten is not None:
            return rewritten

        bcast = self._rewrite_broadcast_call(final_node, func_attr, func_attr_value)
        if bcast is not None:
            return bcast

        return self._rewrite_framework_module_call(final_node, func_attr, func_attr_value)
