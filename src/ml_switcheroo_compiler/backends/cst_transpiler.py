# ruff: noqa: E402, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, E701, E722, F403, E711, E712, PLR0913, PLR0915
"""Module cst_transpiler.py."""

import os
from typing import cast

"""Syntactic Transpilation Engine (Whitespace/Comment Preserving)."""

import libcst as cst
import libcst.matchers as m

from ml_switcheroo_compiler.backends.transpiler_config_models import load_transpiler_config

_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "transpilation_rules.yaml")
_CONFIG = load_transpiler_config(_CONFIG_PATH)

KNOWN_SOURCE_FRAMEWORKS = {"torch", "jax", "mlx", "numpy", "cupy", "dask", "keras", "tensorflow", "pytorch"}


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

        if src_module in KNOWN_SOURCE_FRAMEWORKS:
            target_module: str = self.target_config.target_module
            if target_module != src_module:
                target_parts: list[str] = target_module.split(".")
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
        for alias in updated_node.names:
            if isinstance(alias.name, cst.Name) and alias.name.value in KNOWN_SOURCE_FRAMEWORKS:
                if alias.name.value != target_module:
                    target_parts: list[str] = target_module.split(".")
                    new_alias: cst.ImportAlias = alias.with_changes(name=cast(cst.Name, _build_attribute_chain(target_parts)))
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

        func_attr_value: str = final_node.func.attr.value

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
        if updated_node.name.value in method_map:
            if updated_node.name.value == "__call__" and self.target_framework == "pytorch":
                if updated_node.params.params and updated_node.params.params[0].name.value == "self":
                    return updated_node.with_changes(name=cst.Name(method_map[updated_node.name.value]))
                return updated_node
            return updated_node.with_changes(name=cst.Name(method_map[updated_node.name.value]))

        return updated_node

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
        if isinstance(updated_node.value, cst.Name) and updated_node.value.value == "self":
            attr_name: str = updated_node.attr.value
            return cst.Subscript(value=cst.Name("state"), slice=[cst.SubscriptElement(slice=cst.Index(value=cst.SimpleString(f'"{attr_name}"')))])
        return updated_node


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
    modified_tree: cst.Module = cast(cst.Module, wrapper.visit(transformer))
    return modified_tree.code


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
