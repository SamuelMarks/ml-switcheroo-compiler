# ruff: noqa: E402, D100, D103, D104, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, D101, D102, D107, E701, E722, F403, E711, E712, PLR0913, PLR0915
from typing import Any

"""Syntactic Transpilation Engine (Whitespace/Comment Preserving)."""

import libcst as cst

TARGET_CONFIGS: dict[str, tuple[str, list[str]]] = {
    "jax": ("jax", ["jax", "numpy"]),
    "mlx": ("mlx.core", ["mlx", "core"]),
    "pytorch": ("torch", ["torch"]),
    "keras": ("keras", ["keras", "ops"]),
    "dask": ("dask.array", ["dask", "array"]),
    "cupy": ("cupy", ["cupy"]),
    "numpy": ("numpy", ["numpy"]),
}

KNOWN_SOURCE_FRAMEWORKS = {"torch", "jax", "mlx", "numpy", "cupy", "dask", "keras", "tensorflow"}


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

        target_framework (str): The target framework to transpile to

        Args:
            target_framework (str): The target_framework parameter.
        """
        super().__init__()
        self.target_framework = target_framework

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
        if not updated_node.module:
            return updated_node

        if isinstance(updated_node.module, cst.Name):
            src_module = updated_node.module.value
        elif isinstance(updated_node.module, cst.Attribute):
            # Extremely basic check: if it's something like mlx.core
            # We'll just look at the root base value
            def _get_base_name(node: cst.BaseExpression) -> str:
                if isinstance(node, cst.Name):
                    return node.value
                elif isinstance(node, cst.Attribute):
                    return _get_base_name(node.value)
                return ""

            src_module = _get_base_name(updated_node.module)
        else:
            return updated_node

        if src_module in KNOWN_SOURCE_FRAMEWORKS and self.target_framework in TARGET_CONFIGS:
            target_module = TARGET_CONFIGS[self.target_framework][0]
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
        if self.target_framework not in TARGET_CONFIGS:
            return updated_node

        new_names = []
        mutated = False
        target_module = TARGET_CONFIGS[self.target_framework][0]
        for alias in updated_node.names:
            if isinstance(alias.name, cst.Name) and alias.name.value in KNOWN_SOURCE_FRAMEWORKS:
                if alias.name.value != target_module:
                    target_parts = target_module.split(".")
                    new_alias = alias.with_changes(name=_build_attribute_chain(target_parts))
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
        if not isinstance(updated_node.func, cst.Attribute):
            return updated_node

        def _get_base_name(node: cst.BaseExpression) -> str:
            if isinstance(node, cst.Name):
                return node.value
            elif isinstance(node, cst.Attribute):
                return _get_base_name(node.value)
            return ""

        src_call_base = _get_base_name(updated_node.func)

        if src_call_base in KNOWN_SOURCE_FRAMEWORKS and self.target_framework in TARGET_CONFIGS:
            target_chain = TARGET_CONFIGS[self.target_framework][1]
            if target_chain != [src_call_base]:
                new_value = _build_attribute_chain(target_chain)
                new_func = updated_node.func.with_changes(value=new_value)
                return updated_node.with_changes(func=new_func)

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
            attr_name = updated_node.attr.value
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
    tree = cst.parse_module(source_code)
    wrapper = cst.MetadataWrapper(tree)
    transformer = CSTTransformer(target_framework=target_framework)
    modified_tree = wrapper.visit(transformer)
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
        tree = cst.parse_module(source_code)
        visitor = TypeInferenceVisitor()
        tree.visit(visitor)
        res = visitor.inferred_types
        res["dry_run"] = "success"
        return res
    except cst.ParserSyntaxError:
        return {"dry_run": "failed"}
