"""Module variable.py."""

from __future__ import annotations

# ruff: noqa: E402, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, E701, E722, F403, E711, E712, PLR0913, PLR0915
"""Provide mixin module."""

from typing import Any

from ml_switcheroo_compiler.ir.core import IRNode

from .common import CommonASTVisitor


class VariableASTVisitor(CommonASTVisitor):
    # pylint: disable=abstract-method
    """Variable and assignment AST generator mixin."""

    def visit_Assign(self, node: IRNode, input_vars: list[str], **kwargs: Any) -> str:
        """Evaluate visit_Assign operation.

        Args:
            node (IRNode): The node parameter.
            input_vars (list[str]): The input_vars parameter.
            **kwargs (Any): Keyword args.

        Returns:
            str: Result.
        """
        pfx = self.generator.get_fallback_prefix()
        return f"{pfx}_assign({input_vars[0]}, {input_vars[1]})"

    def visit_AssignAdd(self, node: IRNode, input_vars: list[str], **kwargs: Any) -> str:
        """Evaluate visit_AssignAdd operation.

        Args:
            node (IRNode): The node parameter.
            input_vars (list[str]): The input_vars parameter.
            **kwargs (Any): Keyword args.

        Returns:
            str: Result.
        """
        pfx = self.generator.get_fallback_prefix()
        return f"{pfx}_assign_add({input_vars[0]}, {input_vars[1]})"

    def visit_AssignSub(self, node: IRNode, input_vars: list[str], **kwargs: Any) -> str:
        """Evaluate visit_AssignSub operation.

        Args:
            node (IRNode): The node parameter.
            input_vars (list[str]): The input_vars parameter.
            **kwargs (Any): Keyword args.

        Returns:
            str: Result.
        """
        pfx = self.generator.get_fallback_prefix()
        return f"{pfx}_assign_sub({input_vars[0]}, {input_vars[1]})"
