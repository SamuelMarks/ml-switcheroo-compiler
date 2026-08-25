"""Module variable.py."""

from __future__ import annotations

# ruff: noqa: E402, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, E701, E722, F403, E711, E712, PLR0913, PLR0915
"""Provide mixin module."""

from .common import CommonASTVisitor


class VariableASTVisitor(CommonASTVisitor):
    # pylint: disable=abstract-method
    """Variable and assignment AST generator mixin."""

    def visit_Assign(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Evaluate visit_Assign operation.

        Args:
        node (object): The node parameter.
        input_vars (object): The input_vars parameter.
        **kwargs (object): Keyword args.

        Returns:
        str: Result.
        """
        pfx: object = self.generator.get_fallback_prefix()
        return f"{pfx}_assign({input_vars[0]}, {input_vars[1]})"

    def visit_AssignAdd(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Evaluate visit_AssignAdd operation.

        Args:
        node (object): The node parameter.
        input_vars (object): The input_vars parameter.
        **kwargs (object): Keyword args.

        Returns:
        str: Result.
        """
        pfx: object = self.generator.get_fallback_prefix()
        return f"{pfx}_assign_add({input_vars[0]}, {input_vars[1]})"

    def visit_AssignSub(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Evaluate visit_AssignSub operation.

        Args:
        node (object): The node parameter.
        input_vars (object): The input_vars parameter.
        **kwargs (object): Keyword args.

        Returns:
        str: Result.
        """
        pfx: object = self.generator.get_fallback_prefix()
        return f"{pfx}_assign_sub({input_vars[0]}, {input_vars[1]})"
