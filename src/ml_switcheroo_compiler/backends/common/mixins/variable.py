# ruff: noqa: E501
"""Provide mixin module."""

from __future__ import annotations

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
        pfx = self.generator._get_backend_prefix()
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
        pfx = self.generator._get_backend_prefix()
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
        pfx = self.generator._get_backend_prefix()
        return f"{pfx}_assign_sub({input_vars[0]}, {input_vars[1]})"
