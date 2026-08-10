from __future__ import annotations

# ruff: noqa: E402, D100, D103, D104, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, D101, D102, D107, E701, E722, F403, E711, E712, PLR0913, PLR0915
"""Provide mixin module."""
from typing import Any

from .common import CommonASTVisitor


class VariableASTVisitor(CommonASTVisitor):
    # pylint: disable=abstract-method
    """Variable and assignment AST generator mixin."""

    def visit_Assign(self, node: Any, input_vars: list[str], **kwargs: Any) -> str:
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

    def visit_AssignAdd(self, node: Any, input_vars: list[str], **kwargs: Any) -> str:
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

    def visit_AssignSub(self, node: Any, input_vars: list[str], **kwargs: Any) -> str:
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
