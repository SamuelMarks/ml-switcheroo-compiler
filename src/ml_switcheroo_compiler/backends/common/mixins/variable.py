"""Mixin module."""

from __future__ import annotations

from .common import CommonASTVisitor


class VariableASTVisitor(CommonASTVisitor):
    # pylint: disable=abstract-method
    """Variable and assignment AST generator mixin."""

    def visit_Assign(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Evaluate Assign."""
        pfx = self.generator._get_backend_prefix()
        return f"{pfx}_assign({input_vars[0]}, {input_vars[1]})"

    def visit_AssignAdd(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Evaluate AssignAdd."""
        pfx = self.generator._get_backend_prefix()
        return f"{pfx}_assign_add({input_vars[0]}, {input_vars[1]})"

    def visit_AssignSub(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Evaluate AssignSub."""
        pfx = self.generator._get_backend_prefix()
        return f"{pfx}_assign_sub({input_vars[0]}, {input_vars[1]})"
