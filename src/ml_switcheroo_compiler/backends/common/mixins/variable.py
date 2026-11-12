"""Implementation of variable operations and utilities for the backends subsystem (`ml_switcheroo_compiler.backends.common.mixins.variable`)."""

from __future__ import annotations

from ml_switcheroo_compiler.ir.core import IRNode

from .common import CommonASTVisitor


class VariableASTVisitor(CommonASTVisitor):
    # pylint: disable=abstract-method
    """Variable and assignment AST generator mixin."""

    def visit_Assign(
        self,
        node: IRNode,
        input_vars: list[str],
        **kwargs: str | int | float | bool | None,
    ) -> str:
        """Emit target backend code or IR transformation for the Assign operation.

        Args:
            node (IRNode): Input `node` argument (node).
            input_vars (list[str]): Input `input_vars` argument (input vars).
            **kwargs (str | int | float | bool | None): Keyword args.

        Returns:
            str: Computed `str` output.
        """
        pfx = self.generator.get_fallback_prefix()
        return f"{pfx}_assign({input_vars[0]}, {input_vars[1]})"

    def visit_AssignAdd(
        self,
        node: IRNode,
        input_vars: list[str],
        **kwargs: str | int | float | bool | None,
    ) -> str:
        """Emit target backend code or IR transformation for the AssignAdd operation.

        Args:
            node (IRNode): Input `node` argument (node).
            input_vars (list[str]): Input `input_vars` argument (input vars).
            **kwargs (str | int | float | bool | None): Keyword args.

        Returns:
            str: Computed `str` output.
        """
        pfx = self.generator.get_fallback_prefix()
        return f"{pfx}_assign_add({input_vars[0]}, {input_vars[1]})"

    def visit_AssignSub(
        self,
        node: IRNode,
        input_vars: list[str],
        **kwargs: str | int | float | bool | None,
    ) -> str:
        """Emit target backend code or IR transformation for the AssignSub operation.

        Args:
            node (IRNode): Input `node` argument (node).
            input_vars (list[str]): Input `input_vars` argument (input vars).
            **kwargs (str | int | float | bool | None): Keyword args.

        Returns:
            str: Computed `str` output.
        """
        pfx = self.generator.get_fallback_prefix()
        return f"{pfx}_assign_sub({input_vars[0]}, {input_vars[1]})"
