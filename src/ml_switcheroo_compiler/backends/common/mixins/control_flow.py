"""Mixin module."""

from __future__ import annotations

from .common import CommonASTVisitor


class ControlFlowASTVisitor(CommonASTVisitor):
    # pylint: disable=abstract-method
    """Control flow AST generator mixin."""

    def visit_Scan(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Evaluate Scan."""
        pfx = self.generator._get_backend_prefix()
        # Natively, backends implement this as a specific scan.
        return f"{pfx}_scan({', '.join(input_vars)})"

    def visit_Switch(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Evaluate Switch."""
        pfx = self.generator._get_backend_prefix()
        # Fallback to a custom runner
        return f"{pfx}_switch({', '.join(input_vars)})"

    def visit_TimeDistributed(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Evaluate time distributed."""
        # Fallback implementation: we assume the frontend has provided a TimeDistributed node.
        # Natively, backends might want to generate a loop or a vmap.
        # For simplicity in this mixin, we return a function call to a backend-specific time_distributed utility.
        return f"{self.generator._get_backend_prefix()}_time_distributed({input_vars[0]}, '{node.attributes.get('wrapped_op_name', '')}')"  # pragma: no cover

    def visit_Assert(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Evaluate Assert."""
        pfx = self.generator._get_backend_prefix()
        data = kwargs.get("data", ["Assertion failed."])
        return f"{pfx}_assert({input_vars[0]}, data={data})"

    def visit_AssociativeScan(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Evaluate AssociativeScan."""
        pfx = self.generator._get_backend_prefix()
        return f"{pfx}_associative_scan({', '.join(input_vars)})"
