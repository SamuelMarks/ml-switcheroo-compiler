"""Mixin module."""

from __future__ import annotations

from .common import CommonASTVisitor


class DistributedASTVisitor(CommonASTVisitor):
    # pylint: disable=abstract-method
    """Distributed communication AST generator mixin."""

    def visit_AllGather(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Evaluate AllGather."""
        pfx = self.generator._get_backend_prefix()
        axis = kwargs.get("axis", 0)
        return f"{pfx}_all_gather({input_vars[0]}, axis={axis})"

    def visit_AllReduce(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Evaluate AllReduce."""
        pfx = self.generator._get_backend_prefix()
        op = kwargs.get("op", "sum")
        return f"{pfx}_all_reduce({input_vars[0]}, op='{op}')"

    def visit_AllToAll(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Evaluate AllToAll."""
        pfx = self.generator._get_backend_prefix()
        split_axis = kwargs.get("split_axis", 0)
        concat_axis = kwargs.get("concat_axis", 0)
        axis_name = kwargs.get("axis_name", "")
        return f"{pfx}_all_to_all({input_vars[0]}, split_axis={split_axis}, concat_axis={concat_axis}, axis_name='{axis_name}')"
