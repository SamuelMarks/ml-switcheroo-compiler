from __future__ import annotations

# ruff: noqa: E402, D100, D103, D104, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, D101, D102, D107, E701, E722, F403, E711, E712, PLR0913, PLR0915
"""Provide mixin module."""
from typing import Any

from .common import CommonASTVisitor


class DistributedASTVisitor(CommonASTVisitor):
    # pylint: disable=abstract-method
    """Distributed communication AST generator mixin."""

    def visit_AllGather(self, node: Any, input_vars: list[str], **kwargs: Any) -> str:
        """Evaluate visit_AllGather operation.

        Args:
        node (object): The node parameter.
        input_vars (object): The input_vars parameter.
        **kwargs (object): Keyword args.

        Returns:
        str: Result.
        """
        pfx = self.generator._get_backend_prefix()
        axis = kwargs.get("axis", 0)
        return f"{pfx}_all_gather({input_vars[0]}, axis={axis})"

    def visit_AllReduce(self, node: Any, input_vars: list[str], **kwargs: Any) -> str:
        """Evaluate visit_AllReduce operation.

        Args:
        node (object): The node parameter.
        input_vars (object): The input_vars parameter.
        **kwargs (object): Keyword args.

        Returns:
        str: Result.
        """
        pfx = self.generator._get_backend_prefix()
        op = kwargs.get("op", "sum")
        return f"{pfx}_all_reduce({input_vars[0]}, op='{op}')"

    def visit_AllToAll(self, node: Any, input_vars: list[str], **kwargs: Any) -> str:
        """Evaluate visit_AllToAll operation.

        Args:
        node (object): The node parameter.
        input_vars (object): The input_vars parameter.
        **kwargs (object): Keyword args.

        Returns:
        str: Result.
        """
        pfx = self.generator._get_backend_prefix()
        split_axis = kwargs.get("split_axis", 0)
        concat_axis = kwargs.get("concat_axis", 0)
        axis_name = kwargs.get("axis_name", "")
        return f"{pfx}_all_to_all({input_vars[0]}, split_axis={split_axis}, concat_axis={concat_axis}, axis_name='{axis_name}')"
