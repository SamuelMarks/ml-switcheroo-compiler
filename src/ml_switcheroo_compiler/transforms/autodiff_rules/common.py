"""Common autodiff rules definitions."""

from __future__ import annotations

import enum
import typing
from collections.abc import Callable, Sequence

if typing.TYPE_CHECKING:
    from ml_switcheroo_ir import LogicalGraph, LogicalNode


class UnconnectedGradients(enum.Enum):
    """Enum representing handling of unconnected gradients."""

    NONE = "none"
    ZERO = "zero"


VJPCallable = Callable[["LogicalGraph", "LogicalNode", str], tuple[UnconnectedGradients, ...]]
JVPCallable = Callable[["LogicalGraph", "LogicalNode", typing.Union[Sequence[str], str]], str]  # noqa: UP007


def make_zero_vjp(name: str) -> VJPCallable:
    """Create a VJP function that returns zero gradients for a given operation.

    Args:
        name (str): The name of the operation.

    Returns:
        VJPCallable: The generated VJP function.
    """

    def vjp(graph: LogicalGraph, node: LogicalNode, cotangent: str) -> tuple[UnconnectedGradients, ...]:
        """Return zero gradients for all inputs.

        Args:
            graph (LogicalGraph): The IR graph.
            node (LogicalNode): The node.
            cotangent (str): The cotangent ID.

        Returns:
            tuple[UnconnectedGradients, ...]: Tuple of ZERO values.
        """
        return tuple(UnconnectedGradients.ZERO for _ in node.inputs)

    return vjp


def make_zero_jvp(name: str) -> JVPCallable:
    """Create a JVP function that returns zero gradients for a given operation.

    Args:
        name (str): The name of the operation.

    Returns:
        JVPCallable: The generated JVP function.
    """

    def jvp(graph: LogicalGraph, node: LogicalNode, tangents: Sequence[str] | str) -> str:
        """Return empty string to represent a zero tangent.

        Args:
            graph (LogicalGraph): The IR graph.
            node (LogicalNode): The node.
            tangents (Sequence[str] | str): The input tangents.

        Returns:
            str: Empty string representing zero tangent.
        """
        return ""

    return jvp
