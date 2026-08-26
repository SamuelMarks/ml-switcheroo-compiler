# ruff: noqa: E402, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, E701, E722, F403, E711, E712, PLR0913, PLR0915
"""Common autodiff rules definitions."""

import enum
from typing import Callable


class UnconnectedGradients(enum.Enum):
    """Enum representing handling of unconnected gradients."""

    NONE = "none"
    ZERO = "zero"


def make_zero_vjp(name: str):
    """Create a VJP function that returns zero gradients for a given operation.

    Args:
        name (str): The name of the operation.

    Returns:
        Callable: The generated VJP function.
    """

    def vjp(graph, node, cotangent: str):
        """Return zero gradients for all inputs.

        Args:
            graph (object): The IR graph.
            node (object): The node.
            cotangent (str): The cotangent ID.

        Returns:
            tuple: Tuple of ZERO values.
        """
        return tuple(UnconnectedGradients.ZERO for _ in node.inputs)

    return vjp


def make_zero_jvp(name: str):
    """Create a JVP function that returns zero gradients for a given operation.

    Args:
        name (str): The name of the operation.

    Returns:
        Callable: The generated JVP function.
    """

    def jvp(graph, node, tangents) -> str:
        """Return None to represent a zero tangent.

        Args:
            graph (object): The IR graph.
            node (object): The node.
            tangents (tuple): The input tangents.

        Returns:
            str: None.
        """
        return ""

    return jvp
