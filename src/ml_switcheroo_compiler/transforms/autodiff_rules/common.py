"""Common autodiff rules definitions."""

import enum
from typing import Callable


class UnconnectedGradients(enum.Enum):
    """Enum representing handling of unconnected gradients."""

    NONE = "none"
    ZERO = "zero"


def make_zero_vjp(name: str) -> Callable:
    """Create a VJP function that returns zero gradients for a given operation.

    Args:
        name (str): The name of the operation.

    Returns:
        Callable: The generated VJP function.
    """

    def vjp(graph: object, node: object, cotangent: str) -> tuple:
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


def make_zero_jvp(name: str) -> Callable:
    """Create a JVP function that returns zero gradients for a given operation.

    Args:
        name (str): The name of the operation.

    Returns:
        Callable: The generated JVP function.
    """

    def jvp(graph: object, node: object, tangents: tuple) -> str:
        """Return None to represent a zero tangent.

        Args:
            graph (object): The IR graph.
            node (object): The node.
            tangents (tuple): The input tangents.

        Returns:
            str: None.
        """
        return None

    return jvp
