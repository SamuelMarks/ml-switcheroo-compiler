"""Common autodiff rules definitions."""

import enum
from typing import Callable


class UnconnectedGradients(enum.Enum):
    """Enum representing handling of unconnected gradients."""

    NONE = "none"
    ZERO = "zero"


def make_zero_vjp(name: str) -> Callable:
    """Make zero VJP."""

    def vjp(graph: object, node: object, cotangent: str) -> tuple:
        return tuple(UnconnectedGradients.ZERO for _ in node.inputs)

    return vjp


def make_zero_jvp(name: str) -> Callable:
    """Make zero JVP."""

    def jvp(graph: object, node: object, tangents: tuple) -> str:
        return None

    return jvp
