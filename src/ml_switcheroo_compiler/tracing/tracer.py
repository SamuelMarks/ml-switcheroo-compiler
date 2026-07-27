"""Tracing engine for constructing LogicalGraphs via operator overloading."""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING, Any, TypeVar

from ml_switcheroo_ir import LogicalGraph

from ml_switcheroo_compiler.core.mixins import (
    TensorArithmeticMixin,
    TensorBitwiseMixin,
    TensorLogicalMixin,
)

if TYPE_CHECKING:
    from ml_switcheroo_compiler.ir.core import IRNode
from ml_switcheroo_compiler.tracing.state import global_tracing_state
from ml_switcheroo_compiler.tracing.tracer_mixins import ProxyMathOverloadsMixin

T = TypeVar("T", bound="ProxyTensor")


_TRACE_COUNTS: dict[int, int] = {}


def get_trace_count(func: object) -> int:
    """Return the number of times the function has been traced.

    Args:
        func (object): The function to check.

    Returns:
        int: The number of times the function has been traced.
    """
    return _TRACE_COUNTS.get(id(func), 0)


def increment_trace_count(func: object) -> None:
    """Increment the trace count for the given function.

    Args:
        func (object): The function to increment the trace count for.
    """
    _TRACE_COUNTS[id(func)] = get_trace_count(func) + 1


def reset_trace_count(func: object) -> None:
    """Reset the trace count for the given function.

    Args:
        func (object): The function to reset the trace count for.
    """
    if id(func) in _TRACE_COUNTS:
        del _TRACE_COUNTS[id(func)]


class TracerTape(threading.local):
    """Thread-local tape for tracking active graph construction."""

    def __init__(self) -> None:
        """Initialize the tracer tape."""
        super().__init__()

    def start_tracing(self, name: str = "Model") -> LogicalGraph:
        """Begin tracking a new graph.

        name (str): The name of the graph

        Returns:
            LogicalGraph: The new active graph

        Args:
            name (str): Argument name
        """
        return global_tracing_state.start_tracing(name)

    def stop_tracing(self) -> LogicalGraph | None:
        """Stop tracking and return the current graph.

        Returns:
            Optional[LogicalGraph]: The captured graph, if any
        """
        return global_tracing_state.stop_tracing()

    def add_node(self, node: IRNode) -> None:
        """Add a node to the active graph.

        Raises:
            RuntimeError: If not currently tracing

        Args:
            node (IRNode): Argument node
        """
        global_tracing_state.add_node(node)


# Global tracer instance
_tracer = TracerTape()


class ProxyTensor(ProxyMathOverloadsMixin, TensorArithmeticMixin, TensorBitwiseMixin, TensorLogicalMixin):
    """A proxy object that intercepts mathematical operations and builds the IR graph.

    Attributes:
        id (str): The ID of the IRNode producing this tensor
        shape (Tuple[Union[int, str], ...]): The shape of the tensor
        dtype (str): The data type of the tensor
        sparsity (dict[str, Any] | None): Optional sparsity pattern metadata (e.g. CSR, COO, BCOO)
    """

    def __init__(
        self,
        id: str,
        shape: tuple[int | str, ...],
        dtype: str = "float32",
        sparsity: dict[str, Any] | None = None,
    ) -> None:
        """Initialize a ProxyTensor.

        id (str): Node ID producing this tensor
            shape (Tuple[Union[int, str], ...]): Tensor shape
            dtype (str): Tensor data type
            sparsity (dict[str, Any] | None): Sparsity pattern metadata

        Args:
            id (str): Node ID producing this tensor
            shape (tuple[int | str, ...]): Tensor shape
            dtype (str): Tensor data type
            sparsity (dict[str, Any] | None, optional): Sparsity pattern metadata. Defaults to None.
        """
        self.id = id
        self.shape = shape
        self.dtype = dtype
        self.sparsity = sparsity
