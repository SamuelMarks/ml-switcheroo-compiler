"""Tracing engine for constructing LogicalGraphs via operator overloading."""

from __future__ import annotations

import collections
import threading
import typing
import weakref
from collections.abc import Sequence

from ml_switcheroo_ir import LogicalGraph

from ml_switcheroo_compiler.core.mixins import (
    TensorArithmeticMixin,
    TensorBitwiseMixin,
    TensorLogicalMixin,
)

if typing.TYPE_CHECKING:
    from ml_switcheroo_ir import IRNode, NodeDict, TensorSpec

    from ml_switcheroo_compiler.core.shape import OpValue, ShapeDim, ShapeTuple
from ml_switcheroo_compiler.tracing.state import global_tracing_state as global_tracing_state
from ml_switcheroo_compiler.tracing.tracer_mixins import ProxyMathOverloadsMixin

T = typing.TypeVar("T", bound="ProxyTensor")


@typing.runtime_checkable
class _ActiveGraphInputsProtocol(typing.Protocol):
    """Protocol for graph objects exposing `inputs` and `nodes`."""

    inputs: list[str]
    nodes: NodeDict


@typing.runtime_checkable
class _ActiveGraphSpecsProtocol(typing.Protocol):
    """Protocol for graph objects exposing `input_specs`."""

    input_specs: dict[str, TensorSpec]


_TRACE_COUNTS_LOCK = threading.Lock()
_WEAK_TRACE_COUNTS: weakref.WeakKeyDictionary[OpValue, int] = weakref.WeakKeyDictionary()
_FALLBACK_TRACE_COUNTS: collections.OrderedDict[int, int] = collections.OrderedDict()
_MAX_FALLBACK_TRACE_ENTRIES = 1024


def get_trace_count(func: OpValue) -> int:
    """Return the number of times the function has been traced.

    Args:
        func (OpValue): The function to check.

    Returns:
        int: The number of times the function has been traced.
    """
    with _TRACE_COUNTS_LOCK:
        try:
            return _WEAK_TRACE_COUNTS.get(func, 0)
        except TypeError:
            return _FALLBACK_TRACE_COUNTS.get(id(func), 0)


def increment_trace_count(func: OpValue) -> None:
    """Increment the trace count for the given function.

    Args:
        func (OpValue): The function to increment the trace count for.
    """
    with _TRACE_COUNTS_LOCK:
        try:
            _WEAK_TRACE_COUNTS[func] = _WEAK_TRACE_COUNTS.get(func, 0) + 1
        except TypeError:
            fid = id(func)
            _FALLBACK_TRACE_COUNTS[fid] = _FALLBACK_TRACE_COUNTS.get(fid, 0) + 1
            _FALLBACK_TRACE_COUNTS.move_to_end(fid)
            while len(_FALLBACK_TRACE_COUNTS) > _MAX_FALLBACK_TRACE_ENTRIES:
                _FALLBACK_TRACE_COUNTS.popitem(last=False)


def reset_trace_count(func: OpValue) -> None:
    """Reset the trace count for the given function.

    Args:
        func (OpValue): The function to reset the trace count for.
    """
    with _TRACE_COUNTS_LOCK:
        try:
            _WEAK_TRACE_COUNTS.pop(func, None)
        except TypeError:
            _FALLBACK_TRACE_COUNTS.pop(id(func), None)


class TracerTape:
    """Context-isolated tape facade delegating to `ContextVar`-backed `global_tracing_state`."""

    def __init__(self) -> None:
        """Initialize the tracer tape."""

    def start_tracing(self, name: str = "Model") -> LogicalGraph:
        """Begin tracking operations and constructing a new computational graph.

        Args:
            name (str): The string identifier used to name the newly created graph.

        Returns:
            LogicalGraph: The newly initialized empty graph ready for node insertion.
        """
        return global_tracing_state.start_tracing(name)

    def stop_tracing(self) -> LogicalGraph | None:
        """Stop tracking operations and finalize the current computational graph.

        Returns:
            LogicalGraph | None: The finalized graph containing all traced operations, or None if no trace was active.
        """
        return global_tracing_state.stop_tracing()

    def add_node(self, node: IRNode) -> None:
        """Append a newly constructed intermediate representation node to the active graph.

        Args:
            node (IRNode): The intermediate representation node to insert.
        """
        global_tracing_state.add_node(node)


# Global tracer instance
_tracer = TracerTape()


class ProxyTensor(ProxyMathOverloadsMixin, TensorArithmeticMixin, TensorBitwiseMixin, TensorLogicalMixin):
    """Provide a proxy object that intercepts mathematical operations and builds the IR graph.

    Attributes:
        id (str): The ID of the IRNode producing this tensor.
        shape (tuple[int | str, ...]): The shape of the tensor.
        dtype (str): The data type of the tensor.
        sparsity (dict[str, str | int | float | bool] | None): Optional sparsity pattern metadata (e.g. CSR, COO, BCOO).
    """

    concrete_value: OpValue = None

    def __init__(
        self,
        id: str,
        shape: Sequence[ShapeDim],
        dtype: str = "float32",
        sparsity: dict[str, str | int | float | bool] | None = None,
    ) -> None:
        """Initialize a ProxyTensor.

        Args:
            id (str): Node ID producing this tensor.
            shape (Sequence[ShapeDim]): Tensor shape.
            dtype (str, optional): Tensor data type. Defaults to "float32".
            sparsity (dict[str, str | int | float | bool] | None, optional): Sparsity pattern metadata. Defaults to
                None.
        """
        self.id = id
        self.shape: ShapeTuple = tuple(shape)
        self.dtype = dtype
        self.sparsity = sparsity

        if global_tracing_state.is_tracing and global_tracing_state.active_graph is not None:
            active_g = global_tracing_state.active_graph
            if isinstance(active_g, _ActiveGraphInputsProtocol):
                if self.id not in active_g.nodes and self.id not in active_g.inputs:
                    active_g.inputs.append(self.id)
                    if isinstance(active_g, _ActiveGraphSpecsProtocol):
                        from ml_switcheroo_ir import DType as IRDType
                        from ml_switcheroo_ir import TensorSpec

                        ir_dt: IRDType | str = (
                            IRDType(self.dtype) if self.dtype in IRDType._value2member_map_ else str(self.dtype)
                        )
                        spec_shape = tuple(
                            d if isinstance(d, int) else (str(d) if d is not None else "?") for d in self.shape
                        )
                        active_g.input_specs[self.id] = TensorSpec(shape=spec_shape, dtype=ir_dt)
