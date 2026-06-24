"""Tracing engine for constructing LogicalGraphs via operator overloading."""

from __future__ import annotations


import threading
import uuid
from typing import TypeVar

from ml_switcheroo_ir import LogicalGraph
from ml_switcheroo_compiler.core.mixins import (
    TensorArithmeticMixin,
    TensorBitwiseMixin,
    TensorLogicalMixin,
)

from ml_switcheroo_compiler.ir.core import IRNode
from ml_switcheroo_compiler.ir.shape_system import broadcast_shapes

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
        self.active_graph: LogicalGraph | None = None
        self.is_tracing: bool = False
        self.constant_cache: dict[int, str] = {}

    def start_tracing(self, name: str = "Model") -> LogicalGraph:
        """Begin tracking a new graph.

        name (str): The name of the graph

        Returns:
            LogicalGraph: The new active graph

        Args:
            name (str): Argument name
        """
        self.active_graph = LogicalGraph(name=name)
        self.constant_cache = {}
        self.is_tracing = True
        return self.active_graph

    def stop_tracing(self) -> LogicalGraph | None:
        """Stop tracking and return the current graph.

        Returns:
            Optional[LogicalGraph]: The captured graph, if any
        """
        graph = self.active_graph
        self.active_graph = None
        self.is_tracing = False
        return graph

    def add_node(self, node: IRNode) -> None:
        """Add a node to the active graph.

        node (IRNode): The node to add

        Raises:
            RuntimeError: If not currently tracing

        Args:
            node (IRNode): Argument node
        """
        if not self.is_tracing or self.active_graph is None:
            msg = "Cannot add node: not currently tracing."
            raise RuntimeError(msg)

        if getattr(node, "source_ast_ref", None) is None:
            from ml_switcheroo_compiler.backends.linker import get_source_ast_ref

            node.source_ast_ref = get_source_ast_ref(back_frames=2)

        from ml_switcheroo_compiler.core.config import config

        if hasattr(node, "stream") and node.stream is None and config.current_stream != "default":
            node.stream = config.current_stream

        self.active_graph.nodes[node.id] = node


# Global tracer instance
_tracer = TracerTape()


class ProxyTensor(TensorArithmeticMixin, TensorBitwiseMixin, TensorLogicalMixin):
    """A proxy object that intercepts mathematical operations and builds the IR graph.

    Attributes:
        id (str): The ID of the IRNode producing this tensor
        shape (Tuple[Union[int, str], ...]): The shape of the tensor
        dtype (str): The data type of the tensor
    """

    def __init__(
        self,
        id: str,
        shape: tuple[int | str, ...],
        dtype: str = "float32",
    ) -> None:
        """Initialize a ProxyTensor.

        id (str): Node ID producing this tensor
            shape (Tuple[Union[int, str], ...]): Tensor shape
            dtype (str): Tensor data type

        Args:
            id (str): Argument id
            shape (tuple[Union[int, str], ...]): The shape of the tensor.
            dtype (str): The data type
        """
        self.id = id
        self.shape = shape
        self.dtype = dtype

    def _binary_op(self, other: object, op_type: str) -> ProxyTensor:
        """Help with binary operations.

        Args:
            other (Any): The right-hand side operand
            op_type (str): The ONNX operation type (e.g., 'Add')

        Returns:
            ProxyTensor: A tensor containing the result of the operation.
        """
        if not _tracer.is_tracing:  # pragma: no cover
            msg = f"Cannot perform {op_type} outside of a tracing context."  # pragma: no cover
            raise RuntimeError(  # pragma: no cover
                msg,
            )

        other_id = getattr(other, "id", None)  # pragma: no cover
        other_shape = getattr(other, "shape", ())  # pragma: no cover

        # Broadcast shapes
        out_shape = broadcast_shapes(self.shape, other_shape)  # pragma: no cover
        out_dtype = self.dtype  # pragma: no cover

        if other_id is None:  # pragma: no cover
            # Constant scalar logic would wrap 'other' in a Constant node
            other_id = str(uuid.uuid4())  # pragma: no cover
            const_node = IRNode(  # pragma: no cover
                id=other_id,
                op_type="Constant",
                attributes={"value": other},
                shape_metadata=(),
            )
            _tracer.add_node(const_node)  # pragma: no cover

        out_id = str(uuid.uuid4())  # pragma: no cover
        node = IRNode(  # pragma: no cover
            id=out_id,
            op_type=op_type,
            inputs=[self.id, other_id],
            shape_metadata=out_shape,
        )
        _tracer.add_node(node)  # pragma: no cover

        return ProxyTensor(id=out_id, shape=out_shape, dtype=out_dtype)  # pragma: no cover

    def _unary_op(self, op_type: str) -> ProxyTensor:
        """Evaluate unary op.

        Args:
            op_type (str): Argument op_type

        Returns:
            'ProxyTensor': The result of the operation
        """
        if not _tracer.is_tracing:  # pragma: no cover
            msg = f"Cannot perform {op_type} outside of a tracing context."  # pragma: no cover
            raise RuntimeError(  # pragma: no cover
                msg,
            )

        out_id = str(uuid.uuid4())  # pragma: no cover
        node = IRNode(  # pragma: no cover
            id=out_id,
            op_type=op_type,
            inputs=[self.id],
            shape_metadata=self.shape,
        )
        _tracer.add_node(node)  # pragma: no cover
        return ProxyTensor(id=out_id, shape=self.shape, dtype=self.dtype)  # pragma: no cover

    def __getitem__(self, key: object) -> ProxyTensor:
        """Evaluate getitem.

        Args:
            key (object): Argument key

        Returns:
            'ProxyTensor': The result of the operation
        """
        if not _tracer.is_tracing:
            msg = "Cannot perform Slice outside of a tracing context."
            raise RuntimeError(msg)

        out_id = str(uuid.uuid4())
        # Note: True shape tracking for slices is complex and often deferred to
        # shape inference passes in the pass manager. We approximate it here
        node = IRNode(
            id=out_id,
            op_type="Slice",
            inputs=[self.id],
            attributes={"slices": str(key)},
            shape_metadata=self.shape,
        )
        _tracer.add_node(node)
        return ProxyTensor(id=out_id, shape=self.shape, dtype=self.dtype)

    def __matmul__(self, other: object) -> ProxyTensor:
        """Matrix multiplication.

        Args:
            other (object): The other parameter for the operation.

        Returns:
            ProxyTensor: A tensor containing the result of the operation.
        """
        from ml_switcheroo_compiler.ir.shape_system import matmul_shape

        if not _tracer.is_tracing:
            msg = "Cannot perform MatMul outside of a tracing context."
            raise RuntimeError(msg)

        other_id = getattr(other, "id", None)
        if other_id is None:
            msg = "MatMul right hand side must be a ProxyTensor."
            raise ValueError(msg)

        other_shape = getattr(other, "shape", ())

        out_shape = matmul_shape(self.shape, other_shape)
        out_dtype = self.dtype

        out_id = str(uuid.uuid4())
        node = IRNode(
            id=out_id,
            op_type="MatMul",
            inputs=[self.id, other_id],
            shape_metadata=out_shape,
        )
        _tracer.add_node(node)

        return ProxyTensor(id=out_id, shape=out_shape, dtype=out_dtype)

    def assign(self, value: ProxyTensor) -> ProxyTensor:
        """Assign a new value to a variable proxy.

        Args:
            value (ProxyTensor): The new value to assign

        Returns:
            ProxyTensor: A proxy tensor representing the updated variable
        """
        if not _tracer.is_tracing:
            msg = "Cannot perform assign outside of a tracing context."
            raise RuntimeError(msg)

        node = _tracer.active_graph.nodes.get(self.id)
        if node is None or node.op_type not in ("ReadVariable", "AssignVariable"):
            msg = "assign() can only be called on a variable proxy."
            raise ValueError(msg)

        var_name = node.attributes.get("variable_name")

        # Constant wrapping if not a proxy tensor
        value_id = getattr(value, "id", None)
        value_shape = getattr(value, "shape", ())
        value_dtype = getattr(value, "dtype", self.dtype)

        if value_id is None:
            value_id = str(uuid.uuid4())
            const_node = IRNode(
                id=value_id,
                op_type="Constant",
                attributes={"value": value},
                shape_metadata=(),
            )
            _tracer.add_node(const_node)

        out_id = str(uuid.uuid4())
        assign_node = IRNode(
            id=out_id,
            op_type="AssignVariable",
            inputs=[value_id],
            attributes={"variable_name": var_name},
            shape_metadata=value_shape,
        )
        _tracer.add_node(assign_node)

        return ProxyTensor(id=out_id, shape=value_shape, dtype=value_dtype)

    def assign_add(self, value: ProxyTensor) -> ProxyTensor:
        """Add value to variable proxy and return updated proxy.

        Args:
            value (ProxyTensor): The value to add

        Returns:
            ProxyTensor: A proxy tensor representing the updated variable
        """
        return self.assign(self + value)

    def assign_sub(self, value: ProxyTensor) -> ProxyTensor:
        """Subtract value from variable proxy and return updated proxy.

        Args:
            value (ProxyTensor): The value to subtract

        Returns:
            ProxyTensor: A proxy tensor representing the updated variable
        """
        return self.assign(self - value)
