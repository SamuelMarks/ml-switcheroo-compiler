"""Tracing engine for constructing LogicalGraphs via operator overloading."""

import threading
from ml_switcheroo.shape import broadcast_shapes
from typing import Any, Optional, TypeVar, Union
import uuid

from ml_switcheroo_ir import LogicalGraph, LogicalNode

T = TypeVar("T", bound="ProxyTensor")


class TracerTape(threading.local):
    """Thread-local tape for tracking active graph construction."""

    def __init__(self) -> None:
        """Initialize the tracer tape."""
        self.active_graph: Optional[LogicalGraph] = None
        self.is_tracing: bool = False

    def start_tracing(self, name: str = "Model") -> LogicalGraph:
        """Begin tracking a new graph.

        Args:
            name (str): The name of the graph.

        Returns:
            LogicalGraph: The new active graph.
        """
        self.active_graph = LogicalGraph(name=name)
        self.is_tracing = True
        return self.active_graph

    def stop_tracing(self) -> Optional[LogicalGraph]:
        """Stop tracking and return the current graph.

        Returns:
            Optional[LogicalGraph]: The captured graph, if any.
        """
        graph = self.active_graph
        self.active_graph = None
        self.is_tracing = False
        return graph

    def add_node(self, node: LogicalNode) -> None:
        """Add a node to the active graph.

        Args:
            node (LogicalNode): The node to add.

        Raises:
            RuntimeError: If not currently tracing.
        """
        if not self.is_tracing or self.active_graph is None:
            raise RuntimeError("Cannot add node: not currently tracing.")

        if getattr(node, "source_ast_ref", None) is None:
            from ml_switcheroo.linker import get_source_ast_ref

            node.source_ast_ref = get_source_ast_ref(back_frames=2)

        self.active_graph.nodes[node.id] = node


# Global tracer instance
_tracer = TracerTape()


class ProxyTensor:
    """A proxy object that intercepts mathematical operations and builds the IR graph.

    Attributes:
        id (str): The ID of the LogicalNode producing this tensor.
        shape (Tuple[Union[int, str], ...]): The shape of the tensor.
        dtype (str): The data type of the tensor.
    """

    def __init__(
        self, id: str, shape: tuple[Union[int, str], ...], dtype: str = "float32"
    ) -> None:
        """Initialize a ProxyTensor.

        Args:
            id (str): Node ID producing this tensor.
            shape (Tuple[Union[int, str], ...]): Tensor shape.
            dtype (str): Tensor data type.
        """
        self.id = id
        self.shape = shape
        self.dtype = dtype

    def _binary_op(self, other: Any, op_type: str) -> "ProxyTensor":
        """Internal helper for binary operations.

        Args:
            other (Any): The right-hand side operand.
            op_type (str): The ONNX operation type (e.g., 'Add').

        Returns:
            ProxyTensor: The resulting proxy tensor.
        """
        if not _tracer.is_tracing:
            raise RuntimeError(
                f"Cannot perform {op_type} outside of a tracing context."
            )

        other_id = getattr(other, "id", None)
        other_shape = getattr(other, "shape", ())

        # Broadcast shapes
        out_shape = broadcast_shapes(self.shape, other_shape)
        out_dtype = self.dtype

        if other_id is None:
            # Constant scalar logic would wrap 'other' in a Constant node
            other_id = str(uuid.uuid4())
            const_node = LogicalNode(
                id=other_id,
                op_type="Constant",
                attributes={"value": other},
                shape_metadata=(),
            )
            _tracer.add_node(const_node)

        out_id = str(uuid.uuid4())
        node = LogicalNode(
            id=out_id,
            op_type=op_type,
            inputs=[self.id, other_id],
            shape_metadata=out_shape,
        )
        _tracer.add_node(node)

        return ProxyTensor(id=out_id, shape=out_shape, dtype=out_dtype)

    def __add__(self, other: Any) -> "ProxyTensor":
        """Addition."""
        return self._binary_op(other, "Add")

    def __radd__(self, other: Any) -> "ProxyTensor":
        """Right addition."""
        # Note: In real implementation, Constant is left-hand side.
        return self._binary_op(other, "Add")

    def __sub__(self, other: Any) -> "ProxyTensor":
        """Subtraction."""
        return self._binary_op(other, "Sub")

    def __rsub__(self, other: Any) -> "ProxyTensor":
        """Right subtraction."""
        return self._binary_op(other, "Sub")

    def __mul__(self, other: Any) -> "ProxyTensor":
        """Multiplication."""
        return self._binary_op(other, "Mul")

    def __rmul__(self, other: Any) -> "ProxyTensor":
        """Right multiplication."""
        return self._binary_op(other, "Mul")

    def __truediv__(self, other: Any) -> "ProxyTensor":
        """Division."""
        return self._binary_op(other, "Div")

    def __rtruediv__(self, other: Any) -> "ProxyTensor":
        """Right division."""
        return self._binary_op(other, "Div")

    def __pow__(self, other: Any) -> "ProxyTensor":
        """Power."""
        return self._binary_op(other, "Pow")

    def __floordiv__(self, other: Any) -> "ProxyTensor":
        """Docstring."""
        return self._binary_op(other, "FloorDiv")

    def __rfloordiv__(self, other: Any) -> "ProxyTensor":
        """Docstring."""
        return self._binary_op(other, "FloorDiv")

    def __mod__(self, other: Any) -> "ProxyTensor":
        """Docstring."""
        return self._binary_op(other, "Mod")

    def __rmod__(self, other: Any) -> "ProxyTensor":
        """Docstring."""
        return self._binary_op(other, "Mod")

    def __and__(self, other: Any) -> "ProxyTensor":
        """Docstring."""
        return self._binary_op(other, "BitwiseAnd")

    def __rand__(self, other: Any) -> "ProxyTensor":
        """Docstring."""
        return self._binary_op(other, "BitwiseAnd")

    def __or__(self, other: Any) -> "ProxyTensor":
        """Docstring."""
        return self._binary_op(other, "BitwiseOr")

    def __ror__(self, other: Any) -> "ProxyTensor":
        """Docstring."""
        return self._binary_op(other, "BitwiseOr")

    def __xor__(self, other: Any) -> "ProxyTensor":
        """Docstring."""
        return self._binary_op(other, "BitwiseXor")

    def __rxor__(self, other: Any) -> "ProxyTensor":
        """Docstring."""
        return self._binary_op(other, "BitwiseXor")

    def __lshift__(self, other: Any) -> "ProxyTensor":
        """Docstring."""
        return self._binary_op(other, "BitShiftLeft")

    def __rlshift__(self, other: Any) -> "ProxyTensor":
        """Docstring."""
        return self._binary_op(other, "BitShiftLeft")

    def __rshift__(self, other: Any) -> "ProxyTensor":
        """Docstring."""
        return self._binary_op(other, "BitShiftRight")

    def __rrshift__(self, other: Any) -> "ProxyTensor":
        """Docstring."""
        return self._binary_op(other, "BitShiftRight")

    def _unary_op(self, op_type: str) -> "ProxyTensor":
        if not _tracer.is_tracing:
            raise RuntimeError(
                f"Cannot perform {op_type} outside of a tracing context."
            )

        out_id = str(uuid.uuid4())
        node = LogicalNode(
            id=out_id,
            op_type=op_type,
            inputs=[self.id],
            shape_metadata=self.shape,
        )
        _tracer.add_node(node)
        return ProxyTensor(id=out_id, shape=self.shape, dtype=self.dtype)

    def __neg__(self) -> "ProxyTensor":
        """Docstring."""
        return self._unary_op("Neg")

    def __pos__(self) -> "ProxyTensor":
        """Docstring."""
        return self

    def __abs__(self) -> "ProxyTensor":
        """Docstring."""
        return self._unary_op("Abs")

    def __invert__(self) -> "ProxyTensor":
        """Docstring."""
        return self._unary_op("BitwiseNot")

    def __getitem__(self, key: Any) -> "ProxyTensor":
        """Docstring."""
        if not _tracer.is_tracing:
            raise RuntimeError("Cannot perform Slice outside of a tracing context.")

        out_id = str(uuid.uuid4())
        # Note: True shape tracking for slices is complex and often deferred to
        # shape inference passes in the pass manager. We approximate it here.
        node = LogicalNode(
            id=out_id,
            op_type="Slice",
            inputs=[self.id],
            attributes={"slices": str(key)},
            shape_metadata=self.shape,
        )
        _tracer.add_node(node)
        return ProxyTensor(id=out_id, shape=self.shape, dtype=self.dtype)

    def __matmul__(self, other: Any) -> "ProxyTensor":
        """Matrix multiplication."""
        from ml_switcheroo.shape import matmul_shape

        if not _tracer.is_tracing:
            raise RuntimeError("Cannot perform MatMul outside of a tracing context.")

        other_id = getattr(other, "id", None)
        if other_id is None:
            raise ValueError("MatMul right hand side must be a ProxyTensor.")

        other_shape = getattr(other, "shape", ())

        out_shape = matmul_shape(self.shape, other_shape)
        out_dtype = self.dtype

        out_id = str(uuid.uuid4())
        node = LogicalNode(
            id=out_id,
            op_type="MatMul",
            inputs=[self.id, other_id],
            shape_metadata=out_shape,
        )
        _tracer.add_node(node)

        return ProxyTensor(id=out_id, shape=out_shape, dtype=out_dtype)
