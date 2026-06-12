"""Tracing engine for constructing LogicalGraphs via operator overloading."""

from __future__ import annotations

import threading
import uuid
from typing import TypeVar

from ml_switcheroo_ir import LogicalGraph

from ml_switcheroo_compiler.ir.core import IRNode
from ml_switcheroo_compiler.ir.shape_system import broadcast_shapes

T = TypeVar("T", bound="ProxyTensor")


class TracerTape(threading.local):
    """Thread-local tape for tracking active graph construction."""

    def __init__(self) -> None:
        """Initialize the tracer tape."""
        self.active_graph: LogicalGraph | None = None
        self.is_tracing: bool = False

    def start_tracing(self, name: str = "Model") -> LogicalGraph:
        """Begin tracking a new graph.

        name (str): The name of the graph

        Returns:
            LogicalGraph: The new active graph

        Args:
            name (str): Argument name
        """
        self.active_graph = LogicalGraph(name=name)
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


class ProxyTensor:
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
            shape (tuple[Union[int, str], ...]): Argument shape
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
            ProxyTensor: The resulting proxy tensor
        """
        if not _tracer.is_tracing:
            msg = f"Cannot perform {op_type} outside of a tracing context."
            raise RuntimeError(
                msg,
            )

        other_id = getattr(other, "id", None)
        other_shape = getattr(other, "shape", ())

        # Broadcast shapes
        out_shape = broadcast_shapes(self.shape, other_shape)
        out_dtype = self.dtype

        if other_id is None:
            # Constant scalar logic would wrap 'other' in a Constant node
            other_id = str(uuid.uuid4())
            const_node = IRNode(
                id=other_id,
                op_type="Constant",
                attributes={"value": other},
                shape_metadata=(),
            )
            _tracer.add_node(const_node)

        out_id = str(uuid.uuid4())
        node = IRNode(
            id=out_id,
            op_type=op_type,
            inputs=[self.id, other_id],
            shape_metadata=out_shape,
        )
        _tracer.add_node(node)

        return ProxyTensor(id=out_id, shape=out_shape, dtype=out_dtype)

    def __add__(self, other: object) -> ProxyTensor:
        """Addition."""
        return self._binary_op(other, "Add")

    def __radd__(self, other: object) -> ProxyTensor:
        """Right addition."""
        # Note: In real implementation, Constant is left-hand side
        return self._binary_op(other, "Add")

    def __sub__(self, other: object) -> ProxyTensor:
        """Subtraction."""
        return self._binary_op(other, "Sub")

    def __rsub__(self, other: object) -> ProxyTensor:
        """Right subtraction."""
        return self._binary_op(other, "Sub")

    def __mul__(self, other: object) -> ProxyTensor:
        """Multiplication."""
        return self._binary_op(other, "Mul")

    def __rmul__(self, other: object) -> ProxyTensor:
        """Right multiplication."""
        return self._binary_op(other, "Mul")

    def __truediv__(self, other: object) -> ProxyTensor:
        """Division."""
        return self._binary_op(other, "Div")

    def __rtruediv__(self, other: object) -> ProxyTensor:
        """Right division."""
        return self._binary_op(other, "Div")

    def __pow__(self, other: object) -> ProxyTensor:
        """Power."""
        return self._binary_op(other, "Pow")

    def __floordiv__(self, other: object) -> ProxyTensor:
        """Evaluate floordiv.

        Args:
            other (object): Argument other

        Returns:
            'ProxyTensor': The result of the operation
        """
        return self._binary_op(other, "FloorDiv")

    def __rfloordiv__(self, other: object) -> ProxyTensor:
        """Evaluate reverse floordiv.

        Args:
            other (object): Argument other

        Returns:
            'ProxyTensor': The result of the operation
        """
        return self._binary_op(other, "FloorDiv")

    def __mod__(self, other: object) -> ProxyTensor:
        """Evaluate mod.

        Args:
            other (object): Argument other

        Returns:
            'ProxyTensor': The result of the operation
        """
        return self._binary_op(other, "Mod")

    def __rmod__(self, other: object) -> ProxyTensor:
        """Evaluate reverse mod.

        Args:
            other (object): Argument other

        Returns:
            'ProxyTensor': The result of the operation
        """
        return self._binary_op(other, "Mod")

    def __and__(self, other: object) -> ProxyTensor:
        """Evaluate and.

        Args:
            other (object): Argument other

        Returns:
            'ProxyTensor': The result of the operation
        """
        return self._binary_op(other, "BitwiseAnd")

    def __rand__(self, other: object) -> ProxyTensor:
        """Evaluate reverse and.

        Args:
            other (object): Argument other

        Returns:
            'ProxyTensor': The result of the operation
        """
        return self._binary_op(other, "BitwiseAnd")

    def __or__(self, other: object) -> ProxyTensor:
        """Evaluate or.

        Args:
            other (object): Argument other

        Returns:
            'ProxyTensor': The result of the operation
        """
        return self._binary_op(other, "BitwiseOr")

    def __ror__(self, other: object) -> ProxyTensor:
        """Evaluate reverse or.

        Args:
            other (object): Argument other

        Returns:
            'ProxyTensor': The result of the operation
        """
        return self._binary_op(other, "BitwiseOr")

    def __xor__(self, other: object) -> ProxyTensor:
        """Evaluate xor.

        Args:
            other (object): Argument other

        Returns:
            'ProxyTensor': The result of the operation
        """
        return self._binary_op(other, "BitwiseXor")

    def __rxor__(self, other: object) -> ProxyTensor:
        """Evaluate reverse xor.

        Args:
            other (object): Argument other

        Returns:
            'ProxyTensor': The result of the operation
        """
        return self._binary_op(other, "BitwiseXor")

    def __lshift__(self, other: object) -> ProxyTensor:
        """Evaluate lshift.

        Args:
            other (object): Argument other

        Returns:
            'ProxyTensor': The result of the operation
        """
        return self._binary_op(other, "BitShiftLeft")

    def __rlshift__(self, other: object) -> ProxyTensor:
        """Evaluate reverse lshift.

        Args:
            other (object): Argument other

        Returns:
            'ProxyTensor': The result of the operation
        """
        return self._binary_op(other, "BitShiftLeft")

    def __rshift__(self, other: object) -> ProxyTensor:
        """Evaluate rshift.

        Args:
            other (object): Argument other

        Returns:
            'ProxyTensor': The result of the operation
        """
        return self._binary_op(other, "BitShiftRight")

    def __rrshift__(self, other: object) -> ProxyTensor:
        """Evaluate reverse rshift.

        Args:
            other (object): Argument other

        Returns:
            'ProxyTensor': The result of the operation
        """
        return self._binary_op(other, "BitShiftRight")

    def _unary_op(self, op_type: str) -> ProxyTensor:
        """Evaluate unary op.

        Args:
            op_type (str): Argument op_type

        Returns:
            'ProxyTensor': The result of the operation
        """
        if not _tracer.is_tracing:
            msg = f"Cannot perform {op_type} outside of a tracing context."
            raise RuntimeError(
                msg,
            )

        out_id = str(uuid.uuid4())
        node = IRNode(
            id=out_id,
            op_type=op_type,
            inputs=[self.id],
            shape_metadata=self.shape,
        )
        _tracer.add_node(node)
        return ProxyTensor(id=out_id, shape=self.shape, dtype=self.dtype)

    def __neg__(self) -> ProxyTensor:
        """Evaluate neg.

        Returns:
            'ProxyTensor': The result of the operation
        """
        return self._unary_op("Neg")

    def __pos__(self) -> ProxyTensor:
        """Evaluate pos.

        Returns:
            'ProxyTensor': The result of the operation
        """
        return self

    def __abs__(self) -> ProxyTensor:
        """Evaluate abs.

        Returns:
            'ProxyTensor': The result of the operation
        """
        return self._unary_op("Abs")

    def __invert__(self) -> ProxyTensor:
        """Evaluate invert.

        Returns:
            'ProxyTensor': The result of the operation
        """
        return self._unary_op("BitwiseNot")

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
        """Matrix multiplication."""
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
