"""Mixins for Tracer."""

from __future__ import annotations

import typing
import uuid

from ml_switcheroo_ir import AttributeValue

from ml_switcheroo_compiler.core.shape import TensorOrScalar
from ml_switcheroo_compiler.tracing.state import global_tracing_state

if typing.TYPE_CHECKING:
    from ml_switcheroo_compiler.tracing.tracer import ProxyTensor


class ProxyMathOverloadsMixin:
    """Math Overloads Mixin."""

    id: str
    shape: tuple[int | str, ...]
    dtype: str

    def _binary_op(self, other: TensorOrScalar, op_type: str) -> ProxyTensor:
        """Trace a binary mathematical operation and append it to the computation graph.

        Args:
            other (TensorOrScalar): The right-hand side operand (either a ProxyTensor or a scalar constant).
            op_type (str): The name of the binary operation (e.g., 'Add', 'Mul').

        Returns:
            ProxyTensor: A new proxy tensor representing the result of the binary operation.

        Raises:
            TracingError: If invoked outside of an active tracing context.
        """
        if not global_tracing_state.is_tracing:
            msg = f"Cannot perform {op_type} outside of a tracing context."
            from ml_switcheroo_compiler.core.errors import TracingError

            raise TracingError(
                msg,
            )

        other_id = getattr(other, "id", None)
        other_shape = getattr(other, "shape", ())

        # Broadcast shapes
        from ml_switcheroo_compiler.ir.shape_system import broadcast_shapes

        out_shape_raw = broadcast_shapes(self.shape, other_shape)
        out_shape: tuple[int | str, ...] = tuple(d if isinstance(d, (int, str)) else str(d) for d in out_shape_raw)
        out_dtype = self.dtype

        if other_id is None:
            # Constant scalar logic would wrap 'other' in a Constant node
            other_id = str(uuid.uuid4())

            from ml_switcheroo_compiler.ir.core import IRNode

            const_val: AttributeValue = other if isinstance(other, (int, float, str, bool, bytes)) or other is None else str(other)
            const_attrs: dict[str, AttributeValue] = {"value": const_val}
            const_node = IRNode(
                id=other_id,
                op_type="Constant",
                attributes=const_attrs,
                shape_metadata=(),
            )
            global_tracing_state.add_node(const_node)

        out_id = str(uuid.uuid4())

        from ml_switcheroo_compiler.ir.core import IRNode

        node = IRNode(
            id=out_id,
            op_type=op_type,
            inputs=[self.id, other_id],
            shape_metadata=out_shape,
        )
        global_tracing_state.add_node(node)

        from ml_switcheroo_compiler.tracing.tracer import ProxyTensor

        return ProxyTensor(id=out_id, shape=out_shape, dtype=out_dtype)

    def _unary_op(self, op_type: str) -> ProxyTensor:
        """Trace a unary mathematical operation and append it to the computation graph.

        Args:
            op_type (str): The name of the unary operation (e.g., 'Neg', 'Exp').

        Returns:
            ProxyTensor: A new proxy tensor representing the result of the unary operation.

        Raises:
            TracingError: If invoked outside of an active tracing context.
        """
        if not global_tracing_state.is_tracing:
            msg = f"Cannot perform {op_type} outside of a tracing context."
            from ml_switcheroo_compiler.core.errors import TracingError

            raise TracingError(
                msg,
            )

        out_id = str(uuid.uuid4())

        from ml_switcheroo_compiler.ir.core import IRNode

        node = IRNode(
            id=out_id,
            op_type=op_type,
            inputs=[self.id],
            shape_metadata=self.shape,
        )
        global_tracing_state.add_node(node)
        from ml_switcheroo_compiler.tracing.tracer import ProxyTensor

        return ProxyTensor(id=out_id, shape=self.shape, dtype=self.dtype)

    def __getitem__(self, key: TensorOrScalar) -> ProxyTensor:
        """Trace a tensor slicing or indexing operation.

        Args:
            key (TensorOrScalar): The slice, integer, or tuple defining the sub-region to extract.

        Returns:
            ProxyTensor: A new proxy tensor representing the sliced result.

        Raises:
            TracingError: If invoked outside of an active tracing context.
        """
        if not global_tracing_state.is_tracing:
            msg = "Cannot perform Slice outside of a tracing context."
            from ml_switcheroo_compiler.core.errors import TracingError

            raise TracingError(msg)

        out_id = str(uuid.uuid4())

        from ml_switcheroo_compiler.ir.core import IRNode

        node = IRNode(
            id=out_id,
            op_type="Slice",
            inputs=[self.id],
            attributes={"slices": str(key)},
            shape_metadata=self.shape,
        )
        global_tracing_state.add_node(node)
        from ml_switcheroo_compiler.tracing.tracer import ProxyTensor

        return ProxyTensor(id=out_id, shape=self.shape, dtype=self.dtype)

    def __matmul__(self, other: TensorOrScalar) -> ProxyTensor:
        """Trace a matrix multiplication operation.

        Args:
            other (TensorOrScalar): The right-hand side proxy tensor.

        Returns:
            ProxyTensor: A new proxy tensor representing the matrix multiplication result.

        Raises:
            TracingError: If invoked outside of an active tracing context.
            ValueError: If the right-hand side is not a valid ProxyTensor.
        """
        if not global_tracing_state.is_tracing:
            msg = "Cannot perform MatMul outside of a tracing context."
            from ml_switcheroo_compiler.core.errors import TracingError

            raise TracingError(msg)

        other_id = getattr(other, "id", None)
        if other_id is None:
            msg = "MatMul right hand side must be a ProxyTensor."
            raise ValueError(msg)

        other_shape = getattr(other, "shape", ())

        from ml_switcheroo_compiler.ir.shape_system import matmul_shape

        out_shape = matmul_shape(self.shape, other_shape)
        out_dtype = self.dtype

        out_id = str(uuid.uuid4())
        out_shape_tuple: tuple[int | str, ...] = tuple(d if isinstance(d, (int, str)) else str(d) for d in out_shape)

        from ml_switcheroo_compiler.ir.core import IRNode

        node = IRNode(
            id=out_id,
            op_type="MatMul",
            inputs=[self.id, other_id],
            shape_metadata=out_shape_tuple,
        )
        global_tracing_state.add_node(node)

        from ml_switcheroo_compiler.tracing.tracer import ProxyTensor

        return ProxyTensor(id=out_id, shape=out_shape_tuple, dtype=out_dtype)

    def assign(self, value: ProxyTensor | TensorOrScalar) -> ProxyTensor:
        """Trace an assignment operation, updating a variable's state.

        Args:
            value (ProxyTensor | TensorOrScalar): The incoming tensor to assign to the variable.

        Returns:
            ProxyTensor: A proxy tensor representing the updated state.

        Raises:
            TracingError: If invoked outside of an active tracing context.
            ValueError: If the current proxy is not bound to a variable.
        """
        if not global_tracing_state.is_tracing or global_tracing_state.active_graph is None:
            msg = "Cannot perform assign outside of a tracing context."
            from ml_switcheroo_compiler.core.errors import TracingError

            raise TracingError(msg)

        node = global_tracing_state.active_graph.nodes.get(self.id)
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

            from ml_switcheroo_compiler.ir.core import IRNode

            const_val: AttributeValue = value if isinstance(value, (int, float, str, bool, bytes)) or value is None else str(value)
            const_attrs: dict[str, AttributeValue] = {"value": const_val}
            const_node = IRNode(
                id=value_id,
                op_type="Constant",
                attributes=const_attrs,
                shape_metadata=(),
            )
            global_tracing_state.add_node(const_node)

        out_id = str(uuid.uuid4())

        from ml_switcheroo_compiler.ir.core import IRNode

        assign_node = IRNode(
            id=out_id,
            op_type="AssignVariable",
            inputs=[value_id],
            attributes={"variable_name": var_name},
            shape_metadata=value_shape,
        )
        global_tracing_state.add_node(assign_node)

        from ml_switcheroo_compiler.tracing.tracer import ProxyTensor

        return ProxyTensor(id=out_id, shape=value_shape, dtype=value_dtype)

    def assign_add(self, value: ProxyTensor) -> ProxyTensor:
        """Add value to variable proxy and return updated proxy.

        Args:
            value (ProxyTensor): The value to add

        Returns:
            ProxyTensor: A proxy tensor representing the updated variable
        """
        return self.assign(self._binary_op(value, "Add"))

    def assign_sub(self, value: ProxyTensor) -> ProxyTensor:
        """Subtract value from variable proxy and return updated proxy.

        Args:
            value (ProxyTensor): The value to subtract

        Returns:
            ProxyTensor: A proxy tensor representing the updated variable
        """
        return self.assign(self._binary_op(value, "Sub"))
