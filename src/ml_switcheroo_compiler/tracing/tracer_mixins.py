# ruff: noqa: E402, D100, D103, D104, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, D101, D102, D107, E701, E722, F403, E711, E712, PLR0913, PLR0915
from typing import Any

"""Mixins for Tracer."""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ml_switcheroo_compiler.tracing.tracer import ProxyTensor


import uuid

from ml_switcheroo_compiler.tracing.state import global_tracing_state


class ProxyMathOverloadsMixin:
    """Math Overloads Mixin."""

    def _binary_op(self, other: Any, op_type: str) -> "ProxyTensor":
        """Trace a binary mathematical operation and append it to the computation graph.

        Args:
            other (object): The right-hand side operand (either a ProxyTensor or a scalar constant).
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

        out_shape = broadcast_shapes(self.shape, other_shape)  # type: ignore  # Justification: Polymorphic / Duck Typing for Framework Agnosticism
        out_dtype = self.dtype  # type: ignore  # Justification: Polymorphic / Duck Typing for Framework Agnosticism

        if other_id is None:
            # Constant scalar logic would wrap 'other' in a Constant node
            other_id = str(uuid.uuid4())

            from ml_switcheroo_compiler.ir.core import IRNode

            const_node = IRNode(
                id=other_id,
                op_type="Constant",
                attributes={"value": other},
                shape_metadata=(),
            )
            global_tracing_state.add_node(const_node)

        out_id = str(uuid.uuid4())

        from ml_switcheroo_compiler.ir.core import IRNode

        node = IRNode(
            id=out_id,
            op_type=op_type,
            inputs=[self.id, other_id],  # type: ignore  # Justification: Polymorphic / Duck Typing for Framework Agnosticism
            shape_metadata=out_shape,
        )
        global_tracing_state.add_node(node)

        from ml_switcheroo_compiler.tracing.tracer import ProxyTensor

        return ProxyTensor(id=out_id, shape=out_shape, dtype=out_dtype)

    def _unary_op(self, op_type: str) -> "ProxyTensor":
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
            inputs=[self.id],  # type: ignore  # Justification: Polymorphic / Duck Typing for Framework Agnosticism
            shape_metadata=self.shape,  # type: ignore  # Justification: Polymorphic / Duck Typing for Framework Agnosticism
        )
        global_tracing_state.add_node(node)
        from ml_switcheroo_compiler.tracing.tracer import ProxyTensor

        return ProxyTensor(id=out_id, shape=self.shape, dtype=self.dtype)  # type: ignore  # Justification: Polymorphic / Duck Typing for Framework Agnosticism

    def __getitem__(self, key: Any) -> "ProxyTensor":
        """Trace a tensor slicing or indexing operation.

        Args:
            key (object): The slice, integer, or tuple defining the sub-region to extract.

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
        # Note: True shape tracking for slices is complex and often deferred to
        # shape inference passes in the pass manager. We approximate it here

        from ml_switcheroo_compiler.ir.core import IRNode

        node = IRNode(
            id=out_id,
            op_type="Slice",
            inputs=[self.id],  # type: ignore  # Justification: Polymorphic / Duck Typing for Framework Agnosticism
            attributes={"slices": str(key)},
            shape_metadata=self.shape,  # type: ignore  # Justification: Polymorphic / Duck Typing for Framework Agnosticism
        )
        global_tracing_state.add_node(node)
        from ml_switcheroo_compiler.tracing.tracer import ProxyTensor

        return ProxyTensor(id=out_id, shape=self.shape, dtype=self.dtype)  # type: ignore  # Justification: Polymorphic / Duck Typing for Framework Agnosticism

    def __matmul__(self, other: Any) -> "ProxyTensor":
        """Trace a matrix multiplication operation.

        Args:
            other (object): The right-hand side proxy tensor.

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

        out_shape = matmul_shape(self.shape, other_shape)  # type: ignore  # Justification: Polymorphic / Duck Typing for Framework Agnosticism
        out_dtype = self.dtype  # type: ignore  # Justification: Polymorphic / Duck Typing for Framework Agnosticism

        out_id = str(uuid.uuid4())

        from ml_switcheroo_compiler.ir.core import IRNode

        node = IRNode(
            id=out_id,
            op_type="MatMul",
            inputs=[self.id, other_id],  # type: ignore  # Justification: Polymorphic / Duck Typing for Framework Agnosticism
            shape_metadata=out_shape,
        )
        global_tracing_state.add_node(node)

        from ml_switcheroo_compiler.tracing.tracer import ProxyTensor

        return ProxyTensor(id=out_id, shape=out_shape, dtype=out_dtype)

    def assign(self, value: "ProxyTensor") -> "ProxyTensor":
        """Trace an assignment operation, updating a variable's state.

        Args:
            value (ProxyTensor): The incoming tensor to assign to the variable.

        Returns:
            ProxyTensor: A proxy tensor representing the updated state.

        Raises:
            TracingError: If invoked outside of an active tracing context.
            ValueError: If the current proxy is not bound to a variable.
        """
        if not global_tracing_state.is_tracing:
            msg = "Cannot perform assign outside of a tracing context."
            from ml_switcheroo_compiler.core.errors import TracingError

            raise TracingError(msg)

        node = global_tracing_state.active_graph.nodes.get(self.id)  # type: ignore  # Justification: Polymorphic / Duck Typing for Framework Agnosticism
        if node is None or node.op_type not in ("ReadVariable", "AssignVariable"):
            msg = "assign() can only be called on a variable proxy."
            raise ValueError(msg)

        var_name = node.attributes.get("variable_name")

        # Constant wrapping if not a proxy tensor
        value_id = getattr(value, "id", None)
        value_shape = getattr(value, "shape", ())
        value_dtype = getattr(value, "dtype", self.dtype)  # type: ignore  # Justification: Polymorphic / Duck Typing for Framework Agnosticism

        if value_id is None:
            value_id = str(uuid.uuid4())

            from ml_switcheroo_compiler.ir.core import IRNode

            const_node = IRNode(
                id=value_id,
                op_type="Constant",
                attributes={"value": value},
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

        return ProxyTensor(id=out_id, shape=value_shape, dtype=value_dtype)  # type: ignore  # Justification: Polymorphic / Duck Typing for Framework Agnosticism

    def assign_add(self, value: "ProxyTensor") -> "ProxyTensor":
        """Add value to variable proxy and return updated proxy.

        Args:
            value (ProxyTensor): The value to add

        Returns:
            ProxyTensor: A proxy tensor representing the updated variable
        """
        return self.assign(self + value)  # type: ignore  # Justification: Polymorphic / Duck Typing for Framework Agnosticism

    def assign_sub(self, value: "ProxyTensor") -> "ProxyTensor":
        """Subtract value from variable proxy and return updated proxy.

        Args:
            value (ProxyTensor): The value to subtract

        Returns:
            ProxyTensor: A proxy tensor representing the updated variable
        """
        return self.assign(self - value)  # type: ignore  # Justification: Polymorphic / Duck Typing for Framework Agnosticism
