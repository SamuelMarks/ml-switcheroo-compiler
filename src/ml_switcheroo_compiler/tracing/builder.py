"""Implementation of builder operations and utilities for the tracing subsystem (`ml_switcheroo_compiler.tracing.builder`)."""

from __future__ import annotations

import uuid
from collections.abc import Mapping, Sequence
from typing import cast

from ml_switcheroo_ir import AttributeValue, LogicalNode

from ml_switcheroo_compiler.backends.eager_registry import EagerCallableProtocol
from ml_switcheroo_compiler.core.shape import AttrValue, TensorOrScalar
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.ops.shape_inference import infer_shape
from ml_switcheroo_compiler.ops.type_inference import resolve_output_dtype_and_device
from ml_switcheroo_compiler.tracing.state import global_tracing_state
from ml_switcheroo_compiler.tracing.tracer import ProxyTensor


class TracingNodeBuilder:
    """Build tracing nodes."""

    @staticmethod
    def create_constant_node(val: TensorOrScalar, shape: Sequence[int | str]) -> str:
        """Create a constant node in the active tracing graph.

        Args:
            val (TensorOrScalar): The constant scalar or array value.
            shape (Sequence[int | str]): The shape of the constant tensor.

        Returns:
            str: The unique identifier of the created node.
        """
        out_id = str(uuid.uuid4())
        node = LogicalNode(
            id=out_id,
            op_type="Constant",
            attributes=cast("Mapping[str, AttributeValue]", {"value": val}),
            shape_metadata=shape,
        )
        global_tracing_state.add_node(node)
        return out_id

    @staticmethod
    def extract_from_tensor(a: Tensor) -> tuple[str, tuple[int | str, ...]]:
        """Extract the node identifier and shape metadata from a tensor or proxy.

        Args:
            a (Tensor): The input tensor, proxy tensor, or raw array.

        Returns:
            tuple[str, tuple[int | str, ...]]: A tuple containing the node ID and its shape.
        """
        if hasattr(a.data, "id"):
            return getattr(a.data, "id"), a.shape
        data_id = id(a.data)
        if hasattr(global_tracing_state, "constant_cache") and data_id in global_tracing_state.constant_cache:
            return global_tracing_state.constant_cache[data_id], a.shape
        val = getattr(a.data, "tolist", lambda a=a: a.data)()
        out_id = TracingNodeBuilder.create_constant_node(val, a.shape)
        if hasattr(global_tracing_state, "constant_cache"):
            global_tracing_state.constant_cache[data_id] = out_id
        return out_id, a.shape

    @staticmethod
    def extract_from_constant(a: TensorOrScalar | ProxyTensor) -> tuple[str | list[str], tuple[int | str, ...] | list[tuple[int | str, ...]]]:
        """Extract the node identifier and shape metadata from a constant value.

        Args:
            a (TensorOrScalar | ProxyTensor): The constant scalar, list, or array value.

        Returns:
            tuple[str | list[str], tuple[int | str, ...] | list[tuple[int | str, ...]]]: A tuple containing the node ID and its shape.
        """
        if isinstance(a, (list, tuple)) and any(type(x).__name__ in ("ProxyTensor", "Tensor") for x in a):
            ids: list[str] = []
            shapes: list[tuple[int | str, ...] | list[tuple[int | str, ...]]] = []
            for x in a:
                if type(x).__name__ == "ProxyTensor":
                    ids.append(cast("ProxyTensor", x).id)
                    shapes.append(cast("ProxyTensor", x).shape)
                else:
                    oid, oshape = TracingNodeBuilder.extract_from_constant(x)
                    ids.append(cast("str", oid))
                    shapes.append(cast("tuple[int | str, ...]", oshape))
            return ids, cast("list[tuple[int | str, ...]]", shapes)

        from ml_switcheroo_compiler.backends.registry import get_active_backend

        backend = get_active_backend()
        arr = backend.array(a)
        val = getattr(arr, "tolist", lambda arr=arr: arr)()
        shape = getattr(arr, "shape", ())
        out_id = TracingNodeBuilder.create_constant_node(val, shape)
        return out_id, shape

    @staticmethod
    def extract_proxy_inputs(
        args: Sequence[TensorOrScalar | ProxyTensor],
    ) -> tuple[list[str | list[str]], list[tuple[int | str, ...] | list[tuple[int | str, ...]]], Tensor | None]:
        """Extract proxy node IDs and shapes from a list of arguments.

        Args:
            args (Sequence[TensorOrScalar | ProxyTensor]): The positional arguments.

        Returns:
            tuple[list[str | list[str]], list[tuple[int | str, ...] | list[tuple[int | str, ...]]], Tensor | None]: Node IDs, shapes, and first tensor.
        """
        input_ids: list[str | list[str]] = []
        shapes: list[tuple[int | str, ...] | list[tuple[int | str, ...]]] = []
        first_tensor: Tensor | None = None

        for a in args:
            out_id: str | list[str]
            shape: tuple[int | str, ...] | list[tuple[int | str, ...]]
            if isinstance(a, Tensor):
                first_tensor = a if first_tensor is None else first_tensor
                out_id, shape = TracingNodeBuilder.extract_from_tensor(a)
            elif hasattr(a, "id"):
                out_id, shape = getattr(a, "id"), getattr(a, "shape", ())
            else:
                out_id, shape = TracingNodeBuilder.extract_from_constant(a)

            input_ids.append(out_id)
            shapes.append(shape)

        return input_ids, shapes, first_tensor

    @staticmethod
    def create_tracing_logical_node(
        op_type: str,
        input_ids: list[str],
        kwargs: dict[str, AttrValue],
        out_shape: Sequence[int | str],
    ) -> str:
        """Create and add a new logical node to the active tracing graph.

        Args:
            op_type (str): The type of the operation.
            input_ids (list[str]): The input node IDs.
            kwargs (dict[str, AttrValue]): The attributes for the node.
            out_shape (Sequence[int | str]): The output shape metadata.

        Returns:
            str: The unique identifier of the created node.
        """
        out_id = str(uuid.uuid4())
        node = LogicalNode(
            id=out_id,
            op_type=op_type,
            inputs=input_ids,
            attributes=cast("Mapping[str, AttributeValue]", kwargs),
            shape_metadata=out_shape,
        )
        global_tracing_state.add_node(node)
        return out_id

    @staticmethod
    def emit_tracing_node(op_type: str, *args: TensorOrScalar | ProxyTensor, **kwargs: AttrValue) -> Tensor:
        """Emit a new tracing node and return its corresponding tensor.

        Args:
            op_type (str): The operation type.
            *args (TensorOrScalar | ProxyTensor): Positional arguments.
            **kwargs (AttrValue): Keyword arguments.

        Returns:
            Tensor: The newly created proxy tensor.
        """
        input_ids, shapes, first_tensor = TracingNodeBuilder.extract_proxy_inputs(args)

        out_shape = cast("EagerCallableProtocol", infer_shape)(op_type, *shapes, **kwargs)
        if out_shape is None:
            out_shape = ()
        out_dtype, device = resolve_output_dtype_and_device(first_tensor, kwargs)

        out_id = TracingNodeBuilder.create_tracing_logical_node(
            op_type,
            [str(i) for i in input_ids],
            kwargs,
            cast("Sequence[int | str]", out_shape),
        )

        dtype_val = out_dtype.value if hasattr(out_dtype, "value") else str(out_dtype)
        proxy = cast("EagerCallableProtocol", ProxyTensor)(id=out_id, shape=out_shape, dtype=dtype_val)
        return Tensor(proxy, TensorConfig(out_shape, out_dtype, device))
