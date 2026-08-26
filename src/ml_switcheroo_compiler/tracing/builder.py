# ruff: noqa: E402, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, E701, E722, F403, E711, E712, PLR0913, PLR0915
"""Tracing node builder module."""

import uuid

from ml_switcheroo_ir import LogicalNode

from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.ops.shape_inference import infer_shape
from ml_switcheroo_compiler.ops.type_inference import resolve_output_dtype_and_device
from ml_switcheroo_compiler.tracing.state import global_tracing_state
from ml_switcheroo_compiler.tracing.tracer import ProxyTensor


class TracingNodeBuilder:
    """Build tracing nodes."""

    @staticmethod
    def create_constant_node(val, shape) -> str:
        """Create a constant node in the active tracing graph.

        Args:
            val (object): The constant scalar or array value.
            shape (tuple): The shape of the constant tensor.

        Returns:
            str: The unique identifier of the created node.
        """
        out_id = str(uuid.uuid4())
        node = LogicalNode(
            id=out_id,
            op_type="Constant",
            attributes={"value": val},
            shape_metadata=shape,
        )
        global_tracing_state.add_node(node)
        return out_id

    @staticmethod
    def extract_from_tensor(a) -> tuple[str, tuple]:
        """Extract the node identifier and shape metadata from a tensor or proxy.

        Args:
            a (object): The input tensor, proxy tensor, or raw array.

        Returns:
            tuple: A tuple containing the node ID and its shape.
        """
        if hasattr(a.data, "id"):
            return a.data.id, a.shape
        data_id = id(a.data)
        if hasattr(global_tracing_state, "constant_cache") and data_id in global_tracing_state.constant_cache:
            return global_tracing_state.constant_cache[data_id], a.shape
        val = getattr(a.data, "tolist", lambda a=a: a.data)()
        out_id = TracingNodeBuilder.create_constant_node(val, a.shape)
        if hasattr(global_tracing_state, "constant_cache"):
            global_tracing_state.constant_cache[data_id] = out_id
        return out_id, a.shape

    @staticmethod
    def extract_from_constant(a):
        """Extract the node identifier and shape metadata from a constant value.

        Args:
            a (object): The constant scalar, list, or array value.

        Returns:
            tuple: A tuple containing the node ID and its shape.
        """
        if isinstance(a, (list, tuple)) and any(type(x).__name__ in ("ProxyTensor", "Tensor") for x in a):
            ids, shapes = [], []
            for x in a:
                if type(x).__name__ == "ProxyTensor":
                    ids.append(x.id)
                    shapes.append(x.shape)
                else:
                    oid, oshape = TracingNodeBuilder.extract_from_constant(x)
                    ids.append(oid)
                    shapes.append(oshape)
            return ids, shapes

        from ml_switcheroo_compiler.backends.registry import get_active_backend

        backend = get_active_backend()
        arr = backend.array(a)
        val = getattr(arr, "tolist", lambda arr=arr: arr)()
        shape = getattr(arr, "shape", ())
        out_id = TracingNodeBuilder.create_constant_node(val, shape)
        return out_id, shape

    @staticmethod
    def extract_proxy_inputs(args):
        """Extract proxy node IDs and shapes from a list of arguments.

        Args:
            args (tuple[object, ...]): The positional arguments.

        Returns:
            tuple: A tuple containing lists of node IDs, shapes, and the first tensor found.
        """
        input_ids = []
        shapes = []
        first_tensor = None

        for a in args:
            if isinstance(a, Tensor):
                first_tensor = a if first_tensor is None else first_tensor
                out_id, shape = TracingNodeBuilder.extract_from_tensor(a)
            elif hasattr(a, "id"):
                out_id, shape = a.id, getattr(a, "shape", ())
            else:
                out_id, shape = TracingNodeBuilder.extract_from_constant(a)

            input_ids.append(out_id)
            shapes.append(shape)

        return input_ids, shapes, first_tensor

    @staticmethod
    def create_tracing_logical_node(op_type: str, input_ids: list[str], kwargs, out_shape) -> str:
        """Create and add a new logical node to the active tracing graph.

        Args:
            op_type (str): The type of the operation.
            input_ids (list[str]): The input node IDs.
            kwargs (dict): The attributes for the node.
            out_shape (tuple): The output shape metadata.

        Returns:
            str: The unique identifier of the created node.
        """
        out_id = str(uuid.uuid4())
        node = LogicalNode(
            id=out_id,
            op_type=op_type,
            inputs=input_ids,
            attributes=kwargs,
            shape_metadata=out_shape,
        )
        global_tracing_state.add_node(node)
        return out_id

    @staticmethod
    def emit_tracing_node(op_type: str, *args, **kwargs):
        """Emit a new tracing node and return its corresponding tensor.

        Args:
            op_type (str): The operation type.
            *args (object): Positional arguments.
            **kwargs (object): Keyword arguments.

        Returns: Tensor: The newly created proxy tensor.
        """
        input_ids, shapes, first_tensor = TracingNodeBuilder.extract_proxy_inputs(args)

        out_shape = infer_shape(op_type, *shapes, **kwargs)
        out_dtype, device = resolve_output_dtype_and_device(first_tensor, kwargs)

        out_id = TracingNodeBuilder.create_tracing_logical_node(op_type, input_ids, kwargs, out_shape)

        dtype_val = out_dtype.value if hasattr(out_dtype, "value") else str(out_dtype)
        proxy = ProxyTensor(id=out_id, shape=out_shape, dtype=dtype_val)
        return Tensor(proxy, TensorConfig(out_shape, out_dtype, device))
