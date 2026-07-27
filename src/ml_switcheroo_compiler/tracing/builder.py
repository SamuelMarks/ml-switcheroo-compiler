"""Tracing node builder module."""

import uuid

from ml_switcheroo_ir import LogicalNode

from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.ops.shape_inference import infer_shape
from ml_switcheroo_compiler.ops.type_inference import resolve_output_dtype_and_device
from ml_switcheroo_compiler.tracing.state import global_tracing_state
from ml_switcheroo_compiler.tracing.tracer import ProxyTensor


class TracingNodeBuilder:
    """Builds tracing nodes."""

    @staticmethod
    def create_constant_node(val: object, shape: tuple) -> str:
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
    def extract_from_tensor(a: object) -> tuple[str, tuple]:
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
    def extract_from_constant(a: object) -> tuple[str, tuple]:
        """Evaluate and process the extract from constant operation.

        Args:
            a (object): Required parameter for a.

        Returns:
            tuple: The evaluated or processed output.
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
    def extract_proxy_inputs(args: tuple[object, ...]) -> tuple[list[str], list[tuple], object]:
        """Evaluate and process the extract proxy inputs operation.

        Args:
            args (tuple): Required parameter for args.

        Returns:
            tuple: The evaluated or processed output.
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
    def create_tracing_logical_node(op_type: str, input_ids: list[str], kwargs: dict, out_shape: tuple) -> str:
        """Evaluate and process the create tracing logical node operation.

        Args:
            op_type (str): Required parameter for op_type.
            input_ids (list): Required parameter for input_ids.
            kwargs (dict): Required parameter for kwargs.
            out_shape (tuple): Required parameter for out_shape.

        Returns:
            str: The evaluated or processed output.
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
    def emit_tracing_node(op_type: str, *args: object, **kwargs: object) -> object:
        """Evaluate and process the emit tracing node operation.

        Args:
            op_type (str): Required parameter for op_type.
            *args (Any): Variable positional arguments.
            **kwargs (Any): Arbitrary keyword arguments.

        Returns:
            object: The evaluated or processed output.
        """
        input_ids, shapes, first_tensor = TracingNodeBuilder.extract_proxy_inputs(args)

        out_shape = infer_shape(op_type, *shapes, **kwargs)
        out_dtype, device = resolve_output_dtype_and_device(first_tensor, kwargs)

        out_id = TracingNodeBuilder.create_tracing_logical_node(op_type, input_ids, kwargs, out_shape)

        dtype_val = out_dtype.value if hasattr(out_dtype, "value") else str(out_dtype)
        proxy = ProxyTensor(id=out_id, shape=out_shape, dtype=dtype_val)
        return Tensor(proxy, TensorConfig(out_shape, out_dtype, device))
