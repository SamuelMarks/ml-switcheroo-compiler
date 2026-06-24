"""Tracing node builder module."""

import uuid

from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.tracing import ProxyTensor, _tracer
from ml_switcheroo_ir import LogicalNode
from ml_switcheroo_compiler.backends.registry import get_active_backend
from ml_switcheroo_compiler.ops.type_inference import resolve_output_dtype_and_device
from ml_switcheroo_compiler.ops.shape_inference import infer_shape


class TracingNodeBuilder:
    """Builds tracing nodes."""

    @staticmethod
    def create_constant_node(val: object, shape: tuple) -> str:
        """Docstring."""
        out_id = str(uuid.uuid4())
        node = LogicalNode(
            id=out_id,
            op_type="Constant",
            attributes={"value": val},
            shape_metadata=shape,
        )
        _tracer.add_node(node)
        return out_id

    @staticmethod
    def extract_from_tensor(a: object) -> tuple[str, tuple]:
        """Docstring."""
        if hasattr(a.data, "id"):
            return a.data.id, a.shape
        data_id = id(a.data)
        if hasattr(_tracer, "constant_cache") and data_id in _tracer.constant_cache:
            return _tracer.constant_cache[data_id], a.shape
        val = getattr(a.data, "tolist", lambda a=a: a.data)()
        out_id = TracingNodeBuilder.create_constant_node(val, a.shape)
        if hasattr(_tracer, "constant_cache"):  # pragma: no branch
            _tracer.constant_cache[data_id] = out_id
        return out_id, a.shape

    @staticmethod
    def extract_from_constant(a: object) -> tuple[str, tuple]:
        """Docstring."""
        backend = get_active_backend()
        arr = backend.array(a)
        val = getattr(arr, "tolist", lambda arr=arr: arr)()
        shape = getattr(arr, "shape", ())
        out_id = TracingNodeBuilder.create_constant_node(val, shape)
        return out_id, shape

    @staticmethod
    def extract_proxy_inputs(args: tuple[object, ...]) -> tuple[list[str], list[tuple], object]:
        """Docstring."""
        input_ids = []
        shapes = []
        first_tensor = None

        for a in args:
            if isinstance(a, Tensor):
                if first_tensor is None:
                    first_tensor = a
                out_id, shape = TracingNodeBuilder.extract_from_tensor(a)
            elif hasattr(a, "id"):
                out_id, shape = a.id, getattr(a, "shape", ())
            else:
                out_id, shape = TracingNodeBuilder.extract_from_constant(a)

            input_ids.append(out_id)
            shapes.append(shape)

        return input_ids, shapes, first_tensor

    @staticmethod
    def create_tracing_logical_node(
        op_type: str, input_ids: list[str], kwargs: dict, out_shape: tuple
    ) -> str:
        """Docstring."""
        out_id = str(uuid.uuid4())
        node = LogicalNode(
            id=out_id,
            op_type=op_type,
            inputs=input_ids,
            attributes=kwargs,
            shape_metadata=out_shape,
        )
        _tracer.add_node(node)
        return out_id

    @staticmethod
    def emit_tracing_node(op_type: str, *args: object, **kwargs: object) -> object:
        """Docstring."""
        input_ids, shapes, first_tensor = TracingNodeBuilder.extract_proxy_inputs(args)
        out_shape = infer_shape(op_type, *shapes, **kwargs)
        out_dtype, device = resolve_output_dtype_and_device(first_tensor, kwargs)

        out_id = TracingNodeBuilder.create_tracing_logical_node(
            op_type, input_ids, kwargs, out_shape
        )

        proxy = ProxyTensor(id=out_id, shape=out_shape, dtype=out_dtype.value)
        return Tensor(proxy, TensorConfig(out_shape, out_dtype, device))
