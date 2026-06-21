"""Base definitions for the operation registry."""

from __future__ import annotations


from abc import ABC, abstractmethod
from typing import Any, Callable, TypeVar

from ml_switcheroo_compiler.core.tensor import TensorConfig

import re
import uuid
import functools
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.tensor import Tensor
from ml_switcheroo_compiler.tracing import _tracer, ProxyTensor
from ml_switcheroo_compiler.backends.registry import get_active_backend
from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_ir import LogicalNode

T = TypeVar("T", bound="OpDef")

# Global operation registry
_OP_REGISTRY: dict[str, type[OpDef]] = {}


class OpDef(ABC):
    """Abstract base class for all operations in the compiler."""

    op_type: str = "Unknown"

    def _resolve_dtype(self, res_data: object, first_tensor: object) -> object:
        """Resolve the dtype for the eager result.

        Args:
            res_data (object): The result data.
            first_tensor (object): The first tensor argument if any.

        Returns:
            object: The resolved DType.
        """
        if hasattr(res_data, "dtype"):
            dtype_str = str(res_data.dtype)
            if "dtype" in dtype_str:
                m = re.search(r"dtype\('(.*?)'\)", dtype_str)
                if m:
                    dtype_str = m.group(1)
            if dtype_str.startswith("dtype"):
                dtype_str = "float32"
            dtype_str = dtype_str.split(".")[-1]
            try:
                return DType(dtype_str)
            except ValueError:
                return DType.Float32
        elif first_tensor is not None:
            return first_tensor.dtype
        return DType.Float32

    def _evaluate_eagerly(self, *args: object, **kwargs: object) -> object:
        """Evaluate the operation eagerly.

        Args:
            *args (object): Argument args.
            **kwargs (object): Argument kwargs.

        Returns:
            object: The result of the eager evaluation.
        """
        res_data = self.eager_eval(
            *[a.data if isinstance(a, Tensor) else a for a in args],
            **kwargs,
        )

        first_tensor = next((a for a in args if isinstance(a, Tensor)), None)
        device = first_tensor.device if first_tensor is not None else None

        dtype = self._resolve_dtype(res_data, first_tensor)
        shape = res_data.shape if hasattr(res_data, "shape") else ()
        return Tensor(res_data, TensorConfig(shape, dtype, device))

    def _create_constant_node(self, val: object, shape: tuple) -> str:

        out_id = str(uuid.uuid4())
        node = LogicalNode(
            id=out_id,
            op_type="Constant",
            attributes={"value": val},
            shape_metadata=shape,
        )
        _tracer.add_node(node)
        return out_id

    def _extract_from_tensor(self, a: object, tracer: object) -> tuple[str, tuple]:
        """Extract proxy input ID and shape from a Tensor.

        Args:
            a: The tensor.
            tracer: The tracer.

        Returns:
            The ID and shape.
        """
        if hasattr(a.data, "id"):
            return a.data.id, a.shape
        data_id = id(a.data)
        if hasattr(tracer, "constant_cache") and data_id in tracer.constant_cache:
            return tracer.constant_cache[data_id], a.shape
        val = getattr(a.data, "tolist", lambda a=a: a.data)()
        out_id = self._create_constant_node(val, a.shape)
        if hasattr(tracer, "constant_cache"):
            tracer.constant_cache[data_id] = out_id
        return out_id, a.shape

    def _extract_from_constant(self, a: object, backend: object) -> tuple[str, tuple]:
        """Extract proxy input ID and shape from a constant.

        Args:
            a: The constant.
            backend: The active backend.

        Returns:
            The ID and shape.
        """
        arr = backend.array(a)
        val = getattr(arr, "tolist", lambda arr=arr: arr)()
        shape = getattr(arr, "shape", ())
        out_id = self._create_constant_node(val, shape)
        return out_id, shape

    def _extract_proxy_inputs(self, args: tuple[Any, ...]) -> tuple[list[str], list[Any], Any]:
        """Extract proxy input IDs and shapes from arguments.

        Args:
            args (tuple[Any, ...]): The arguments.

        Returns:
            tuple[list[str], list[Any], Any]: Input IDs, shapes, and the first tensor found.
        """
        backend = get_active_backend()
        input_ids = []
        shapes = []
        first_tensor = None

        for a in args:
            if isinstance(a, Tensor):
                if first_tensor is None:
                    first_tensor = a
                out_id, shape = self._extract_from_tensor(a, _tracer)
            elif hasattr(a, "id"):
                out_id, shape = a.id, getattr(a, "shape", ())
            else:
                out_id, shape = self._extract_from_constant(a, backend)

            input_ids.append(out_id)
            shapes.append(shape)

        return input_ids, shapes, first_tensor

    def _resolve_output_dtype_and_device(
        self, first_tensor: object, kwargs: dict
    ) -> tuple[object, object]:

        out_dtype = None
        if "dtype" in kwargs:
            out_dtype = kwargs["dtype"]
        elif first_tensor is not None:
            out_dtype = first_tensor.dtype
        else:
            out_dtype = DType.Float32
        device = first_tensor.device if first_tensor is not None else None
        return out_dtype, device

    def _create_tracing_logical_node(
        self, input_ids: list[str], kwargs: dict, out_shape: tuple
    ) -> str:

        out_id = str(uuid.uuid4())
        node = LogicalNode(
            id=out_id,
            op_type=self.op_type,
            inputs=input_ids,
            attributes=kwargs,
            shape_metadata=out_shape,
        )
        _tracer.add_node(node)
        return out_id

    def _emit_tracing_node(self, *args: object, **kwargs: object) -> object:
        """Emit a logical node for tracing.

        Args:
            *args (object): Argument args.
            **kwargs (object): Argument kwargs.

        Returns:
            object: The resulting proxy tensor.
        """
        input_ids, shapes, first_tensor = self._extract_proxy_inputs(args)
        out_shape = self.infer_shape(*shapes, **kwargs)
        out_dtype, device = self._resolve_output_dtype_and_device(first_tensor, kwargs)

        out_id = self._create_tracing_logical_node(input_ids, kwargs, out_shape)

        proxy = ProxyTensor(id=out_id, shape=out_shape, dtype=out_dtype.value)
        return Tensor(proxy, TensorConfig(out_shape, out_dtype, device))

    def __call__(self, *args: object, **kwargs: object) -> object:
        """Universal dispatcher for the operation.

        Args:
            *args (object): Argument args.
            **kwargs (object): Argument kwargs.

        Returns:
            object: The result of the operation.
        """
        if config.eager_mode:
            return self._evaluate_eagerly(*args, **kwargs)

        if not _tracer.is_tracing:
            msg = f"Cannot emit {self.op_type} node outside of a tracing context."
            raise RuntimeError(msg)

        return self._emit_tracing_node(*args, **kwargs)

    @abstractmethod
    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infer the output shape(s) and dtype(s) of the operation.

            *args: Positional arguments (typically TensorSpec or shapes).
            **kwargs: Keyword arguments for the operation.

        Returns:
            The output TensorSpec(s) or shape(s).

        Args:
            *args (object): Argument args.
            **kwargs (object): Argument kwargs.

        """
        ...

    def eager_eval(self, *args: object, **kwargs: object) -> object:
        """Execute eager_eval.

        Args:
            *args (Any): Argument *args.
            **kwargs (Any): Argument **kwargs.

        Returns:
        Any: The result.
        """
        backend = get_active_backend()
        return backend.execute_op(self.op_type, *args, **kwargs)


def register_op(name: str) -> Callable[[type[T]], type[T]]:
    """Register an operation class in the global registry.

    name: The unique string name of the operation (e.g., 'Add', 'Sin').

    Returns:
    A class decorator.

    Args:
        name (str): Argument name.

    """

    def decorator(cls: type[T]) -> type[T]:
        """Decorator.

        Returns:
            type[T]: The resulting output.
        """
        if name in _OP_REGISTRY and _OP_REGISTRY[name].__name__ != cls.__name__:
            msg = f"Operation '{name}' is already registered."
            raise ValueError(msg)
        cls.op_type = name
        _OP_REGISTRY[name] = cls
        return cls

    return decorator


def get_op(name: str) -> type[OpDef]:
    """Retrieve an operation class by name.

    name: The name of the operation.

    Returns:
    The operation class.

    Raises:
    KeyError: If the operation is not registered.

    Args:
        name (str): Argument name.

    """
    if name not in _OP_REGISTRY:
        msg = f"Operation '{name}' not found in registry."
        raise KeyError(msg)
    return _OP_REGISTRY[name]


def emit_ir_node(
    graph: object,
    op_type: str,
    inputs: list[str],
    shape_metadata: object = None,
    attributes: dict | None = None,
) -> str:
    """Emit a new node into the IR graph directly.

    Args:
        graph (object): The graph.
        op_type (str): The op_type.
        inputs (list[str]): The inputs.
        shape_metadata (object): The shape_metadata.
        attributes (dict | None): The attributes.

    Returns:
        str: The computed result.
    """
    nid = f"{op_type.lower()}_{uuid.uuid4().hex[:6]}"
    node = LogicalNode(
        id=nid,
        op_type=op_type,
        inputs=inputs,
        shape_metadata=shape_metadata,
        attributes=attributes or {},
    )
    graph.nodes[nid] = node
    return nid


def dispatch_eager(op_name: str) -> Callable:
    """Execute dispatch_eager.

    Args:
        op_name (Any): Argument op_name.

    Returns:
    Any: The result.
    """

    def decorator(func: Callable) -> Callable:
        """Execute decorator.

        Args:
            func (Any): Argument func.

        Returns:
        Any: The result.
        """

        @functools.wraps(func)
        def wrapper(*args: object, **kwargs: object) -> object:
            """Execute wrapper.

            Args:
                *args (Any): Argument *args.
                **kwargs (Any): Argument **kwargs.

            Returns:
            Any: The result.
            """
            if config.eager_mode:
                backend = get_active_backend()
                # Extract raw data from Tensor args
                raw_args = [a.data if isinstance(a, Tensor) else a for a in args]
                data = backend.execute_op(op_name, *raw_args, **kwargs)
                # Find the first tensor for dtype/device
                first_tensor = next((a for a in args if isinstance(a, Tensor)), None)
                device = first_tensor.device if first_tensor is not None else None
                dtype = first_tensor.dtype if first_tensor is not None else None
                return Tensor(
                    backend.array(data), TensorConfig(backend.array(data).shape, dtype, device)
                )
            return func(*args, **kwargs)

        return wrapper

    return decorator
