"""Base definitions for the operation registry."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable, TypeVar

T = TypeVar("T", bound="OpDef")

# Global operation registry
_OP_REGISTRY: dict[str, type[OpDef]] = {}


class OpDef(ABC):
    """Abstract base class for all operations in the compiler."""

    op_type: str = "Unknown"

    def __call__(self, *args: object, **kwargs: object) -> object:
        """Universal dispatcher for the operation.

        Args:
            *args (object): Argument args.
            **kwargs (object): Argument kwargs.

        Returns:
            object: The result of the operation.
        """
        import uuid

        import numpy as np
        from ml_switcheroo_ir import LogicalNode

        from ml_switcheroo_compiler.core.config import config
        from ml_switcheroo_compiler.core.dtype import DType
        from ml_switcheroo_compiler.core.tensor import Tensor
        from ml_switcheroo_compiler.tracing import ProxyTensor, _tracer

        if config.eager_mode:
            # Extract underlying data

            res_data = self.numpy_eval(
                *[a.data if isinstance(a, Tensor) else a for a in args],
                **kwargs,
            )

            # Find the first Tensor to inherit device/dtype (or default)
            first_tensor = next((a for a in args if isinstance(a, Tensor)), None)
            device = first_tensor.device if first_tensor is not None else None

            # Simple heuristic for dtype
            if hasattr(res_data, "dtype"):
                from ml_switcheroo_compiler.core.dtype import DType

                dtype = DType(str(res_data.dtype))
            elif first_tensor is not None:
                dtype = first_tensor.dtype
            else:
                dtype = DType.Float32

            shape = res_data.shape if hasattr(res_data, "shape") else ()
            return Tensor(data=res_data, shape=shape, dtype=dtype, device=device)
        if not _tracer.is_tracing:
            msg = f"Cannot emit {self.op_type} node outside of a tracing context."
            raise RuntimeError(
                msg,
            )

        # Extract proxy IDs and shapes
        input_ids = []
        shapes = []
        first_tensor = None
        for a in args:
            if isinstance(a, Tensor):
                if first_tensor is None:
                    first_tensor = a
                if hasattr(a.data, "id"):
                    input_ids.append(a.data.id)
                else:
                    # Eager tensor passed during tracing, lift to constant
                    out_id = str(uuid.uuid4())
                    val = getattr(a.data, "tolist", lambda a=a: a.data)()
                    node = LogicalNode(
                        id=out_id,
                        op_type="Constant",
                        attributes={"value": val},
                        shape_metadata=a.shape,
                    )
                    _tracer.add_node(node)
                    input_ids.append(out_id)
                shapes.append(a.shape)
            elif hasattr(a, "id"):  # It might be a ProxyTensor directly
                input_ids.append(a.id)
                shapes.append(a.shape)
            else:
                # Primitive/list passed, lift to constant
                import numpy as np

                arr = np.array(a)
                out_id = str(uuid.uuid4())
                node = LogicalNode(
                    id=out_id,
                    op_type="Constant",
                    attributes={"value": arr.tolist()},
                    shape_metadata=arr.shape,
                )
                _tracer.add_node(node)
                input_ids.append(out_id)
                shapes.append(arr.shape)

        # Infer shape using the op's infer_shape method
        # Many infer_shape methods expect x, y (which are shapes).
        # Some expect shapes, some expect the objects.
        # Currently `shape_inference_pass` passes shapes. So we pass shapes!
        out_shape = self.infer_shape(*shapes, **kwargs)

        # Determine output dtype
        out_dtype = None
        if "dtype" in kwargs:
            out_dtype = kwargs["dtype"]
        elif first_tensor is not None:
            out_dtype = first_tensor.dtype
        else:
            out_dtype = DType.Float32

        # Emit Node
        out_id = str(uuid.uuid4())
        node = LogicalNode(
            id=out_id,
            op_type=self.op_type,
            inputs=input_ids,
            attributes=kwargs,
            shape_metadata=out_shape,
        )
        _tracer.add_node(node)

        proxy = ProxyTensor(id=out_id, shape=out_shape, dtype=out_dtype.value)
        device = first_tensor.device if first_tensor is not None else None
        return Tensor(data=proxy, shape=out_shape, dtype=out_dtype, device=device)

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

    @abstractmethod
    def numpy_eval(self, *args: object, **kwargs: object) -> object:
        """Evaluate the operation eagerly using NumPy.

            *args: Positional arguments (NumPy arrays or scalars).
            **kwargs: Keyword arguments.

        Returns:
            The result of the operation as a NumPy array or scalar.

        Args:
            *args (object): Argument args.
            **kwargs (object): Argument kwargs.

        """
        ...


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
        if name in _OP_REGISTRY:
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
    """Emit a new node into the IR graph directly."""
    import uuid

    from ml_switcheroo_ir import LogicalNode

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
