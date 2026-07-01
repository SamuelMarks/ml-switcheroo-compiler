"""Base definitions for the operation registry."""

from __future__ import annotations

import functools
import uuid
from abc import ABC, abstractmethod
from typing import Callable, TypeVar

from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.backends.registry import get_active_backend
from ml_switcheroo_ir import LogicalNode

# We import dispatcher directly
from ml_switcheroo_compiler.ops.dispatcher import dispatch_op

T = TypeVar("T", bound="OpDef")

# Global operation registry

_OP_REGISTRY: dict[str, type[OpDef]] = {}


class OpDef(ABC):
    """Abstract base class for all operations in the compiler.

    Acts strictly as a definition schema holding operation metadata.
    """

    op_type: str = "Unknown"

    def __call__(self, *args: object, **kwargs: object) -> object:
        """Universal dispatcher for the operation."""
        return dispatch_op(self.op_type, *args, **kwargs)

    @abstractmethod
    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infer the output shape(s) and dtype(s) of the operation."""
        ...

    def eager_eval(self, *args: object, **kwargs: object) -> object:
        """Execute eager_eval."""
        backend = get_active_backend()
        return backend.execute_op(self.op_type, *args, **kwargs)


def register_op(name: str) -> Callable[[type[T]], type[T]]:
    """Register an operation class in the global registry."""

    def decorator(cls: type[T]) -> type[T]:
        """Function docstring."""
        if name in _OP_REGISTRY and _OP_REGISTRY[name].__name__ != cls.__name__:
            msg = f"Operation '{name}' is already registered."
            raise ValueError(msg)
        cls.op_type = name
        _OP_REGISTRY[name] = cls
        return cls

    return decorator


def get_op(name: str) -> type[OpDef]:
    """Retrieve an operation class by name."""
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
    """Execute dispatch_eager."""

    def decorator(func: Callable) -> Callable:
        """Function docstring.

        Args:
        func: Arg.
        """

        @functools.wraps(func)
        def wrapper(*args: object, **kwargs: object) -> object:
            """Function docstring.

            Args:
            args: Arg.
            kwargs: Arg.
            """
            if config.eager_mode:
                backend = get_active_backend()
                raw_args = [a.data if isinstance(a, Tensor) else a for a in args]
                data = backend.execute_op(op_name, *raw_args, **kwargs)
                first_tensor = next((a for a in args if isinstance(a, Tensor)), None)
                device = first_tensor.device if first_tensor is not None else None
                dtype = first_tensor.dtype if first_tensor is not None else None
                if isinstance(data, tuple):
                    return tuple(
                        Tensor(
                            backend.array(d),
                            TensorConfig(
                                backend.array(d).shape, getattr(d, "dtype", dtype), device
                            ),
                        )
                        for d in data
                    )
                return Tensor(
                    backend.array(data), TensorConfig(backend.array(data).shape, dtype, device)
                )
            return func(*args, **kwargs)

        return wrapper

    return decorator


__all__ = [
    "OpDef",
    "dispatch_eager",
    "emit_ir_node",
    "get_op",
    "register_op",
]
