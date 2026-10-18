# ruff: noqa: E402
"""Define base definitions for the operation registry."""

from __future__ import annotations

import functools
import uuid
from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Optional, TypeVar, Union, cast  # noqa: F401

from ml_switcheroo_ir import AttributeValue, LogicalGraph, LogicalNode

from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.shape import TensorOrScalar
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig

# We import dispatcher directly
from ml_switcheroo_compiler.ops.dispatcher import dispatch_op

# Global operation registry

T = TypeVar("T", bound="OpDef")
OpArgValue = TensorOrScalar


class OpDef:
    """Abstract base class for all operations in the compiler.

    Acts strictly as a definition schema holding operation metadata.
    """

    op_type: str = "Unknown"

    if TYPE_CHECKING:
        __call__: Callable[..., Tensor]
        infer_shape: Callable[..., TensorOrScalar]
        eager_eval: Callable[..., TensorOrScalar]
    else:

        def __call__(self, *args: TensorOrScalar, **kwargs: TensorOrScalar) -> Tensor:
            """Universal dispatcher for the operation.

            Args:
                *args: Positional args.
                **kwargs: Keyword args.

            Returns:
                Tensor: Result.
            """
            return cast(Tensor, dispatch_op(self.op_type, *args, **kwargs))

        def infer_shape(self, *args: TensorOrScalar, **kwargs: TensorOrScalar) -> TensorOrScalar:
            """Compute the infer shape tensor operation.

            Args:
                args: Positional args.
                kwargs: Keyword args.

            Returns:
                tuple[int, ...] | tuple[tuple[int, ...], ...] | str | None: Inferred output shape or tuple of output shapes.
            """
            return tuple()

        def eager_eval(self, *args: TensorOrScalar, **kwargs: TensorOrScalar) -> TensorOrScalar:
            """Compute the eager eval tensor operation.

            Args:
                *args: Positional args.
                **kwargs: Keyword args.

            Returns:
                OpArgValue: Result.
            """
            from ml_switcheroo_compiler.backends.registry import get_active_backend

            backend = get_active_backend()
            return backend.execute_op(self.op_type, *args, **kwargs)


def emit_ir_node(
    graph: LogicalGraph | None,
    op_type: str,
    inputs: list[str],
    shape_metadata: tuple[int | str, ...] | list[int | str] | None = None,
    attributes: Mapping[str, AttributeValue] | Mapping[str, TensorOrScalar] | None = None,
) -> str:
    """Emit a new node into the IR graph directly.

    Args:
        graph (LogicalGraph | None): The graph parameter.
        op_type (str): The op_type parameter.
        inputs (list[str]): The inputs parameter.
        shape_metadata (tuple[int | str, ...] | list[int | str] | None): The shape_metadata parameter.
        attributes (Mapping[str, AttributeValue] | Mapping[str, TensorOrScalar] | None): The attributes parameter.

    Returns:
        str: Emitted node ID.
    """
    nid = f"{op_type.lower()}_{uuid.uuid4().hex[:6]}"
    node = LogicalNode(
        id=nid,
        op_type=op_type,
        inputs=inputs,
        shape_metadata=shape_metadata,
        attributes=cast("Mapping[str, AttributeValue]", attributes or {}),
    )
    if graph is not None:
        graph.nodes[nid] = node
    else:
        from ml_switcheroo_compiler.tracing.state import global_tracing_state

        global_tracing_state.add_node(node)
    return nid


def dispatch_eager(op_name: str) -> Callable[[Callable[..., OpArgValue]], Callable[..., OpArgValue]]:
    """Compute the dispatch eager tensor operation.

    Args:
        op_name (str): The op_name parameter.

    Returns:
        Callable: Result.
    """

    def decorator(func: Callable[..., OpArgValue]) -> Callable[..., OpArgValue]:
        """Compute the decorator tensor operation.

        Args:
            func (Callable): The func parameter.

        Returns:
            Callable: Result.
        """

        @functools.wraps(func)
        def wrapper(*args: OpArgValue, **kwargs: OpArgValue) -> OpArgValue:
            """Compute the wrapper tensor operation.

            Args:
            *args: Positional args.
            **kwargs: Keyword args.

            Returns:
            OpArgValue: Result.
            """
            if config.eager_mode:
                from ml_switcheroo_compiler.backends.registry import get_active_backend

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
                            TensorConfig(getattr(backend.array(d), "shape"), getattr(d, "dtype", dtype), device),
                        )
                        for d in data
                    )
                return Tensor(backend.array(data), TensorConfig(getattr(backend.array(data), "shape"), dtype, device))
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

from ml_switcheroo_compiler.ops.registry import get_op, register_op
