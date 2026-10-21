# ruff: noqa: E402
"""Define base definitions for the operation registry."""

from __future__ import annotations

import functools
import typing  # noqa: F401
import uuid
from collections.abc import Callable, Mapping, Sequence

from ml_switcheroo_ir import AttributeValue, LogicalGraph, LogicalNode

from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.shape import TensorOrScalar, TypedVariadicCallable
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig

# We import dispatcher directly
from ml_switcheroo_compiler.ops.dispatcher import dispatch_op

# Global operation registry

T = typing.TypeVar("T", bound="OpDef")
_F = typing.TypeVar("_F")
OpArgValue = TensorOrScalar


class OpDef:
    """Abstract base class for all operations in the compiler.

    Acts strictly as a definition schema holding operation metadata.
    """

    op_type: str = "Unknown"

    if typing.TYPE_CHECKING:

        def __call__(self, *args: TensorOrScalar, **kwargs: TensorOrScalar) -> Tensor:
            """Universal dispatcher for the operation."""
            ...

        def __getattr__(self, name: str) -> TypedVariadicCallable[TensorOrScalar]:
            """Resolve infer_shape, eager_eval, and operation-specific methods."""
            ...

    else:

        def __call__(self, *args: TensorOrScalar, **kwargs: TensorOrScalar) -> Tensor:
            """Universal dispatcher for the operation.

            Args:
                *args: Positional args.
                **kwargs: Keyword args.

            Returns:
                Tensor: Result.
            """
            return dispatch_op(self.op_type, *args, **kwargs)

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
    from ml_switcheroo_ir import SymInt, SymNode

    from ml_switcheroo_compiler.tracing.builder import _to_attr_value

    norm_shape: tuple[int | str | SymNode | SymInt, ...] | None = None
    raw_shape: TensorOrScalar = getattr(shape_metadata, "dims", shape_metadata)
    if isinstance(raw_shape, (tuple, list)):
        first_elem: TensorOrScalar = raw_shape[0] if len(raw_shape) > 0 else None
        flat_shape = first_elem if isinstance(first_elem, (tuple, list)) else raw_shape
        norm_shape = tuple(int(d) if isinstance(d, float) else (d if isinstance(d, (int, str, SymNode, SymInt)) and not isinstance(d, bool) else ("?" if d is None else str(d))) for d in flat_shape)
    raw_inputs: Sequence[TensorOrScalar] = inputs
    norm_inputs: list[str] = [i if isinstance(i, str) else str(getattr(i, "id", id(i))) for i in raw_inputs]
    norm_attrs: dict[str, AttributeValue] = {str(k): _to_attr_value(v) for k, v in (attributes or {}).items()}
    node = LogicalNode(
        id=nid,
        op_type=op_type,
        inputs=norm_inputs,
        shape_metadata=norm_shape,
        attributes=norm_attrs,
    )
    if graph is not None:
        graph.nodes[nid] = node
    else:
        from ml_switcheroo_compiler.tracing.state import global_tracing_state

        global_tracing_state.add_node(node)
    return nid


def dispatch_eager(op_name: str) -> Callable[[_F], _F]:
    """Compute the dispatch eager tensor operation.

    Args:
        op_name (str): The op_name parameter.

    Returns:
        Callable: Result.
    """

    def decorator(func: _F) -> _F:
        """Compute the decorator tensor operation.

        Args:
            func (Callable): The func parameter.

        Returns:
            Callable: Result.
        """
        callable_func = typing.cast("TypedVariadicCallable[OpArgValue]", func)

        @functools.wraps(callable_func)
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
            return callable_func(*args, **kwargs)

        return typing.cast("_F", wrapper)

    return decorator


__all__ = [
    "OpDef",
    "dispatch_eager",
    "emit_ir_node",
    "get_op",
    "register_op",
]

from ml_switcheroo_compiler.ops.registry import get_op, register_op
