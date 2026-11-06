# ruff: noqa: E402
"""Define base definitions for the operation registry."""

from __future__ import annotations

import functools
import typing  # noqa: F401
import uuid
from collections.abc import Callable, Mapping, Sequence

from ml_switcheroo_ir import AttributeValue, LogicalGraph, LogicalNode

from ml_switcheroo_compiler.backends.registry import ensure_callable, ensure_eager_tensor
from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.shape import (
    DataHolderProtocol as DataHolderProtocol,
)
from ml_switcheroo_compiler.core.shape import (
    DeviceHolderProtocol as DeviceHolderProtocol,
)
from ml_switcheroo_compiler.core.shape import (
    DTypeHolderProtocol as DTypeHolderProtocol,
)
from ml_switcheroo_compiler.core.shape import (
    LinearOperatorOperandProtocol as LinearOperatorOperandProtocol,
)
from ml_switcheroo_compiler.core.shape import (
    RaggedOperandProtocol as RaggedOperandProtocol,
)
from ml_switcheroo_compiler.core.shape import (
    ShapeMetadataProtocol as ShapeMetadataProtocol,
)
from ml_switcheroo_compiler.core.shape import (
    ShapeProtocol as ShapeProtocol,
)
from ml_switcheroo_compiler.core.shape import (
    SparseOperandProtocol as SparseOperandProtocol,
)
from ml_switcheroo_compiler.core.shape import (
    TensorArrayOperandProtocol as TensorArrayOperandProtocol,
)
from ml_switcheroo_compiler.core.shape import (
    TensorOperandProtocol as TensorOperandProtocol,
)
from ml_switcheroo_compiler.core.shape import (
    TensorOrScalar,
    TracedDataProtocol,
    TypedVariadicCallable,
    extract_data,
    extract_device,
    extract_dtype,
    extract_shape,
)
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.ir.core import ensure_attr_val

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

    def __call__(self, *args: TensorOrScalar, **kwargs: TensorOrScalar) -> Tensor:
        """Universal dispatcher for the operation.

        Args:
            *args (TensorOrScalar): Positional args.
            **kwargs (TensorOrScalar): Keyword args.

        Returns:
            Tensor: Result.
        """
        return dispatch_op(self.op_type, *args, **kwargs)

    def infer_shape(self, *args: TensorOrScalar, **kwargs: TensorOrScalar) -> TensorOrScalar:
        """Compute the infer shape tensor operation.

        Args:
            *args (TensorOrScalar): Positional args.
            **kwargs (TensorOrScalar): Keyword args.

        Returns:
            TensorOrScalar: Inferred output shape or tuple of output shapes.
        """
        return tuple()

    def eager_eval(self, *args: TensorOrScalar, **kwargs: TensorOrScalar) -> TensorOrScalar:
        """Compute the eager eval tensor operation.

        Args:
            *args (TensorOrScalar): Positional args.
            **kwargs (TensorOrScalar): Keyword args.

        Returns:
            OpArgValue: Result.
        """
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        backend = get_active_backend()
        return backend.execute_op(self.op_type, *args, **kwargs)


@typing.runtime_checkable
class _DimsShapeProtocol(typing.Protocol):
    """Protocol for shape wrappers exposing a `dims` sequence."""

    dims: Sequence[int | str | float | None]


def _is_typed_variadic_op(fn: _F) -> typing.TypeGuard[TypedVariadicCallable[OpArgValue]]:
    """Type guard narrowing `fn` to `TypedVariadicCallable[OpArgValue]`.

    Args:
        fn (_F): Candidate callable.

    Returns:
        bool: True if callable.
    """
    return callable(fn)


def _as_f(fn: TypedVariadicCallable[OpArgValue], orig: _F) -> _F:
    """Narrow wrapper callable to `_F`.

    Args:
        fn (TypedVariadicCallable[OpArgValue]): Wrapped callable.
        orig (_F): Original callable type reference.

    Returns:
        _F: Narrowed callable.
    """
    del orig
    raw_fn: TypedVariadicCallable[OpArgValue] = fn

    def _guard(cand: TypedVariadicCallable[OpArgValue]) -> typing.TypeGuard[_F]:
        """Narrow candidate callable to `_F`.

        Args:
            cand (TypedVariadicCallable[OpArgValue]): Candidate callable.

        Returns:
            bool: True if callable.
        """
        return callable(cand)

    assert _guard(raw_fn)
    return raw_fn


def emit_ir_node(
    graph: LogicalGraph | None,
    op_type: str,
    inputs: list[str],
    shape_metadata: tuple[int | str, ...] | list[int | str] | _DimsShapeProtocol | TensorOrScalar = None,
    attributes: Mapping[str, AttributeValue] | Mapping[str, TensorOrScalar] | None = None,
) -> str:
    """Emit a new node into the IR graph directly.

    Args:
        graph (LogicalGraph | None): The graph parameter.
        op_type (str): The op_type parameter.
        inputs (list[str]): The inputs parameter.
        shape_metadata (tuple[int | str, ...] | list[int | str] | _DimsShapeProtocol | TensorOrScalar): The shape_metadata parameter.
        attributes (Mapping[str, AttributeValue] | Mapping[str, TensorOrScalar] | None): The attributes parameter.

    Returns:
        str: Emitted node ID.
    """
    nid = f"{op_type.lower()}_{uuid.uuid4().hex[:6]}"
    from ml_switcheroo_ir import SymInt, SymNode

    norm_shape: tuple[int | str | SymNode | SymInt, ...] | None = None
    raw_sm: _DimsShapeProtocol | TensorOrScalar = shape_metadata
    raw_shape: Sequence[int | str | float | None] | _DimsShapeProtocol | TensorOrScalar = raw_sm.dims if isinstance(raw_sm, _DimsShapeProtocol) else shape_metadata
    if isinstance(raw_shape, (tuple, list)):
        first_elem: TensorOrScalar = raw_shape[0] if len(raw_shape) > 0 else None
        flat_shape = first_elem if isinstance(first_elem, (tuple, list)) else raw_shape
        norm_shape = tuple(int(d) if isinstance(d, float) else (d if isinstance(d, (int, str, SymNode, SymInt)) and not isinstance(d, bool) else ("?" if d is None else str(d))) for d in flat_shape)
    raw_inputs: Sequence[TensorOrScalar] = inputs
    norm_inputs: list[str] = [i if isinstance(i, str) else str(i.id if isinstance(i, TracedDataProtocol) else id(i)) for i in raw_inputs]
    norm_attrs: dict[str, AttributeValue] = {str(k): ensure_attr_val(v) for k, v in (attributes or {}).items()}
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
        cand_func: _F = func
        assert _is_typed_variadic_op(cand_func)
        callable_func = cand_func

        @functools.wraps(ensure_callable(callable_func))
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
                dtype = first_tensor.dtype if first_tensor is not None else "float32"
                if isinstance(data, tuple):
                    return tuple(
                        Tensor(
                            backend.array(d),
                            TensorConfig(extract_shape(backend.array(d)), extract_dtype(d, dtype), device),
                        )
                        for d in data
                    )
                return Tensor(backend.array(data), TensorConfig(extract_shape(backend.array(data)), dtype if first_tensor is not None else None, device))
            return ensure_eager_tensor(ensure_callable(callable_func)(*args, **kwargs))

        return _as_f(wrapper, func)

    return decorator


__all__ = [
    "DataHolderProtocol",
    "DTypeHolderProtocol",
    "DeviceHolderProtocol",
    "LinearOperatorOperandProtocol",
    "OpDef",
    "RaggedOperandProtocol",
    "ShapeMetadataProtocol",
    "ShapeProtocol",
    "SparseOperandProtocol",
    "TensorArrayOperandProtocol",
    "TensorOperandProtocol",
    "dispatch_eager",
    "emit_ir_node",
    "extract_data",
    "extract_device",
    "extract_dtype",
    "extract_shape",
    "get_op",
    "register_op",
]

from ml_switcheroo_compiler.ops.registry import get_op, register_op
