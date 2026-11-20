# ruff: noqa: E402
"""Define base definitions for the operation registry."""

from __future__ import annotations

import typing  # noqa: F401
import uuid
from collections.abc import Callable, Mapping, Sequence

from ml_switcheroo_ir import AttributeValue, LogicalGraph, LogicalNode

from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.shape import (
    CallableLike,
    OpValue,
    TracedDataProtocol,
    extract_data,
    extract_device,
    extract_dtype,
    extract_shape,
    is_variadic_callable,
)
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
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.ir.core import ensure_attr_val

# We import dispatcher directly
from ml_switcheroo_compiler.ops.dispatcher import dispatch_op

if typing.TYPE_CHECKING:
    from ml_switcheroo_compiler.ops.shape_inference import ShapeType

# Global operation registry

T = typing.TypeVar("T", bound="OpDef")
_F = typing.TypeVar("_F", bound=OpValue)
OpArgValue = OpValue
OpInferResult: typing.TypeAlias = "OpValue | ShapeType"


def _default_op_infer_shape(*args: OpValue, **kwargs: OpValue) -> tuple[int, ...]:
    """Return an empty tuple as the default shape inference fallback for `OpSchemaBase`.

    Args:
        *args (OpValue): Positional args.
        **kwargs (OpValue): Keyword args.

    Returns:
        tuple[int, ...]: Empty shape tuple.
    """
    del args, kwargs
    return ()


class OpSchemaBase:
    """Base schema class holding operation metadata, shape inference, and eager evaluation."""

    op_type: str = "Unknown"
    infer_shape: CallableLike[OpInferResult] = staticmethod(_default_op_infer_shape)

    def dispatch_infer_shape(self, *args: OpValue, **kwargs: OpValue) -> OpValue:
        """Dynamically invoke this operation's `infer_shape` method.

        Args:
            *args (OpValue): Positional args.
            **kwargs (OpValue): Keyword args.

        Returns:
            OpValue: Inferred output shape or tuple of output shapes.
        """
        _callable_fn_1: OpValue = self.infer_shape
        assert is_variadic_callable(_callable_fn_1)
        return _callable_fn_1(*args, **kwargs)

    def eager_eval(self, *args: OpValue, **kwargs: OpValue) -> OpValue:
        """Compute the eager evaluation of the operation on the active backend.

        Args:
            *args (OpValue): Positional args.
            **kwargs (OpValue): Keyword args.

        Returns:
            OpArgValue: Computed `OpArgValue` output.
        """
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        backend = get_active_backend()
        return backend.execute_op(self.op_type, *args, **kwargs)


class OpDef(OpSchemaBase):
    """Abstract base class for single-output tensor operations in the compiler.

    Acts strictly as a definition schema holding operation metadata.
    """

    def __call__(self, *args: OpValue, **kwargs: OpValue) -> Tensor:
        """Universal dispatcher for the operation.

        Args:
            *args (OpValue): Positional args.
            **kwargs (OpValue): Keyword args.

        Returns:
            Tensor: Computed `Tensor` output.
        """
        return dispatch_op(self.op_type, *args, **kwargs)


class MultiOutputOpDef(OpSchemaBase):
    """Base class for operations returning tuples or composite structures of tensors."""


@typing.runtime_checkable
class _DimsShapeProtocol(typing.Protocol):
    """Protocol for shape wrappers exposing a `dims` sequence."""

    dims: Sequence[int | str | float | None]


@typing.runtime_checkable
class _NamedCallableProtocol(typing.Protocol):
    """Protocol for callables exposing ``__name__`` and ``__doc__``."""

    __name__: str
    __doc__: str | None


class EagerDispatchedCallable(typing.Generic[_F]):
    """Typed callable wrapper routing to the active backend in eager mode or original fn when tracing."""

    def __init__(self, op_name: str, func: _F) -> None:
        """Initialize eager-dispatched callable wrapper.

        Args:
            op_name (str): Canonical PascalCase operation name.
            func (_F): Underlying symbolic/tracing implementation function.
        """
        self._op_name: str = op_name
        self._func: _F = func
        self.__wrapped__: _F = func
        self.__name__: str = func.__name__ if isinstance(func, _NamedCallableProtocol) else op_name
        self.__doc__: str | None = func.__doc__ if isinstance(func, _NamedCallableProtocol) else None

    def __call__(self, *args: OpArgValue, **kwargs: OpArgValue) -> OpArgValue:
        """Execute operation via active backend in eager mode or invoke underlying symbolic function.

        Args:
            *args (OpArgValue): Positional operands.
            **kwargs (OpArgValue): Keyword operation attributes.

        Returns:
            OpArgValue: Evaluated tensor(s) or symbolic graph result.
        """
        if config.eager_mode:
            from ml_switcheroo_compiler.backends.registry import get_active_backend

            backend = get_active_backend()
            raw_args = [a.data if isinstance(a, Tensor) else a for a in args]
            data = backend.execute_op(self._op_name, *raw_args, **kwargs)
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
            return Tensor(
                backend.array(data),
                TensorConfig(extract_shape(backend.array(data)), dtype if first_tensor is not None else None, device),
            )
        _callable_fn_2: OpValue = self._func
        assert is_variadic_callable(_callable_fn_2)
        return _callable_fn_2(*args, **kwargs)


def emit_ir_node(
    graph: LogicalGraph | None,
    op_type: str,
    inputs: list[str],
    shape_metadata: tuple[int | str, ...] | list[int | str] | _DimsShapeProtocol | OpValue = None,
    attributes: Mapping[str, AttributeValue] | Mapping[str, OpValue] | None = None,
) -> str:
    """Emit a new node into the IR graph directly.

    Args:
        graph (LogicalGraph | None): Input `graph` argument (graph).
        op_type (str): Input `op_type` argument (op type).
        inputs (list[str]): Input `inputs` argument (inputs).
        shape_metadata (tuple[int | str, ...] | list[int | str] | _DimsShapeProtocol | OpValue): Input
            `shape_metadata` argument (shape metadata).
        attributes (Mapping[str, AttributeValue] | Mapping[str, OpValue] | None): Input `attributes` argument
            (attributes).

    Returns:
        str: Emitted node ID.
    """
    nid = f"{op_type.lower()}_{uuid.uuid4().hex[:6]}"
    from ml_switcheroo_compiler.ir import SymInt, SymNode

    norm_shape: tuple[int | str | SymNode | SymInt, ...] | None = None
    raw_sm: _DimsShapeProtocol | OpValue = shape_metadata
    raw_shape: Sequence[int | str | float | None] | _DimsShapeProtocol | OpValue = (
        raw_sm.dims if isinstance(raw_sm, _DimsShapeProtocol) else shape_metadata
    )
    if isinstance(raw_shape, (tuple, list)):
        first_elem: OpValue = raw_shape[0] if len(raw_shape) > 0 else None
        flat_shape = first_elem if isinstance(first_elem, (tuple, list)) else raw_shape
        norm_shape = tuple(
            int(d)
            if isinstance(d, float)
            else (
                d
                if isinstance(d, (int, str, SymNode, SymInt)) and not isinstance(d, bool)
                else ("?" if d is None else str(d))
            )
            for d in flat_shape
        )
    raw_inputs: Sequence[OpValue] = inputs
    norm_inputs: list[str] = [
        i if isinstance(i, str) else str(i.id if isinstance(i, TracedDataProtocol) else id(i)) for i in raw_inputs
    ]
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


def dispatch_eager(op_name: str) -> Callable[[_F], EagerDispatchedCallable[_F]]:
    """Decorate an operation function to route through the active backend in eager mode.

    Args:
        op_name (str): Canonical PascalCase operation name.

    Returns:
        Callable[[_F], EagerDispatchedCallable[_F]]: Decorator returning an ``EagerDispatchedCallable``.
    """

    def decorator(func: _F) -> EagerDispatchedCallable[_F]:
        """Wrap ``func`` in an ``EagerDispatchedCallable``.

        Args:
            func (_F): Underlying operation function.

        Returns:
            EagerDispatchedCallable[_F]: Wrapped operation callable.
        """
        return EagerDispatchedCallable(op_name, func)

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
