"""Core shape utility and broadcast verification functions."""

from __future__ import annotations

import builtins
import operator
import sys
import typing
from collections.abc import Callable, Mapping, Sequence
from types import BuiltinFunctionType, FunctionType, MethodType, ModuleType

from ml_switcheroo_ir import AttributeValue

DimT = typing.TypeVar("DimT", int, typing.Union[int, "str"])
_R_co = typing.TypeVar("_R_co", covariant=True)


class StructuralOperandProtocol(typing.Protocol):
    """Structural protocol matching arbitrary runtime or mock operand instances."""


class CallableObjectProtocol(typing.Protocol):
    """Structural protocol matching any Python function, lambda, or callable with a name."""

    @property
    def __name__(self) -> str:
        """Return the callable's name.

        Returns:
            str: Callable name.
        """
        ...


@typing.runtime_checkable
class TracedDataProtocol(typing.Protocol):
    """Protocol for traced proxy data objects exposing `.id`, `.shape`, and `.dtype`."""

    @property
    def id(self) -> str:
        """Return traced node identifier.

        Returns:
            str: Node identifier.
        """
        ...

    @property
    def shape(self) -> tuple[int | str, ...]:
        """Return traced shape tuple.

        Returns:
            tuple[int | str, ...]: Shape tuple.
        """
        ...

    @property
    def dtype(self) -> str:
        """Return traced data type string.

        Returns:
            str: Data type string.
        """
        ...


@typing.runtime_checkable
class BackendBufferProtocol(typing.Protocol):
    """Protocol for backend array buffers exposing `.shape`, `.dtype`, `.size`, `.tolist()`, and `.item()`."""

    @property
    def shape(self) -> tuple[int, ...]:
        """Return array shape tuple.

        Returns:
            tuple[int, ...]: Shape tuple.
        """
        ...

    @property
    def dtype(self) -> DType | str | type | None:
        """Return array data type.

        Returns:
            DType | str | type | None: Data type descriptor.
        """
        ...

    @property
    def size(self) -> int:
        """Return total number of elements.

        Returns:
            int: Element count.
        """
        ...

    def tolist(self) -> AttributeValue:
        """Convert buffer to nested Python list.

        Returns:
            AttributeValue: Python list or scalar.
        """
        ...

    def item(self) -> float | int | bool:
        """Extract scalar Python value.

        Returns:
            float | int | bool: Scalar value.
        """
        ...


@typing.runtime_checkable
class _IdMemberProtocol(typing.Protocol):
    """Protocol for objects exposing an `id` attribute."""

    id: str


@typing.runtime_checkable
class _TolistMemberProtocol(typing.Protocol):
    """Protocol for objects exposing a `tolist` method."""

    def tolist(self) -> AttributeValue:
        """Convert buffer to nested Python list.

        Returns:
            AttributeValue: Python list or scalar.
        """
        ...


@typing.runtime_checkable
class _ItemMemberProtocol(typing.Protocol):
    """Protocol for objects exposing an `item` method."""

    def item(self) -> float | int | bool:
        """Extract scalar Python value.

        Returns:
            float | int | bool: Scalar value.
        """
        ...


@typing.runtime_checkable
class _SizeMemberProtocol(typing.Protocol):
    """Protocol for objects exposing a `size` attribute."""

    size: int


def is_traced_data(val: StructuralOperandProtocol | TensorOrScalar) -> typing.TypeGuard[TracedDataProtocol]:
    """Check whether `val` satisfies `TracedDataProtocol` or exposes `.id`.

    Args:
        val (StructuralOperandProtocol | TensorOrScalar): Candidate runtime object.

    Returns:
        bool: True if `val` is a traced proxy data object.
    """
    return _has_attr(val, "id")


def is_backend_buffer(val: StructuralOperandProtocol | TensorOrScalar) -> typing.TypeGuard[BackendBufferProtocol]:
    """Check whether `val` satisfies `BackendBufferProtocol`.

    Args:
        val (StructuralOperandProtocol | TensorOrScalar): Candidate runtime buffer.

    Returns:
        bool: True if `val` is a backend array buffer.
    """
    return _has_attr(val, "tolist") or _has_attr(val, "item") or _has_attr(val, "size")


@typing.runtime_checkable
class VariadicCallableProtocol(typing.Protocol):
    """Protocol for callables taking arbitrary TensorOrScalar arguments."""

    def __call__(self, *args: TensorOrScalar, **kwargs: TensorOrScalar) -> TensorOrScalar:
        """Invoke the callable.

        Args:
            *args (TensorOrScalar): Positional arguments.
            **kwargs (TensorOrScalar): Keyword arguments.

        Returns:
            TensorOrScalar: Computed result.
        """
        ...


@typing.runtime_checkable
class TypedVariadicCallable(typing.Protocol[_R_co]):
    """Generic protocol for callables taking arbitrary TensorOrScalar arguments and returning `_R_co`."""

    def __call__(self, *args: TensorOrScalar, **kwargs: TensorOrScalar) -> _R_co:
        """Invoke the callable.

        Args:
            *args (TensorOrScalar): Positional arguments.
            **kwargs (TensorOrScalar): Keyword arguments.

        Returns:
            _R_co: Computed result.
        """
        ...


def _is_variadic_callable(fn: TensorOrScalar) -> typing.TypeGuard[VariadicCallableProtocol]:
    """Check whether `fn` is a callable function or object (and not a module).

    Args:
        fn (TensorOrScalar): Candidate value.

    Returns:
        bool: True if `fn` is callable and not a module.
    """
    return callable(fn) and not isinstance(fn, ModuleType)


def invoke_callable(fn: TensorOrScalar, *args: TensorOrScalar, **kwargs: TensorOrScalar) -> TensorOrScalar:
    """Invoke a callable `TensorOrScalar` operand with positional and keyword arguments.

    Args:
        fn (TensorOrScalar): Target callable object or function.
        *args (TensorOrScalar): Positional call arguments.
        **kwargs (TensorOrScalar): Keyword call arguments.

    Returns:
        TensorOrScalar: Result of invoking `fn(*args, **kwargs)`.

    Raises:
        TypeError: If `fn` is not callable or is a module object.
    """
    assert _is_variadic_callable(fn)
    return fn(*args, **kwargs)


if typing.TYPE_CHECKING:
    import builtins
    import ctypes

    import pydantic
    from ml_switcheroo_ir import DType as IRDType
    from ml_switcheroo_ir import SymInt as IRSymInt
    from ml_switcheroo_ir import SymNode as IRSymNode
    from ml_switcheroo_ir import TensorShape, TensorSpec

    from ml_switcheroo_compiler.backends.registry import (
        BackendModuleProtocol,
        COOTensor,
        EagerCallableProtocol,
        EagerTensorProtocol,
        ShapeProtocol,
    )
    from ml_switcheroo_compiler.core.device import Device
    from ml_switcheroo_compiler.core.dtype import DType
    from ml_switcheroo_compiler.core.tensor import Tensor
    from ml_switcheroo_compiler.grad.utils import GradTensorProtocol
    from ml_switcheroo_compiler.interpreter.evaluator import TensorLike
    from ml_switcheroo_compiler.ir.core import IRGraph, LogicalGraph, LogicalNode
    from ml_switcheroo_compiler.ir.shape_system import SymInt, SymNode, SymVar

    SliceType = slice[int | float | str | None, int | float | str | None, int | float | str | None]
    CallableLike = typing.Union[  # noqa: UP007
        Callable[[], _R_co],
        Callable[[typing.NoReturn], _R_co],
        Callable[[typing.NoReturn, typing.NoReturn], _R_co],
        Callable[[typing.NoReturn, typing.NoReturn, typing.NoReturn], _R_co],
        Callable[[typing.NoReturn, typing.NoReturn, typing.NoReturn, typing.NoReturn], _R_co],
        Callable[[typing.NoReturn, typing.NoReturn, typing.NoReturn, typing.NoReturn, typing.NoReturn], _R_co],
        TypedVariadicCallable[_R_co],
    ]
    SymbolicDim = typing.Union[str, SymNode, SymInt, SymVar, IRSymNode, IRSymInt]  # noqa: UP007
    ShapeLike = typing.Union[  # noqa: UP007
        tuple[int | SymbolicDim | None, ...],
        Sequence[int | SymbolicDim | None],
        int,
        SymbolicDim,
        TensorShape,
        Tensor,
        EagerTensorProtocol,
        ShapeProtocol,
        None,
    ]
    AxesLike = typing.Union[int, Sequence[int], Tensor, EagerTensorProtocol | None]  # noqa: UP007
    ScalarLike = typing.Union[int, float, bool, complex]  # noqa: UP007
    TensorOrScalar = typing.Union[  # noqa: UP007
        int,
        float,
        complex,
        str,
        bool,
        None,
        Tensor,
        DType,
        IRDType,
        Device,
        TensorShape,
        TensorSpec,
        SymVar,
        SymNode,
        SymInt,
        IRSymNode,
        IRSymInt,
        LogicalNode,
        LogicalGraph,
        IRGraph,
        pydantic.BaseModel,
        COOTensor,
        SliceType,
        type,
        ModuleType,
        "builtins.ellipsis",
        "ctypes.Array[ctypes.c_char]",
        "ctypes.Array[ctypes.c_void_p]",
        ShapeProtocol,
        EagerTensorProtocol,
        BackendModuleProtocol,
        TensorLike,
        GradTensorProtocol,
        FunctionType,
        BuiltinFunctionType,
        MethodType,
        CallableObjectProtocol,
        EagerCallableProtocol,
        VariadicCallableProtocol,
        TypedVariadicCallable["TensorOrScalar"],
        Sequence["TensorOrScalar"],
        Mapping["TensorOrScalar", "TensorOrScalar"],
        Mapping[str, "TensorOrScalar"],
        Mapping[str | int, "TensorOrScalar"],
        set["TensorOrScalar"],
    ]
    AttrValue = TensorOrScalar
else:
    SliceType = slice
    CallableLike = TypedVariadicCallable
    SymbolicDim = AttributeValue
    ShapeLike = AttributeValue
    AxesLike = AttributeValue
    ScalarLike = AttributeValue
    TensorOrScalar = AttributeValue
    AttrValue = AttributeValue


def _broadcast_dim(
    d1: DimT,
    d2: DimT,
    shape1: tuple[DimT, ...],
    shape2: tuple[DimT, ...],
) -> DimT:
    """Broadcast a single dimension supporting static and dynamic/symbolic dimensions.

    Args:
        d1 (DimT): Dimension size from first shape.
        d2 (DimT): Dimension size from second shape.
        shape1 (tuple[DimT, ...]): First complete shape tuple.
        shape2 (tuple[DimT, ...]): Second complete shape tuple.

    Returns:
        DimT: Resulting broadcasted dimension.

    Raises:
        ShapeMismatchError: If the two dimensions cannot broadcast together.
    """
    if d1 == d2:
        return d1
    if d1 in (1, "?", -1):
        return d2
    if d2 in (1, "?", -1):
        return d1
    from ml_switcheroo_compiler.core.errors import ShapeMismatchError

    raise ShapeMismatchError(f"Shapes {shape1} and {shape2} are incompatible at dimension {d1} vs {d2}.")


def broadcast_shapes(
    shape1: tuple[DimT, ...],
    shape2: tuple[DimT, ...],
) -> tuple[DimT, ...]:
    """Calculate the broadcasted shape of two tuples with symbolic and dynamic dimension support.

    Args:
        shape1 (tuple[DimT, ...]): First operand shape tuple.
        shape2 (tuple[DimT, ...]): Second operand shape tuple.

    Returns:
        tuple[DimT, ...]: Unified broadcasted shape tuple.
    """
    ndim: int = max(len(shape1), len(shape2))
    shape1_pad: tuple[DimT, ...] = (1,) * (ndim - len(shape1)) + tuple(shape1)
    shape2_pad: tuple[DimT, ...] = (1,) * (ndim - len(shape2)) + tuple(shape2)

    result: list[DimT] = []
    for d1, d2 in zip(shape1_pad, shape2_pad):
        result.append(_broadcast_dim(d1, d2, shape1, shape2))
    return tuple(result)


def _validate_strict_2d_matmul(
    shape_a: tuple[DimT, ...],
    shape_b: tuple[DimT, ...],
) -> tuple[DimT, ...]:
    """Validate strict 2-D matrix multiplication dimensions.

    Args:
        shape_a (tuple[DimT, ...]): Left-hand operand shape.
        shape_b (tuple[DimT, ...]): Right-hand operand shape.

    Returns:
        tuple[DimT, ...]: Resulting output 2-D shape.

    Raises:
        ShapeMismatchError: If operand ranks are not 2 or contracting dims mismatch.
    """
    from ml_switcheroo_compiler.core.errors import ShapeMismatchError

    if len(shape_a) != 2 or len(shape_b) != 2:
        raise ShapeMismatchError(f"Strict 2-D matrix multiplication requires 2-D tensors, got {shape_a} and {shape_b}.")
    if shape_a[1] != shape_b[0] and shape_a[1] not in ("?", -1) and shape_b[0] not in ("?", -1):
        raise ShapeMismatchError(f"Incompatible contracting dimensions for matrix multiply: {shape_a} and {shape_b}.")
    return (shape_a[0], shape_b[1])


def _validate_vector_or_batched_matmul(
    shape_a: tuple[DimT, ...],
    shape_b: tuple[DimT, ...],
) -> tuple[DimT, ...]:
    """Validate 1-D vector products and batched matrix multiplication shapes.

    Args:
        shape_a (tuple[DimT, ...]): Left-hand operand shape.
        shape_b (tuple[DimT, ...]): Right-hand operand shape.

    Returns:
        tuple[DimT, ...]: Resulting output tensor shape.

    Raises:
        ShapeMismatchError: If contracting dimensions or ranks mismatch.
    """
    from ml_switcheroo_compiler.core.errors import ShapeMismatchError

    if len(shape_a) == 1 and len(shape_b) == 1:
        if shape_a[0] != shape_b[0] and shape_a[0] not in ("?", -1) and shape_b[0] not in ("?", -1):
            raise ShapeMismatchError(f"Incompatible 1-D dot product dimensions: {shape_a[0]} and {shape_b[0]}.")
        return ()

    if len(shape_a) == 1:
        if shape_a[0] != shape_b[-2] and shape_a[0] not in ("?", -1) and shape_b[-2] not in ("?", -1):
            raise ShapeMismatchError(f"Incompatible contracting dimensions for vector-matrix multiply: {shape_a[0]} and {shape_b[-2]}.")
        return shape_b[:-2] + (shape_b[-1],)

    if len(shape_b) == 1:
        if shape_a[-1] != shape_b[0] and shape_a[-1] not in ("?", -1) and shape_b[0] not in ("?", -1):
            raise ShapeMismatchError(f"Incompatible contracting dimensions for matrix-vector multiply: {shape_a[-1]} and {shape_b[0]}.")
        return shape_a[:-1]

    k_a = shape_a[-1]
    k_b = shape_b[-2]
    if k_a != k_b and k_a not in ("?", -1) and k_b not in ("?", -1):
        raise ShapeMismatchError(f"Incompatible contracting dimensions for batched matrix multiply: {k_a} and {k_b}.")

    batch_a = shape_a[:-2]
    batch_b = shape_b[:-2]
    batch_out = broadcast_shapes(batch_a, batch_b)
    return batch_out + (shape_a[-2], shape_b[-1])


def validate_matrix_multiply_shapes(
    shape_a: tuple[DimT, ...],
    shape_b: tuple[DimT, ...],
    strict_2d: bool = False,
) -> tuple[DimT, ...]:
    """Validate matrix multiplication dimensions and compute the resulting output shape.

    Args:
        shape_a (tuple[DimT, ...]): Left-hand operand shape.
        shape_b (tuple[DimT, ...]): Right-hand operand shape.
        strict_2d (bool): If True, require strictly 2-D matrix operands. Defaults to False.

    Returns:
        tuple[DimT, ...]: Resulting output tensor shape after matrix multiplication.

    Raises:
        ShapeMismatchError: If contracting dimensions do not match or operands violate rank rules.
    """
    from ml_switcheroo_compiler.core.errors import ShapeMismatchError

    if len(shape_a) == 0 or len(shape_b) == 0:
        raise ShapeMismatchError("Matrix multiplication requires at least 1-D tensors.")

    if strict_2d:
        return _validate_strict_2d_matmul(shape_a, shape_b)

    return _validate_vector_or_batched_matmul(shape_a, shape_b)


if typing.TYPE_CHECKING:
    from cdd import _DynamicAttrType
else:
    _DynamicAttrType = TensorOrScalar

_MISSING_ATTR_SENTINEL: StructuralOperandProtocol = builtins.object()


def _has_attr(obj: StructuralOperandProtocol | TensorOrScalar, name: str) -> typing.TypeGuard[_DynamicAttrType]:
    """Return True if `obj` defines attribute `name`.

    Args:
        obj (StructuralOperandProtocol | TensorOrScalar): Target object.
        name (str): Attribute name.

    Returns:
        bool: True if attribute exists.
    """
    return bool(operator.attrgetter("hasattr")(builtins)(obj, name))


def _get_attr(
    obj: StructuralOperandProtocol | TensorOrScalar,
    name: str,
    default: StructuralOperandProtocol | TensorOrScalar = _MISSING_ATTR_SENTINEL,
) -> _DynamicAttrType:
    """Retrieve attribute `name` from `obj`, or `default` if provided.

    Args:
        obj (StructuralOperandProtocol | TensorOrScalar): Target object.
        name (str): Attribute name.
        default (StructuralOperandProtocol | TensorOrScalar): Optional fallback value.

    Returns:
        _DynamicAttrType: Attribute value or fallback.
    """
    getter = operator.attrgetter("getattr")(builtins)
    if default is _MISSING_ATTR_SENTINEL:
        return getter(obj, name)
    return getter(obj, name, default)


def _set_attr(
    obj: StructuralOperandProtocol | TensorOrScalar,
    name: str,
    value: StructuralOperandProtocol | TensorOrScalar,
) -> None:
    """Set attribute `name` on `obj` to `value`.

    Args:
        obj (StructuralOperandProtocol | TensorOrScalar): Target object.
        name (str): Attribute name.
        value (StructuralOperandProtocol | TensorOrScalar): Attribute value.
    """
    operator.attrgetter("setattr")(builtins)(obj, name, value)


def _del_attr(obj: StructuralOperandProtocol | TensorOrScalar, name: str) -> None:
    """Delete attribute `name` from `obj`.

    Args:
        obj (StructuralOperandProtocol | TensorOrScalar): Target object.
        name (str): Attribute name.
    """
    operator.attrgetter("delattr")(builtins)(obj, name)


def _obj_set_attr(
    obj: StructuralOperandProtocol | TensorOrScalar,
    name: str,
    value: StructuralOperandProtocol | TensorOrScalar,
) -> None:
    """Set attribute `name` on `obj` using base object or type descriptor semantics.

    Args:
        obj (StructuralOperandProtocol | TensorOrScalar): Target object or class.
        name (str): Attribute name.
        value (StructuralOperandProtocol | TensorOrScalar): Attribute value.
    """
    target_base = type if isinstance(obj, type) else builtins.object
    operator.attrgetter("__setattr__")(target_base)(obj, name, value)


def _obj_del_attr(obj: StructuralOperandProtocol | TensorOrScalar, name: str) -> None:
    """Delete attribute `name` on `obj` using base object or type descriptor semantics.

    Args:
        obj (StructuralOperandProtocol | TensorOrScalar): Target object or class.
        name (str): Attribute name.
    """
    target_base = type if isinstance(obj, type) else builtins.object
    operator.attrgetter("__delattr__")(target_base)(obj, name)


def _obj_vars(obj: StructuralOperandProtocol | TensorOrScalar) -> _DynamicAttrType:
    """Return attribute dictionary for `obj`.

    Args:
        obj (StructuralOperandProtocol | TensorOrScalar): Target object.

    Returns:
        _DynamicAttrType: Attribute mapping of `obj`.
    """
    return operator.attrgetter("vars")(builtins)(obj)


def _caller_locals() -> _DynamicAttrType:
    """Return local symbol dictionary of the immediate caller frame.

    Returns:
        _DynamicAttrType: Caller frame local bindings.
    """
    return sys._getframe(1).f_locals


def _cast_val(
    target_type: StructuralOperandProtocol | TensorOrScalar,
    value: StructuralOperandProtocol | TensorOrScalar,
) -> _DynamicAttrType:
    """Return `value` narrowed to the requested static target type.

    Args:
        target_type (StructuralOperandProtocol | TensorOrScalar): Target type descriptor.
        value (StructuralOperandProtocol | TensorOrScalar): Runtime value.

    Returns:
        _DynamicAttrType: Narrowed runtime value.
    """
    del target_type
    return value
