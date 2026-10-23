"""Core shape utility and broadcast verification functions."""

from __future__ import annotations

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
