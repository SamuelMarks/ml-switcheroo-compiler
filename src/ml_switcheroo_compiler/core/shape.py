"""Core shape utility and broadcast verification functions."""

from __future__ import annotations

import builtins
import string
import typing
from collections.abc import Callable, Mapping, Sequence
from types import BuiltinFunctionType, FunctionType, MethodType, ModuleType

import pydantic
from ml_switcheroo_ir import (
    AttributeValue,
    LogicalGraph,
    LogicalNode,
    TensorShape,
    TensorSpec,
)
from ml_switcheroo_ir import DType as IRDType
from ml_switcheroo_ir import LogicalGraph as IRGraph
from ml_switcheroo_ir import SymInt as IRSymInt
from ml_switcheroo_ir import SymNode as IRSymNode

from ml_switcheroo_compiler.core.dtype import DType

_AttributeValuePydanticAnnotation = pydantic.PlainValidator(lambda val: val)

DimT = typing.TypeVar("DimT", int, typing.Union[int, "str"])
_R_co = typing.TypeVar("_R_co", covariant=True)


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
class ShapeProtocol(typing.Protocol):
    """Protocol for any array-like object exposing a shape tuple."""

    @property
    def shape(self) -> Sequence[int | str]:
        """Get the shape of the array.

        Returns:
            Sequence[Union[int, str]]: Array shape sequence.
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


@typing.runtime_checkable
class ShapeMetadataProtocol(typing.Protocol):
    """Protocol for IR nodes or descriptors exposing `shape_metadata`."""

    shape_metadata: Sequence[int | str | None] | None


@typing.runtime_checkable
class DTypeHolderProtocol(typing.Protocol):
    """Protocol for tensor/array objects exposing a `dtype` attribute."""

    dtype: DType | str | TensorOrScalar


@typing.runtime_checkable
class DeviceHolderProtocol(typing.Protocol):
    """Protocol for tensor/array objects exposing a `device` attribute."""

    device: Device | str | None


@typing.runtime_checkable
class DataHolderProtocol(typing.Protocol):
    """Protocol for wrapper tensors exposing a `.data` attribute."""

    data: TensorOrScalar


@typing.runtime_checkable
class TensorOperandProtocol(typing.Protocol):
    """Shared structural protocol for tensor/array operator operands."""

    shape: tuple[int, ...]
    dtype: DType | str
    ndim: int
    size: int
    data: TensorOrScalar

    def to_numpy(self) -> EagerTensorProtocol:
        """Convert operand to a NumPy array.

        Returns:
            EagerTensorProtocol: NumPy array representation.
        """
        ...

    def numpy(self) -> EagerTensorProtocol:
        """Return NumPy array representation.

        Returns:
            EagerTensorProtocol: NumPy array representation.
        """
        ...


@typing.runtime_checkable
class RaggedOperandProtocol(typing.Protocol):
    """Shared structural protocol for ragged tensor operator operands."""

    values: TensorOrScalar
    row_splits: TensorOrScalar
    nested_row_splits: Sequence[TensorOrScalar]
    shape: tuple[int | None, ...]
    dtype: DType | str


@typing.runtime_checkable
class SparseOperandProtocol(typing.Protocol):
    """Shared structural protocol for sparse tensor operator operands."""

    indices: TensorOrScalar
    values: TensorOrScalar
    dense_shape: tuple[int, ...]
    shape: tuple[int, ...]
    dtype: DType | str

    def tocoo(self) -> SparseOperandProtocol:
        """Convert sparse operand to COO format.

        Returns:
            SparseOperandProtocol: COO sparse representation.
        """
        ...

    def toarray(self) -> EagerTensorProtocol:
        """Convert sparse operand to dense array.

        Returns:
            EagerTensorProtocol: Dense array representation.
        """
        ...

    def todense(self) -> EagerTensorProtocol:
        """Convert sparse operand to dense tensor.

        Returns:
            EagerTensorProtocol: Dense tensor representation.
        """
        ...


@typing.runtime_checkable
class TensorArrayOperandProtocol(typing.Protocol):
    """Shared structural protocol for dynamic TensorArray operator operands."""

    def size(self) -> int | TensorOrScalar:
        """Return number of elements in the TensorArray.

        Returns:
            int | TensorOrScalar: TensorArray size.
        """
        ...

    def read(self, index: int | TensorOrScalar) -> TensorOrScalar:
        """Read element at `index`.

        Args:
            index (int | TensorOrScalar): Target slot index.

        Returns:
            TensorOrScalar: Element tensor.
        """
        ...

    def write(self, index: int | TensorOrScalar, value: TensorOrScalar) -> TensorArrayOperandProtocol:
        """Write `value` at `index`.

        Args:
            index (int | TensorOrScalar): Target slot index.
            value (TensorOrScalar): Element tensor to write.

        Returns:
            TensorArrayOperandProtocol: Updated TensorArray.
        """
        ...

    def stack(self) -> TensorOrScalar:
        """Stack all elements in the TensorArray along axis 0.

        Returns:
            TensorOrScalar: Stacked tensor.
        """
        ...

    def unstack(self, value: TensorOrScalar) -> TensorArrayOperandProtocol:
        """Unstack `value` along axis 0 into the TensorArray.

        Args:
            value (TensorOrScalar): Tensor to unstack.

        Returns:
            TensorArrayOperandProtocol: Populated TensorArray.
        """
        ...

    def gather(self, indices: TensorOrScalar) -> TensorOrScalar:
        """Gather elements at `indices`.

        Args:
            indices (TensorOrScalar): Slot indices to gather.

        Returns:
            TensorOrScalar: Gathered tensor.
        """
        ...

    def scatter(self, indices: TensorOrScalar, value: TensorOrScalar) -> TensorArrayOperandProtocol:
        """Scatter `value` into `indices`.

        Args:
            indices (TensorOrScalar): Target slot indices.
            value (TensorOrScalar): Values to scatter.

        Returns:
            TensorArrayOperandProtocol: Updated TensorArray.
        """
        ...


@typing.runtime_checkable
class LinearOperatorOperandProtocol(typing.Protocol):
    """Shared structural protocol for structured LinearOperator operands."""

    shape: tuple[int, ...]
    dtype: DType | str
    H: TensorOrScalar
    T: TensorOrScalar
    A: TensorOrScalar

    def matmul(self, x: TensorOrScalar, adjoint: bool = False, adjoint_arg: bool = False) -> TensorOrScalar:
        """Multiply operator by matrix `x`.

        Args:
            x (TensorOrScalar): Right-hand matrix operand.
            adjoint (bool): Take adjoint of this operator.
            adjoint_arg (bool): Take adjoint of `x`.

        Returns:
            TensorOrScalar: Matrix product result.
        """
        ...

    def matvec(self, x: TensorOrScalar, adjoint: bool = False) -> TensorOrScalar:
        """Multiply operator by vector `x`.

        Args:
            x (TensorOrScalar): Right-hand vector operand.
            adjoint (bool): Take adjoint of this operator.

        Returns:
            TensorOrScalar: Matrix-vector product result.
        """
        ...

    def solve(self, rhs: TensorOrScalar, adjoint: bool = False, adjoint_arg: bool = False) -> TensorOrScalar:
        """Solve linear system `Op @ X = rhs`.

        Args:
            rhs (TensorOrScalar): Right-hand side matrix.
            adjoint (bool): Solve with adjoint of this operator.
            adjoint_arg (bool): Take adjoint of `rhs`.

        Returns:
            TensorOrScalar: Solution tensor.
        """
        ...

    def solvevec(self, rhs: TensorOrScalar, adjoint: bool = False) -> TensorOrScalar:
        """Solve linear system `Op @ x = rhs` for vector `rhs`.

        Args:
            rhs (TensorOrScalar): Right-hand side vector.
            adjoint (bool): Solve with adjoint of this operator.

        Returns:
            TensorOrScalar: Solution vector.
        """
        ...

    def to_dense(self) -> TensorOrScalar:
        """Materialize dense matrix representation.

        Returns:
            TensorOrScalar: Dense matrix tensor.
        """
        ...

    def adjoint(self) -> LinearOperatorOperandProtocol:
        """Return conjugate transpose LinearOperator.

        Returns:
            LinearOperatorOperandProtocol: Adjoint operator.
        """
        ...

    def diag_part(self) -> TensorOrScalar:
        """Extract main diagonal part.

        Returns:
            TensorOrScalar: Diagonal vector tensor.
        """
        ...

    def trace(self) -> TensorOrScalar:
        """Compute matrix trace.

        Returns:
            TensorOrScalar: Trace scalar tensor.
        """
        ...

    def det(self) -> TensorOrScalar:
        """Compute matrix determinant.

        Returns:
            TensorOrScalar: Determinant scalar tensor.
        """
        ...

    def log_abs_determinant(self) -> TensorOrScalar:
        """Compute log absolute determinant.

        Returns:
            TensorOrScalar: Log abs determinant tensor.
        """
        ...

    def eigvals(self) -> TensorOrScalar:
        """Compute eigenvalues of operator.

        Returns:
            TensorOrScalar: Eigenvalues tensor.
        """
        ...

    def cond(self) -> TensorOrScalar:
        """Compute condition number of operator.

        Returns:
            TensorOrScalar: Condition number tensor.
        """
        ...

    def cholesky(self) -> LinearOperatorOperandProtocol:
        """Compute Cholesky factor LinearOperator.

        Returns:
            LinearOperatorOperandProtocol: Cholesky factor operator.
        """
        ...

    def inverse(self) -> LinearOperatorOperandProtocol:
        """Compute inverse LinearOperator.

        Returns:
            LinearOperatorOperandProtocol: Inverse operator.
        """
        ...


def _is_int_shape_tuple(val: tuple[int | str | None, ...]) -> typing.TypeGuard[tuple[int, ...]]:
    """Type guard narrowing a shape tuple to `tuple[int, ...]`.

    Args:
        val (tuple[int | str | None, ...]): Shape tuple.

    Returns:
        bool: Always True for shape tuples.
    """
    return isinstance(val, tuple)


_SHAPE_ATTR_FORMATTER = string.Formatter()
_AttrObjT = typing.TypeVar("_AttrObjT")


def has_backend_member(obj: _AttrObjT, attr: str) -> bool:
    """Check whether `obj` exposes member `attr`.

    Args:
        obj (_AttrObjT): Candidate object.
        attr (str): Attribute name.

    Returns:
        bool: True if `obj` has `attr`.
    """
    try:
        _SHAPE_ATTR_FORMATTER.get_field(f"0.{attr}", (obj,), {})
        return True
    except Exception:
        return False


def _has_shape_attr(val: TensorOrScalar) -> typing.TypeGuard[ShapeProtocol]:
    """Check whether `val` exposes a `shape` attribute.

    Args:
        val (TensorOrScalar): Candidate object.

    Returns:
        bool: True if `val` has `shape`.
    """
    return has_backend_member(val, "shape")


def _has_shape_metadata_attr(val: TensorOrScalar) -> typing.TypeGuard[ShapeMetadataProtocol]:
    """Check whether `val` exposes a `shape_metadata` attribute.

    Args:
        val (TensorOrScalar): Candidate object.

    Returns:
        bool: True if `val` has `shape_metadata`.
    """
    return has_backend_member(val, "shape_metadata")


def _has_dtype_holder_attr(val: TensorOrScalar) -> typing.TypeGuard[DTypeHolderProtocol]:
    """Check whether `val` exposes a `dtype` attribute.

    Args:
        val (TensorOrScalar): Candidate object.

    Returns:
        bool: True if `val` has `dtype`.
    """
    return has_backend_member(val, "dtype")


def _has_device_holder_attr(val: TensorOrScalar) -> typing.TypeGuard[DeviceHolderProtocol]:
    """Check whether `val` exposes a `device` attribute.

    Args:
        val (TensorOrScalar): Candidate object.

    Returns:
        bool: True if `val` has `device`.
    """
    return has_backend_member(val, "device")


def _has_data_holder_attr(val: TensorOrScalar) -> typing.TypeGuard[DataHolderProtocol]:
    """Check whether `val` exposes a `data` attribute.

    Args:
        val (TensorOrScalar): Candidate object.

    Returns:
        bool: True if `val` has `data`.
    """
    return has_backend_member(val, "data")


def extract_shape(
    val: TensorOrScalar,
    default: tuple[int, ...] = (),
) -> tuple[int, ...]:
    """Safely extract `.shape` or `.shape_metadata` from `val`, falling back to `default`.

    Args:
        val (TensorOrScalar): Tensor, IRNode, or shape-bearing object.
        default (tuple[int, ...]): Default shape tuple if `val` has no shape.

    Returns:
        tuple[int, ...]: Extracted shape tuple.
    """
    if _has_shape_attr(val) and val.shape is not None:
        res_shape = tuple(val.shape)
        assert _is_int_shape_tuple(res_shape)
        return res_shape
    if _has_shape_metadata_attr(val) and val.shape_metadata is not None:
        res_meta = tuple(val.shape_metadata)
        assert _is_int_shape_tuple(res_meta)
        return res_meta
    return default


def _is_dtype_or_str(val: TensorOrScalar) -> typing.TypeGuard[DType | str]:
    """Type guard narrowing `val` to `DType | str`.

    Args:
        val (TensorOrScalar): Candidate dtype value.

    Returns:
        bool: True if non-None.
    """
    return val is not None


def extract_dtype(
    val: TensorOrScalar,
    default: DType | str = "float32",
) -> DType | str:
    """Safely extract `.dtype` from `val`, falling back to `default`.

    Args:
        val (TensorOrScalar): Tensor or array object.
        default (DType | str): Fallback dtype if `val` has no `.dtype`.

    Returns:
        DType | str: Extracted dtype or fallback.
    """
    if _has_dtype_holder_attr(val) and val.dtype is not None:
        raw_dt: TensorOrScalar = val.dtype
        assert _is_dtype_or_str(raw_dt)
        return raw_dt
    return default


def extract_device(
    val: TensorOrScalar,
    default: Device | str | None = None,
) -> Device | str | None:
    """Safely extract `.device` from `val`, falling back to `default`.

    Args:
        val (TensorOrScalar): Tensor or array object.
        default (Device | str | None): Fallback device if `val` has no `.device`.

    Returns:
        Device | str | None: Extracted device or fallback.
    """
    if _has_device_holder_attr(val) and val.device is not None:
        dev: Device | str = val.device
        return dev
    return default


def extract_data(val: TensorOrScalar) -> TensorOrScalar:
    """Safely unwrap `.data` from `val` if present, otherwise return `val`.

    Args:
        val (TensorOrScalar): Tensor or raw value.

    Returns:
        TensorOrScalar: Unwrapped `.data` or `val`.
    """
    if _has_data_holder_attr(val):
        raw_d: TensorOrScalar = val.data
        return raw_d
    return val


def is_traced_data(val: TensorOrScalar) -> typing.TypeGuard[TracedDataProtocol]:
    """Check whether `val` satisfies `TracedDataProtocol` or exposes `.id`.

    Args:
        val (TensorOrScalar): Candidate runtime object.

    Returns:
        bool: True if `val` is a traced proxy data object.
    """
    return has_backend_member(val, "id")


def is_backend_buffer(val: TensorOrScalar) -> typing.TypeGuard[BackendBufferProtocol]:
    """Check whether `val` satisfies `BackendBufferProtocol`.

    Args:
        val (TensorOrScalar): Candidate runtime buffer.

    Returns:
        bool: True if `val` is a backend array buffer.
    """
    return has_backend_member(val, "tolist") or has_backend_member(val, "item") or has_backend_member(val, "size")


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


def invoke_callable(
    fn: TensorOrScalar,
    *args: TensorOrScalar,
    **kwargs: TensorOrScalar,
) -> TensorOrScalar:
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


@typing.runtime_checkable
class _RuntimePlaceholderProtocol(typing.Protocol):
    """Runtime-checkable placeholder protocol for forward-referenced compiler types."""

    @property
    def __doc__(self) -> str | None:
        """Return docstring attribute.

        Returns:
            str | None: Docstring.
        """
        ...


class _ShapeModuleProtocol(typing.Protocol):
    """Protocol for registering runtime placeholder attributes on `shape` module."""

    BackendModuleProtocol: type
    COOTensor: type
    EagerCallableProtocol: type
    EagerTensorProtocol: type
    GradTensorProtocol: type
    Tensor: type
    TensorLike: type
    SymInt: type
    SymNode: type
    SymVar: type
    Device: type
    TensorOrScalar: type
    _EllipsisType: type
    _TensorOrScalarSelf: type


def _is_shape_mod(mod: ModuleType) -> typing.TypeGuard[_ShapeModuleProtocol]:
    """Type guard narrowing `ModuleType` to `_ShapeModuleProtocol`.

    Args:
        mod (ModuleType): Module instance.

    Returns:
        bool: True if `mod` is a `ModuleType`.
    """
    return isinstance(mod, ModuleType)


def _setup_shape_runtime_protocols() -> None:
    """Register runtime-checkable protocol placeholders for forward-referenced types prior to `TensorOrScalar` evaluation."""
    import sys

    curr_mod = sys.modules[__name__]
    assert _is_shape_mod(curr_mod)
    curr_mod.BackendModuleProtocol = _RuntimePlaceholderProtocol
    curr_mod.COOTensor = _RuntimePlaceholderProtocol
    curr_mod.EagerCallableProtocol = _RuntimePlaceholderProtocol
    curr_mod.EagerTensorProtocol = _RuntimePlaceholderProtocol
    curr_mod.GradTensorProtocol = _RuntimePlaceholderProtocol
    curr_mod.Tensor = _RuntimePlaceholderProtocol
    curr_mod.TensorLike = _RuntimePlaceholderProtocol
    curr_mod.SymInt = _RuntimePlaceholderProtocol
    curr_mod.SymNode = _RuntimePlaceholderProtocol
    curr_mod.SymVar = _RuntimePlaceholderProtocol
    curr_mod.Device = _RuntimePlaceholderProtocol
    curr_mod.TensorOrScalar = _RuntimePlaceholderProtocol
    curr_mod._EllipsisType = type(Ellipsis)
    curr_mod._TensorOrScalarSelf = _RuntimePlaceholderProtocol


_setup_shape_runtime_protocols()

if typing.TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.registry import (
        BackendModuleProtocol,
        COOTensor,
        EagerCallableProtocol,
        EagerTensorProtocol,
    )
    from ml_switcheroo_compiler.core.device import Device
    from ml_switcheroo_compiler.core.dtype import DType
    from ml_switcheroo_compiler.core.tensor import Tensor
    from ml_switcheroo_compiler.grad.utils import GradTensorProtocol
    from ml_switcheroo_compiler.interpreter.evaluator import TensorLike
    from ml_switcheroo_compiler.ir.core import IRGraph, LogicalGraph, LogicalNode
    from ml_switcheroo_compiler.ir.shape_system import SymInt, SymNode, SymVar

    _EllipsisType: typing.TypeAlias = builtins.ellipsis
    _TensorOrScalarSelf: typing.TypeAlias = "TensorOrScalar"


@typing.runtime_checkable
class SliceProtocol(typing.Protocol):
    """Structural protocol matching Python `slice` objects."""

    @property
    def start(self) -> int | float | str | None:
        """Return slice start bound.

        Returns:
            int | float | str | None: Slice start.
        """
        ...

    @property
    def stop(self) -> int | float | str | None:
        """Return slice stop bound.

        Returns:
            int | float | str | None: Slice stop.
        """
        ...

    @property
    def step(self) -> int | float | str | None:
        """Return slice step bound.

        Returns:
            int | float | str | None: Slice step.
        """
        ...


@typing.runtime_checkable
class CtypesArrayProtocol(typing.Protocol):
    """Structural protocol matching `ctypes.Array` instances."""

    _length_: int


SliceType = SliceProtocol
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
ShapeLike = typing.Annotated[
    typing.Union[  # noqa: UP007
        tuple[typing.Union[int, SymbolicDim, None], ...],  # noqa: UP007
        Sequence[typing.Union[int, SymbolicDim, None]],  # noqa: UP007
        int,
        SymbolicDim,
        TensorShape,
        Tensor,
        EagerTensorProtocol,
        ShapeProtocol,
        None,
    ],
    _AttributeValuePydanticAnnotation,
]
AxesLike = typing.Annotated[
    typing.Union[int, Sequence[int], Tensor, EagerTensorProtocol, None],  # noqa: UP007
    _AttributeValuePydanticAnnotation,
]
ScalarLike = typing.Union[int, float, bool, complex]  # noqa: UP007
TensorOrScalar = typing.Annotated[
    typing.Union[  # noqa: UP007
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
        _EllipsisType,
        CtypesArrayProtocol,
        ShapeProtocol,
        ShapeMetadataProtocol,
        DTypeHolderProtocol,
        DeviceHolderProtocol,
        DataHolderProtocol,
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
        TypedVariadicCallable[_TensorOrScalarSelf],
        Sequence[_TensorOrScalarSelf],
        Mapping[_TensorOrScalarSelf, _TensorOrScalarSelf],
        Mapping[str, _TensorOrScalarSelf],
        Mapping[typing.Union[str, int], _TensorOrScalarSelf],  # noqa: UP007
        set[_TensorOrScalarSelf],
    ],
    _AttributeValuePydanticAnnotation,
]
AttrValue = TensorOrScalar


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
