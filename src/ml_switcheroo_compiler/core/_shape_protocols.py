"""Foundational runtime structural protocols for Shape Layer 2."""

from __future__ import annotations

import typing
from collections.abc import Sequence

from ml_switcheroo_ir import AttributeValue

from ml_switcheroo_compiler.core.device import Device
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.ir.shape_system import SymInt, SymNode

if typing.TYPE_CHECKING:
    from ml_switcheroo_compiler.core._shape_types import OpValue

ShapeDim = typing.Union[int, str, SymNode, SymInt, None]  # noqa: UP007
ShapeTuple = tuple[ShapeDim, ...]

DimT = typing.TypeVar(
    "DimT",
    int,
    typing.Union[int, "str"],
    typing.Union[int, "str", SymNode],
    ShapeDim,
)
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
    def shape(self) -> ShapeTuple:
        """Return traced shape tuple.

        Returns:
            ShapeTuple: Shape tuple.
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
    def shape(self) -> Sequence[ShapeDim]:
        """Get the shape of the array.

        Returns:
            Sequence[ShapeDim]: Array shape sequence.
        """
        ...


@typing.runtime_checkable
class _IdMemberProtocol(typing.Protocol):
    """Protocol for objects exposing an `id` attribute."""

    @property
    def id(self) -> str:
        """Return identifier string.

        Returns:
            str: Identifier string.
        """
        ...


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

    @property
    def size(self) -> int:
        """Return total element count.

        Returns:
            int: Element count.
        """
        ...


@typing.runtime_checkable
class ShapeMetadataProtocol(typing.Protocol):
    """Protocol for IR nodes or descriptors exposing `shape_metadata`."""

    @property
    def shape_metadata(self) -> Sequence[ShapeDim] | None:
        """Return symbolic or static shape metadata sequence.

        Returns:
            Sequence[ShapeDim] | None: Shape metadata dimensions.
        """
        ...


@typing.runtime_checkable
class DTypeHolderProtocol(typing.Protocol):
    """Protocol for tensor/array objects exposing a `dtype` attribute."""

    @property
    def dtype(self) -> DType | str | OpValue:
        """Return tensor or array data type descriptor.

        Returns:
            DType | str | OpValue: Data type descriptor.
        """
        ...


@typing.runtime_checkable
class DeviceHolderProtocol(typing.Protocol):
    """Protocol for tensor/array objects exposing a `device` attribute."""

    @property
    def device(self) -> Device | str | None:
        """Return tensor placement device.

        Returns:
            Device | str | None: Placement device.
        """
        ...


@typing.runtime_checkable
class DataHolderProtocol(typing.Protocol):
    """Protocol for wrapper tensors exposing a `.data` attribute."""

    @property
    def data(self) -> OpValue:
        """Return underlying tensor payload.

        Returns:
            OpValue: Underlying data value.
        """
        ...


@typing.runtime_checkable
class VariadicCallableProtocol(typing.Protocol):
    """Protocol for callables taking arbitrary OpValue arguments."""

    def __call__(self, *args: OpValue, **kwargs: OpValue) -> OpValue:
        """Invoke the callable.

        Args:
            *args (OpValue): Positional arguments.
            **kwargs (OpValue): Keyword arguments.

        Returns:
            OpValue: Computed result.
        """
        ...


@typing.runtime_checkable
class TypedVariadicCallable(typing.Protocol[_R_co]):
    """Generic protocol for callables taking arbitrary OpValue arguments and returning `_R_co`."""

    def __call__(self, *args: OpValue, **kwargs: OpValue) -> _R_co:
        """Invoke the callable.

        Args:
            *args (OpValue): Positional arguments.
            **kwargs (OpValue): Keyword arguments.

        Returns:
            _R_co: Computed result.
        """
        ...


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
