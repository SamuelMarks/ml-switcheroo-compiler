"""Sound runtime scalar, shape, and sequence coercion helpers without disguised casts."""

from __future__ import annotations

import types
import typing
from collections.abc import Iterable, Sequence

from ml_switcheroo_compiler.core.shape import CallableLike, VariadicCallableProtocol

from .protocols import (
    BackendArrayProtocol,
    EagerBoundCallable,
    EagerCallableProtocol,
    EagerOpCallable,
    EagerTensorProtocol,
)
from .types_base import BackendArg, BackendOperand

if typing.TYPE_CHECKING:
    from ml_switcheroo_compiler.core._shape_protocols import _ItemMemberProtocol

_ItemT_co = typing.TypeVar("_ItemT_co", bound=BackendArg, covariant=True)
_CallRetT = typing.TypeVar("_CallRetT")


@typing.runtime_checkable
class _SupportsIntProto(typing.Protocol):
    """Protocol for scalar/0-D array objects convertible via `int(x)`."""

    def __int__(self) -> int:
        """Convert value to Python `int`.

        Returns:
            int: Converted integer value.
        """
        ...


@typing.runtime_checkable
class _SupportsFloatProto(typing.Protocol):
    """Protocol for scalar/0-D array objects convertible via `float(x)`."""

    def __float__(self) -> float:
        """Convert value to Python `float`.

        Returns:
            float: Converted floating-point value.
        """
        ...


@typing.runtime_checkable
class _SupportsBytesProto(typing.Protocol):
    """Protocol for buffers exposing `.tobytes()`."""

    def tobytes(self) -> bytes:
        """Convert buffer contents to `bytes`.

        Returns:
            bytes: Raw byte buffer.
        """
        ...


@typing.runtime_checkable
class IndexableSequenceProtocol(typing.Protocol[_ItemT_co]):
    """Covariant protocol for finite indexable sequences and backend array containers."""

    def __len__(self) -> int:
        """Return element count along the leading dimension.

        Returns:
            int: Sequence length.
        """
        ...

    def __getitem__(self, index: int) -> _ItemT_co:
        """Return element at `index`.

        Args:
            index (int): Zero-based index.

        Returns:
            _ItemT_co: Indexed element.
        """
        ...


def _can_unwrap_item(val: BackendOperand) -> typing.TypeGuard[_ItemMemberProtocol]:
    """Return `True` when `val` is a non-module object exposing `.item` without auto-vivifying a mock child.

    Args:
        val (BackendOperand): Candidate scalar wrapper.

    Returns:
        typing.TypeGuard[_ItemMemberProtocol]: `True` if `.item()` may be safely invoked.
    """
    from unittest.mock import NonCallableMock

    from ml_switcheroo_compiler.core._shape_protocols import _ItemMemberProtocol

    if type(val) is types.ModuleType:
        return False
    if isinstance(val, NonCallableMock):
        return "item" in dir(val)
    return isinstance(val, _ItemMemberProtocol)


def coerce_int(val: BackendOperand, default: int = 0, *, _unwrap_item: bool = True) -> int:
    """Validate and convert `val` to a genuine Python `int`, returning `default` when `None`.

    Args:
        val (BackendOperand): Candidate integer or `None`.
        default (int): Default integer when `val` is `None`.
        _unwrap_item (bool): Whether to attempt single-step `.item()` unwrapping on scalar wrappers.

    Returns:
        int: Validated Python integer.

    Raises:
        TypeError: If `val` cannot be converted to `int`.
    """
    if val is None:
        return default
    if isinstance(val, bool):
        return int(val)
    if isinstance(val, (int, float, str, bytes)):
        return int(val)
    if _unwrap_item and _can_unwrap_item(val):
        try:
            return coerce_int(val.item(), default=default, _unwrap_item=False)
        except (TypeError, ValueError):
            pass
    cand: BackendOperand | _SupportsIntProto = val
    if isinstance(cand, _SupportsIntProto):
        return int(cand)
    raise TypeError(f"Expected int-convertible value, got {type(val).__name__}")


def coerce_opt_int(val: BackendOperand) -> int | None:
    """Validate and convert `val` to `int | None`.

    Args:
        val (BackendOperand): Optional integer operand.

    Returns:
        int | None: Validated integer or `None`.
    """
    if val is None:
        return None
    return coerce_int(val)


def coerce_float(val: BackendOperand, default: float = 0.0, *, _unwrap_item: bool = True) -> float:
    """Validate and convert `val` to a genuine Python `float`, returning `default` when `None`.

    Args:
        val (BackendOperand): Candidate numeric scalar or `None`.
        default (float): Default float when `val` is `None`.
        _unwrap_item (bool): Whether to attempt single-step `.item()` unwrapping on scalar wrappers.

    Returns:
        float: Validated Python float.

    Raises:
        TypeError: If `val` cannot be converted to `float`.
    """
    if val is None:
        return default
    if isinstance(val, (int, float, str, bytes)):
        return float(val)
    if _unwrap_item and _can_unwrap_item(val):
        try:
            return coerce_float(val.item(), default=default, _unwrap_item=False)
        except (TypeError, ValueError):
            pass
    cand: BackendOperand | _SupportsFloatProto = val
    if isinstance(cand, _SupportsFloatProto):
        return float(cand)
    raise TypeError(f"Expected float-convertible value, got {type(val).__name__}")


def coerce_str(val: BackendOperand, default: str = "") -> str:
    """Convert `val` to `str`, returning `default` when `None`.

    Args:
        val (BackendOperand): Candidate string value or `None`.
        default (str): Default string when `val` is `None`.

    Returns:
        str: Converted string value.
    """
    if val is None:
        return default
    if isinstance(val, bytes):
        return val.decode("utf-8", errors="replace")
    return val if isinstance(val, str) else str(val)


def coerce_opt_str(val: BackendOperand) -> str | None:
    """Convert `val` to `str | None`.

    Args:
        val (BackendOperand): Optional string operand.

    Returns:
        str | None: Converted string or `None`.
    """
    if val is None:
        return None
    return coerce_str(val)


def coerce_bool(val: BackendOperand, default: bool = False) -> bool:
    """Convert `val` to `bool`, returning `default` when `None`.

    Args:
        val (BackendOperand): Candidate boolean operand or `None`.
        default (bool): Default boolean when `val` is `None`.

    Returns:
        bool: Converted boolean value.
    """
    if val is None:
        return default
    return bool(val)


def coerce_bytes(val: BackendOperand) -> bytes:
    """Validate and convert `val` to `bytes`.

    Args:
        val (BackendOperand): Byte string, bytearray, memoryview, UTF-8 string, or `.tobytes()` buffer.

    Returns:
        bytes: Raw byte buffer.

    Raises:
        TypeError: If `val` cannot be converted to `bytes`.
    """
    if isinstance(val, bytes):
        return val
    if isinstance(val, (bytearray, memoryview)):
        return bytes(val)
    if isinstance(val, str):
        return val.encode("utf-8")
    cand: BackendOperand | _SupportsBytesProto = val
    if isinstance(cand, _SupportsBytesProto):
        return cand.tobytes()
    if isinstance(val, (list, tuple)):
        return bytes(coerce_int(b) for b in val)
    raise TypeError(f"Expected bytes-convertible value, got {type(val).__name__}")


def coerce_shape(val: BackendOperand) -> tuple[int, ...]:
    """Validate and convert `val` into a shape tuple `tuple[int, ...]`.

    Args:
        val (BackendOperand): Dimension integer, sequence of dimensions, or `None`.

    Returns:
        tuple[int, ...]: Validated integer shape tuple.

    Raises:
        TypeError: If `val` is not an integer or iterable of integer dimensions.
    """
    if val is None:
        return ()
    if isinstance(val, int) and not isinstance(val, bool):
        return (val,)
    if isinstance(val, (tuple, list, Sequence)) and not isinstance(val, (str, bytes)):
        return tuple(coerce_int(d) for d in val)
    cand_iter: BackendOperand | Iterable[BackendOperand] = val
    if isinstance(cand_iter, Iterable) and not isinstance(cand_iter, (str, bytes)):
        return tuple(coerce_int(d) for d in cand_iter)
    raise TypeError(f"Expected integer or shape sequence, got {type(val).__name__}")


def coerce_int_list(val: BackendOperand) -> list[int]:
    """Validate and convert `val` into a `list[int]`.

    Args:
        val (BackendOperand): Sequence or iterable of integer-convertible values.

    Returns:
        list[int]: Validated list of integers.
    """
    return list(coerce_shape(val))


def coerce_float_list(val: BackendOperand) -> list[float]:
    """Validate and convert `val` into a `list[float]`.

    Args:
        val (BackendOperand): Sequence or iterable of float-convertible values.

    Returns:
        list[float]: Validated list of floats.

    Raises:
        TypeError: If `val` is not an iterable of numeric values.
    """
    if val is None:
        return []
    if isinstance(val, (int, float)) and not isinstance(val, bool):
        return [float(val)]
    cand_iter: BackendOperand | Iterable[BackendOperand] = val
    if isinstance(cand_iter, Iterable) and not isinstance(cand_iter, (str, bytes)):
        return [coerce_float(x) for x in cand_iter]
    raise TypeError(f"Expected iterable of floats, got {type(val).__name__}")


def coerce_opt_int_seq(val: BackendOperand) -> Sequence[int] | None:
    """Validate and convert `val` into `tuple[int, ...] | None`.

    Args:
        val (BackendOperand): Optional sequence of integers.

    Returns:
        Sequence[int] | None: Validated integer tuple or `None`.
    """
    if val is None:
        return None
    return coerce_shape(val)


def coerce_int_or_tuple(val: BackendOperand) -> int | tuple[int, ...]:
    """Validate and convert `val` into either a single `int` or `tuple[int, ...]`.

    Args:
        val (BackendOperand): Single integer or sequence of integers.

    Returns:
        int | tuple[int, ...]: Validated integer or integer tuple.
    """
    if isinstance(val, int) and not isinstance(val, bool):
        return val
    return coerce_shape(val)


@typing.overload
def coerce_operand_list(val: list[_ItemT_co]) -> list[_ItemT_co]: ...


@typing.overload
def coerce_operand_list(val: tuple[_ItemT_co, ...]) -> list[_ItemT_co]: ...


@typing.overload
def coerce_operand_list(val: BackendArg) -> list[BackendArg]: ...


def coerce_operand_list(
    val: list[_ItemT_co] | tuple[_ItemT_co, ...] | BackendArg,
) -> list[_ItemT_co] | list[BackendArg]:
    """Materialize an iterable operand into a Python `list`.

    Args:
        val (list[_ItemT_co] | tuple[_ItemT_co, ...] | BackendArg): Iterable sequence or array container.

    Returns:
        list[_ItemT_co] | list[BackendArg]: Materialized Python list.

    Raises:
        TypeError: If `val` is not iterable.
    """
    if isinstance(val, (list, tuple)):
        return list(val)
    cand: BackendArg | Iterable[BackendArg] = val
    if isinstance(cand, Iterable):
        return list(cand)
    raise TypeError(f"Expected iterable operand, got {type(val).__name__}")


@typing.overload
def unpack_pair(val: tuple[_ItemT_co, ...]) -> tuple[_ItemT_co, _ItemT_co]: ...


@typing.overload
def unpack_pair(val: list[_ItemT_co]) -> tuple[_ItemT_co, _ItemT_co]: ...


@typing.overload
def unpack_pair(val: EagerTensorProtocol) -> tuple[EagerTensorProtocol, EagerTensorProtocol]: ...


@typing.overload
def unpack_pair(val: BackendArg) -> tuple[BackendArg, BackendArg]: ...


def unpack_pair(
    val: EagerTensorProtocol | tuple[_ItemT_co, ...] | list[_ItemT_co] | BackendArg,
) -> tuple[EagerTensorProtocol, EagerTensorProtocol] | tuple[_ItemT_co, _ItemT_co] | tuple[BackendArg, BackendArg]:
    """Validate length == 2 and unpack `val` into a 2-tuple.

    Args:
        val (EagerTensorProtocol | tuple[_ItemT_co, ...] | list[_ItemT_co] | BackendArg): 2-element sequence.

    Returns:
        tuple[EagerTensorProtocol, EagerTensorProtocol]
        | tuple[_ItemT_co, _ItemT_co]
        | tuple[BackendArg, BackendArg]: Unpacked 2-tuple `(first, second)`.

    Raises:
        TypeError: If `val` is not an indexable sequence.
        ValueError: If `len(val) != 2`.
    """
    cand: BackendArg | IndexableSequenceProtocol[BackendArg] = val
    if not isinstance(cand, IndexableSequenceProtocol):
        raise TypeError(f"Expected 2-element indexable sequence, got {type(val).__name__}")
    if len(cand) != 2:
        raise ValueError(f"Expected 2 elements, got {len(cand)}")
    return (cand[0], cand[1])


@typing.overload
def unpack_triple(val: tuple[_ItemT_co, ...]) -> tuple[_ItemT_co, _ItemT_co, _ItemT_co]: ...


@typing.overload
def unpack_triple(val: list[_ItemT_co]) -> tuple[_ItemT_co, _ItemT_co, _ItemT_co]: ...


@typing.overload
def unpack_triple(val: EagerTensorProtocol) -> tuple[EagerTensorProtocol, EagerTensorProtocol, EagerTensorProtocol]: ...


@typing.overload
def unpack_triple(val: BackendArg) -> tuple[BackendArg, BackendArg, BackendArg]: ...


def unpack_triple(
    val: EagerTensorProtocol | tuple[_ItemT_co, ...] | list[_ItemT_co] | BackendArg,
) -> (
    tuple[EagerTensorProtocol, EagerTensorProtocol, EagerTensorProtocol]
    | tuple[_ItemT_co, _ItemT_co, _ItemT_co]
    | tuple[BackendArg, BackendArg, BackendArg]
):
    """Validate length == 3 and unpack `val` into a 3-tuple.

    Args:
        val (EagerTensorProtocol | tuple[_ItemT_co, ...] | list[_ItemT_co] | BackendArg): 3-element sequence.

    Returns:
        tuple[EagerTensorProtocol, EagerTensorProtocol, EagerTensorProtocol]
        | tuple[_ItemT_co, _ItemT_co, _ItemT_co]
        | tuple[BackendArg, BackendArg, BackendArg]: Unpacked 3-tuple.

    Raises:
        TypeError: If `val` is not an indexable sequence.
        ValueError: If `len(val) != 3`.
    """
    cand: BackendArg | IndexableSequenceProtocol[BackendArg] = val
    if not isinstance(cand, IndexableSequenceProtocol):
        raise TypeError(f"Expected 3-element indexable sequence, got {type(val).__name__}")
    if len(cand) != 3:
        raise ValueError(f"Expected 3 elements, got {len(cand)}")
    return (cand[0], cand[1], cand[2])


@typing.overload
def unpack_quad(val: tuple[_ItemT_co, ...]) -> tuple[_ItemT_co, _ItemT_co, _ItemT_co, _ItemT_co]: ...


@typing.overload
def unpack_quad(val: list[_ItemT_co]) -> tuple[_ItemT_co, _ItemT_co, _ItemT_co, _ItemT_co]: ...


@typing.overload
def unpack_quad(
    val: EagerTensorProtocol,
) -> tuple[EagerTensorProtocol, EagerTensorProtocol, EagerTensorProtocol, EagerTensorProtocol]: ...


@typing.overload
def unpack_quad(val: BackendArg) -> tuple[BackendArg, BackendArg, BackendArg, BackendArg]: ...


def unpack_quad(
    val: EagerTensorProtocol | tuple[_ItemT_co, ...] | list[_ItemT_co] | BackendArg,
) -> (
    tuple[EagerTensorProtocol, EagerTensorProtocol, EagerTensorProtocol, EagerTensorProtocol]
    | tuple[_ItemT_co, _ItemT_co, _ItemT_co, _ItemT_co]
    | tuple[BackendArg, BackendArg, BackendArg, BackendArg]
):
    """Validate length == 4 and unpack `val` into a 4-tuple.

    Args:
        val (EagerTensorProtocol | tuple[_ItemT_co, ...] | list[_ItemT_co] | BackendArg): 4-element sequence.

    Returns:
        tuple[EagerTensorProtocol, EagerTensorProtocol, EagerTensorProtocol, EagerTensorProtocol]
        | tuple[_ItemT_co, _ItemT_co, _ItemT_co, _ItemT_co]
        | tuple[BackendArg, BackendArg, BackendArg, BackendArg]: Unpacked 4-tuple.

    Raises:
        TypeError: If `val` is not an indexable sequence.
        ValueError: If `len(val) != 4`.
    """
    cand: BackendArg | IndexableSequenceProtocol[BackendArg] = val
    if not isinstance(cand, IndexableSequenceProtocol):
        raise TypeError(f"Expected 4-element indexable sequence, got {type(val).__name__}")
    if len(cand) != 4:
        raise ValueError(f"Expected 4 elements, got {len(cand)}")
    return (cand[0], cand[1], cand[2], cand[3])


def coerce_opt_callable(
    val: BackendOperand,
) -> VariadicCallableProtocol | EagerOpCallable | EagerCallableProtocol | None:
    """Validate that `val` is `None` or a runtime callable object.

    Args:
        val (BackendOperand): Optional callable operand.

    Returns:
        VariadicCallableProtocol | EagerOpCallable | EagerCallableProtocol | None: Validated callable or `None`.

    Raises:
        TypeError: If `val` is neither `None` nor callable.
    """
    if val is None:
        return None
    cand: BackendOperand | VariadicCallableProtocol = val
    if isinstance(cand, VariadicCallableProtocol):
        return cand
    raise TypeError(f"Expected callable or None, got {type(val).__name__}")


_RequireCallableCandidate = typing.Union[  # noqa: UP007
    types.ModuleType,
    BackendOperand,
    CallableLike[BackendOperand],
    CallableLike[_CallRetT],
    EagerBoundCallable,
    EagerCallableProtocol,
]


def require_callable(val: _RequireCallableCandidate[_CallRetT]) -> EagerBoundCallable:
    """Validate at runtime that `val` is callable and return it as `EagerBoundCallable`.

    Args:
        val (_RequireCallableCandidate[_CallRetT]): Candidate callable object.

    Returns:
        EagerBoundCallable: Validated callable object.

    Raises:
        TypeError: If `val` is not callable at runtime.
    """
    cand: _RequireCallableCandidate[_CallRetT] = val
    if isinstance(cand, EagerBoundCallable):
        return cand
    raise TypeError(f"Expected callable object, got {type(val).__name__}")


def _is_backend_array(val: BackendOperand) -> typing.TypeGuard[BackendArrayProtocol]:
    """Check whether `val` satisfies `BackendArrayProtocol` including mock arrays.

    Args:
        val (BackendOperand): Candidate value.

    Returns:
        typing.TypeGuard[BackendArrayProtocol]: `True` if `val` is a backend array.
    """
    from ml_switcheroo_compiler.core._mock_protocol import is_mock_with_attrs

    found: bool = isinstance(val, BackendArrayProtocol)
    return found or is_mock_with_attrs(val, "shape")


def coerce_array(
    mod: types.ModuleType | BackendOperand,
    val: BackendArg,
) -> BackendArrayProtocol:
    """Return `val` if already a `BackendArrayProtocol`, else convert via `mod`'s `asarray` / `array`.

    Args:
        mod (types.ModuleType | BackendOperand): Backend module or adapter.
        val (BackendArg): Candidate tensor, sequence, or scalar value.

    Returns:
        BackendArrayProtocol: Validated backend array instance.
    """
    if _is_backend_array(val):
        return val
    from ml_switcheroo_compiler.backends.protocols import adapt_backend_module

    return adapt_backend_module(mod).asarray(val)


def coerce_numeric(val: BackendOperand) -> BackendArrayProtocol | float | int:
    """Validate and narrow `val` to `BackendArrayProtocol | float | int`.

    Args:
        val (BackendOperand): Candidate array or numeric scalar operand.

    Returns:
        BackendArrayProtocol | float | int: Validated array or real scalar.
    """
    if isinstance(val, bool):
        return int(val)
    if isinstance(val, (int, float)):
        return val
    if _is_backend_array(val):
        return val
    return coerce_float(val)


def coerce_array_list(
    mod: types.ModuleType | BackendOperand,
    val: BackendArg,
) -> list[BackendArrayProtocol]:
    """Coerce `val` into a list of `BackendArrayProtocol` elements.

    Args:
        mod (types.ModuleType | BackendOperand): Backend module or adapter.
        val (BackendArg): Iterable of arrays/scalars or single array.

    Returns:
        list[BackendArrayProtocol]: List of validated backend arrays.
    """
    return [coerce_array(mod, item) for item in coerce_operand_list(val)]


__all__ = [
    "IndexableSequenceProtocol",
    "coerce_array",
    "coerce_array_list",
    "coerce_bool",
    "coerce_bytes",
    "coerce_float",
    "coerce_float_list",
    "coerce_int",
    "coerce_int_list",
    "coerce_int_or_tuple",
    "coerce_numeric",
    "coerce_operand_list",
    "coerce_opt_callable",
    "coerce_opt_int",
    "coerce_opt_int_seq",
    "coerce_opt_str",
    "coerce_shape",
    "coerce_str",
    "require_callable",
    "unpack_pair",
    "unpack_quad",
    "unpack_triple",
]
