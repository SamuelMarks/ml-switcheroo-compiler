"""Strongly-typed structural protocols for backend modules and eager tensors."""

from __future__ import annotations

import inspect
import string
import typing

from ml_switcheroo_compiler.backends.eager_registry import (
    EagerCallableProtocol,
    EagerOperand,
    EagerTensorProtocol,
    EagerValue,
    ShapeProtocol,
    ensure_callable,
    ensure_eager_tensor,
    ensure_float,
    ensure_int,
    ensure_shape,
)
from ml_switcheroo_compiler.core.shape import (
    CallableObjectProtocol,
    VariadicCallableProtocol,
)

_ATTR_FORMATTER = string.Formatter()


@typing.runtime_checkable
class _PydanticSchemaHandlerProtocol(typing.Protocol):
    """Protocol for Pydantic core schema handlers."""

    def generate_schema(self, source_type: EagerValue) -> EagerValue:
        """Generate Pydantic core schema for `source_type`.

        Args:
            source_type (EagerValue): Source type annotation.

        Returns:
            EagerValue: Generated core schema.
        """
        ...


@typing.runtime_checkable
class BackendNodeProtocol(typing.Protocol):
    """Structural protocol for backend IR and logical graph nodes."""

    id: str
    op_type: str
    inputs: list[str]
    outputs: list[str]
    attributes: dict[str, EagerValue]
    shape_metadata: tuple[int | str, ...] | list[int | str] | None
    dtype_metadata: str | None
    metadata: dict[str, EagerValue]


@typing.runtime_checkable
class BackendGraphProtocol(typing.Protocol):
    """Structural protocol for backend IR and logical computation graphs."""

    nodes: dict[str, BackendNodeProtocol]
    inputs: list[str]
    outputs: list[str]
    metadata: dict[str, EagerValue]


BackendTensorProtocol = EagerTensorProtocol
_ObjT = typing.TypeVar("_ObjT")
_DefaultT = typing.TypeVar("_DefaultT")


def resolve_backend_member(
    obj: _ObjT,
    attr: str,
    default: _DefaultT | EagerOperand = None,
) -> tuple[bool, EagerTensorProtocol]:
    """Resolve a member on a backend module, class, or instance.

    Args:
        obj (_ObjT): Target object or module.
        attr (str): Attribute name to resolve.
        default (EagerOperand): Default value when attribute is not found.

    Returns:
        tuple[bool, EagerTensorProtocol]: Pair of `(found, resolved_value)`.
    """
    if "." in attr:
        for pair in ensure_eager_tensor(ensure_callable(inspect.getmembers)(obj)):
            if str(pair[0]) == attr:
                return True, ensure_eager_tensor(pair[1])
        return False, ensure_eager_tensor(default)
    try:
        resolved, _ = _ATTR_FORMATTER.get_field(f"0.{attr}", (obj,), {})
        return True, ensure_eager_tensor(resolved)
    except AttributeError:
        return False, ensure_eager_tensor(default)


def has_backend_member(
    obj: _ObjT,
    attr: str,
) -> bool:
    """Return True if `obj` defines member `attr`.

    Args:
        obj (_ObjT): Target object or module.
        attr (str): Attribute name to inspect.

    Returns:
        bool: True if `attr` exists on `obj`.
    """
    return resolve_backend_member(obj, attr)[0]


class _MissingBackendAttrSentinel:
    """Sentinel type for omitted default argument in `get_backend_member`."""


_MISSING_BACKEND_ATTR = _MissingBackendAttrSentinel()


@typing.overload
def get_backend_member(
    obj: _ObjT,
    attr: str,
) -> EagerTensorProtocol: ...


@typing.overload
def get_backend_member(
    obj: _ObjT,
    attr: str,
    default: bool,
) -> bool: ...


@typing.overload
def get_backend_member(
    obj: _ObjT,
    attr: str,
    default: int,
) -> int: ...


@typing.overload
def get_backend_member(
    obj: _ObjT,
    attr: str,
    default: float,
) -> float: ...


@typing.overload
def get_backend_member(
    obj: _ObjT,
    attr: str,
    default: str,
) -> str: ...


@typing.overload
def get_backend_member(
    obj: _ObjT,
    attr: str,
    default: list[EagerValue],
) -> EagerTensorProtocol | list[EagerValue]: ...


@typing.overload
def get_backend_member(
    obj: _ObjT,
    attr: str,
    default: dict[str, EagerValue],
) -> EagerTensorProtocol | dict[str, EagerTensorProtocol]: ...


@typing.overload
def get_backend_member(
    obj: _ObjT,
    attr: str,
    default: tuple[EagerValue, ...],
) -> EagerTensorProtocol | tuple[EagerValue, ...]: ...


@typing.overload
def get_backend_member(
    obj: _ObjT,
    attr: str,
    default: EagerTensorProtocol | BackendModuleProtocol | BackendNamespaceProtocol | EagerCallableProtocol | CallableObjectProtocol | VariadicCallableProtocol | type,
) -> EagerTensorProtocol: ...


@typing.overload
def get_backend_member(
    obj: BackendModuleProtocol | BackendNamespaceProtocol,
    attr: str,
    default: EagerTensorProtocol | None,
) -> EagerTensorProtocol | None: ...


@typing.overload
def get_backend_member(
    obj: _ObjT,
    attr: str,
    default: _ObjT | EagerOperand,
) -> EagerTensorProtocol | EagerValue: ...


def get_backend_member(
    obj: _ObjT,
    attr: str,
    default: _ObjT | EagerOperand | _MissingBackendAttrSentinel = _MISSING_BACKEND_ATTR,
) -> EagerTensorProtocol | EagerValue:
    """Retrieve member `attr` from `obj` or `default`.

    Args:
        obj (EagerValue): Target object or module.
        attr (str): Attribute name to resolve.
        default (EagerOperand | _MissingBackendAttrSentinel): Default value when attribute is not found.

    Returns:
        EagerTensorProtocol | EagerValue: Resolved attribute value.

    Raises:
        AttributeError: If `attr` is not found on `obj` and no `default` was provided.
    """
    found, val = resolve_backend_member(obj, attr, default)
    if not found and isinstance(default, _MissingBackendAttrSentinel):
        raise AttributeError(attr)
    return val


@typing.runtime_checkable
class _BaseNsProto(typing.Protocol):
    """Base protocol for callable sub-namespaces."""

    if typing.TYPE_CHECKING:
        abs: EagerCallableProtocol

        def __call__(self, *args: EagerValue, **kwargs: EagerValue) -> EagerTensorProtocol:
            """Invoke when namespace attribute is itself a callable.

            Args:
                *args (EagerValue): Positional arguments.
                **kwargs (EagerValue): Keyword arguments.

            Returns:
                EagerTensorProtocol: Computed tensor result.
            """
            ...


class BackendNamespaceProtocol(_BaseNsProto):
    """Protocol for sub-namespaces on a backend module (e.g., linalg, fft, random, special)."""


BackendModuleProtocol = EagerTensorProtocol


def _is_backend_mod(val: EagerValue) -> typing.TypeGuard[BackendModuleProtocol]:
    """Type guard narrowing `EagerValue` to `BackendModuleProtocol`.

    Args:
        val (EagerValue): Candidate backend module or namespace.

    Returns:
        bool: True for runtime module objects.
    """
    return not isinstance(val, bytes) or len(val) >= 0


def ensure_mod(val: EagerValue) -> BackendModuleProtocol:
    """Narrow an `EagerValue` module reference to `BackendModuleProtocol` without `typing.cast`.

    Args:
        val (EagerValue): Input backend module or namespace.

    Returns:
        BackendModuleProtocol: Narrowed backend module protocol reference.
    """
    assert _is_backend_mod(val)
    return val


__all__ = [
    "BackendGraphProtocol",
    "BackendModuleProtocol",
    "BackendNamespaceProtocol",
    "BackendNodeProtocol",
    "BackendTensorProtocol",
    "EagerCallableProtocol",
    "EagerOperand",
    "EagerTensorProtocol",
    "EagerValue",
    "ShapeProtocol",
    "ensure_callable",
    "ensure_float",
    "ensure_int",
    "ensure_mod",
    "ensure_shape",
    "ensure_eager_tensor",
    "get_backend_member",
    "has_backend_member",
    "resolve_backend_member",
]
