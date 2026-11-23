"""Runtime protocol registration and safe attribute read helpers for `adapter_bindings`."""

from __future__ import annotations

import abc
import ctypes
import types
import typing
from collections.abc import Callable, Sequence
from unittest.mock import NonCallableMock

_TargetT = typing.TypeVar("_TargetT")
_ValT = typing.TypeVar("_ValT")


def register_dynamic_protocols(protocols: Sequence[abc.ABCMeta]) -> None:
    """Register dynamic `__getattr__` carriers (`ModuleType`, `CDLL`, `NonCallableMock`) on `protocols`.

    In Python 3.12+, `@typing.runtime_checkable` protocols evaluate `inspect.getattr_static` rather than
    invoking `__getattr__`. Registering dynamic carriers as virtual ABC subclasses allows structural narrowing
    to reach direct attribute access, where missing symbols raise `AttributeError` and convert to `(False, None)`.

    Args:
        protocols (Sequence[abc.ABCMeta]): Runtime-checkable `Protocol` classes to register.
    """
    for proto in protocols:
        proto.register(types.ModuleType)
        proto.register(ctypes.CDLL)
        proto.register(NonCallableMock)


def wrap_member_reader(
    reader: Callable[[_TargetT], tuple[bool, _ValT | None]],
) -> Callable[[_TargetT], tuple[bool, _ValT | None]]:
    """Wrap a structural member `reader` so dynamic `AttributeError` returns `(False, None)`.

    Args:
        reader (Callable[[_TargetT], tuple[bool, _ValT | None]]): Structural protocol reader callable.

    Returns:
        Callable[[_TargetT], tuple[bool, _ValT | None]]: Exception-safe structural reader callable.
    """

    def _wrapped(target: _TargetT) -> tuple[bool, _ValT | None]:
        """Read structural member from `target`, returning `(False, None)` on `AttributeError`.

        Args:
            target (_TargetT): Candidate module or operand.

        Returns:
            tuple[bool, _ValT | None]: `(True, value)` when present, else `(False, None)`.
        """
        try:
            return reader(target)
        except AttributeError:
            return False, None

    return _wrapped


def wrap_shard_readers(
    readers: dict[str, Callable[[_TargetT], tuple[bool, _ValT | None]]],
) -> dict[str, Callable[[_TargetT], tuple[bool, _ValT | None]]]:
    """Wrap all structural member readers in a shard dictionary with `AttributeError` handling.

    Args:
        readers (dict[str, Callable[[_TargetT], tuple[bool, _ValT | None]]]): Raw shard reader mapping.

    Returns:
        dict[str, Callable[[_TargetT], tuple[bool, _ValT | None]]]: Wrapped shard reader mapping.
    """
    return {name: wrap_member_reader(fn) for name, fn in readers.items()}
