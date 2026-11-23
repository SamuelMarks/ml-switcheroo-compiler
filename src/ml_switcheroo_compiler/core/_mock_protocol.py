"""Python 3.12+ structural protocol helper for `unittest.mock.NonCallableMock` instances."""

from __future__ import annotations

import typing
from collections.abc import Sequence
from unittest.mock import NonCallableMock

_ValT = typing.TypeVar("_ValT")


def is_mock_with_attrs(val: _ValT, *attrs: str) -> bool:
    """Return `True` when `val` is a `NonCallableMock` exposing all `attrs`.

    In Python 3.12+, `@typing.runtime_checkable` `Protocol` `isinstance()` checks rely on
    `inspect.getattr_static`, which does not inspect `NonCallableMock._mock_children` or
    `NonCallableMock._mock_methods`.

    Args:
        val (_ValT): Candidate runtime value.
        *attrs (str): Required protocol attribute names.

    Returns:
        bool: `True` if `val` is a mock supporting every attribute in `attrs`.
    """
    if not isinstance(val, NonCallableMock):
        return False
    available = set(dir(val))
    if any(attr in val._mock_children and attr not in available for attr in attrs):
        return False
    if all(attr in available for attr in attrs):
        return True
    mock_methods: Sequence[str] | None = val._mock_methods
    return mock_methods is None
