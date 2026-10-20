"""Tests for test_creation."""

from __future__ import annotations

import typing

from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.ops.creation.frontend_basic import _infer_dtype

if typing.TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import EagerCallableProtocol


def test_infer_dtype_from_array_object() -> None:
    """Test function test_infer_dtype_from_array_object."""

    class ObjArr:
        """Test class ObjArr."""

        dtype = "O"

    assert typing.cast("EagerCallableProtocol", _infer_dtype)(typing.cast("EagerCallableProtocol", ObjArr)()) == DType.Object
