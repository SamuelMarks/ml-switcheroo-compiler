"""Tests for test_creation."""

from __future__ import annotations

import typing

from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.ops.creation.frontend_basic import _infer_dtype
from tests.testing_utils import as_callable

if typing.TYPE_CHECKING:
    pass


def test_infer_dtype_from_array_object() -> None:
    """Test function test_infer_dtype_from_array_object."""

    class ObjArr:
        """Test class ObjArr."""

        dtype = "O"

    assert as_callable(_infer_dtype)(as_callable(ObjArr)()) == DType.Object
