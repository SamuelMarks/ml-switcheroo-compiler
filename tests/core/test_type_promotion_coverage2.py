"""Tests for type promotion coverage."""

import importlib

import ml_switcheroo_compiler.core.type_promotion as tp
from ml_switcheroo_compiler.core.dtype import DType


def test_type_promotion_129() -> None:
    """Test type promotion edge cases."""
    importlib.reload(tp)

    class FakeType:
        """Class docstring."""

        def __eq__(self, other: object) -> bool:
            """Function docstring."""
            if type(other) is type(self):
                return False
            return other == DType.Bool

        def __hash__(self) -> int:
            """Function docstring."""
            return hash(DType.Bool)

    res = tp.promote_types(FakeType(), FakeType())
    assert isinstance(res, FakeType)
