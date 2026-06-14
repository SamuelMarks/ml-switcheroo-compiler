"""Tests for type promotion coverage."""


def test_type_promotion_129() -> None:
    """Test type promotion edge cases."""
    import importlib
    import ml_switcheroo_compiler.core.type_promotion as tp

    importlib.reload(tp)
    from ml_switcheroo_compiler.core.dtype import DType

    class FakeType:
        def __eq__(self, other: object) -> bool:
            if type(other) is type(self):
                return False
            return other == DType.Bool

        def __hash__(self) -> int:
            return hash(DType.Bool)

    res = tp.promote_types(FakeType(), FakeType())
    assert isinstance(res, FakeType)
