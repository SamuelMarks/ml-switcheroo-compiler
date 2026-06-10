import pytest
from ml_switcheroo.core.dtype import DType
from ml_switcheroo.core.errors import DTypePromotionError
from ml_switcheroo.core.type_promotion import promote_types


def test_promote_types():
    assert promote_types(DType.Float32, DType.Float32) == DType.Float32
    assert promote_types(DType.Int32, DType.Float32) == DType.Float32
    assert promote_types(DType.Float16, DType.Float32) == DType.Float32
    assert promote_types(DType.BFloat16, DType.Float32) == DType.Float32
    assert promote_types(DType.Float16, DType.BFloat16) == DType.Float32
    assert promote_types(DType.Int16, DType.Int32) == DType.Int32
    assert promote_types(DType.Float32, DType.Complex64) == DType.Complex64
    assert promote_types(DType.Float64, DType.Complex64) == DType.Complex128
    assert promote_types(DType.Bool, DType.Int32) == DType.Int32

    with pytest.raises(DTypePromotionError):
        promote_types("unknown", DType.Float32)
