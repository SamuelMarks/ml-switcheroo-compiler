from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.type_promotion import promote_types


def test_promote_complex128_downcast():
    # Without x64 enabled, complex128 promotes to complex64
    assert promote_types(DType.Complex128, DType.Complex64) == DType.Complex64
