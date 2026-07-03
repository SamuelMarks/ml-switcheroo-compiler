"""Module docstring."""

from ml_switcheroo_compiler.core.config import ConfigContext
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.type_promotion import promote_types


def test_type_promotion_complex128_downgrade() -> object:
    """Function docstring."""
    with ConfigContext(jax_enable_x64=False):
        res = promote_types(DType.Complex128, DType.Complex128)
        assert res == DType.Complex64
