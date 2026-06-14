"""Module docstring."""

from ml_switcheroo_compiler.ops.binary.special import Divmod, Isclose, Atan2


def test_divmod_infer_shape_fallback() -> None:
    """Docstring."""
    op = Divmod()
    assert op.infer_shape(None, (1, 2)) is None


def test_isclose_infer_shape_fallback() -> None:
    """Docstring."""
    op = Isclose()
    assert op.infer_shape(None, (1, 2)) is None


def test_atan2_infer_shape_fallback() -> None:
    """Docstring."""
    op = Atan2()
    assert op.infer_shape(None, (1, 2)) is None
