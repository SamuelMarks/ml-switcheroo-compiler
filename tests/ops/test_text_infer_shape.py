"""Test infer_shape for text ops."""

from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.ops.text.ops import (
    AsString,
    EditDistance,
    Hashing,
    IntegerLookup,
    Lookup,
    RegexFullMatch,
    RegexReplace,
    StringJoin,
    StringLength,
    StringLookup,
    StringLower,
    StringSplit,
    StringSubstr,
    StringToHash,
    StringToNumber,
    StringUpper,
    TextVectorization,
)


def test_text_infer_shape() -> None:
    """Test text ops infer_shape."""
    t = Tensor("dummy", TensorConfig((2, 2), DType.String, None))

    assert StringToHash().infer_shape() == ()
    assert RegexReplace().infer_shape() == ()
    assert StringSplit().infer_shape() == ()
    assert Lookup().infer_shape(t) is t
    assert Hashing().infer_shape(t) is t
    assert StringLookup().infer_shape(t) is t
    assert IntegerLookup().infer_shape(t) is t
    assert TextVectorization().infer_shape(t) is t
    assert StringToNumber().infer_shape(t) == (2, 2)
    assert StringLower().infer_shape(t) == (2, 2)
    assert StringUpper().infer_shape(t) == (2, 2)
    assert StringJoin().infer_shape() == ()
    assert StringLength().infer_shape(t) == (2, 2)
    assert StringSubstr().infer_shape(t) == (2, 2)
    assert RegexFullMatch().infer_shape(t) == (2, 2)
    assert EditDistance().infer_shape(t, t) == (2, 2)
    assert AsString().infer_shape(t) == (2, 2)
