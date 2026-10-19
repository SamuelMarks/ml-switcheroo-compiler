"""Unit tests for text operations shape inference and token batch dimensions."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.ir.shape_system import SymVar
from ml_switcheroo_compiler.ops.text.ops import (
    ArrayRepr,
    ArrayStr,
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

if TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import EagerCallableProtocol, EagerTensorProtocol, EagerValue


def _make_tensor(shape: tuple[int, ...]) -> Tensor:
    """Helper to create dummy tensor with specified shape.

    Args:
        shape (tuple[int, ...]): Shape of tensor.

    Returns:
        Tensor: Dummy tensor instance.
    """
    return Tensor(None, TensorConfig(shape, "string", "cpu"))


def test_text_regex_and_hashing_shapes() -> None:
    """Verify StringToHash, RegexReplace, Hashing, and RegexFullMatch shape preservation."""
    op_hash = StringToHash()
    op_regex = RegexReplace()
    op_hashing = Hashing()
    op_match = RegexFullMatch()

    t = _make_tensor((4, 16))

    assert cast("EagerCallableProtocol", op_hash.infer_shape)(t) == (4, 16)
    assert cast("EagerCallableProtocol", op_regex.infer_shape)(t) == (4, 16)
    assert cast("EagerCallableProtocol", op_hashing.infer_shape)(t) == (4, 16)
    assert cast("EagerCallableProtocol", op_match.infer_shape)(t) == (4, 16)

    # Fallbacks
    assert op_hash.infer_shape() == ()
    assert op_regex.infer_shape() == ()

    class MockMeta:
        """Test class MockMeta."""

        def __init__(self, sm: EagerValue) -> None:
            """Test function __init__.

            Args:
                sm: Test parameter `sm`.
            """
            self.shape_metadata = sm

    class Plain:
        """Test class Plain."""

        pass

    assert cast("EagerCallableProtocol", op_hash.infer_shape)(cast("EagerCallableProtocol", MockMeta)((5, 10))) == (5, 10)
    assert cast("EagerCallableProtocol", op_hash.infer_shape)(cast("EagerCallableProtocol", MockMeta)(())) == ()
    assert cast("EagerCallableProtocol", op_hash.infer_shape)(cast("EagerCallableProtocol", MockMeta)(None)) == ()
    assert cast("EagerCallableProtocol", op_hash.infer_shape)(cast("EagerCallableProtocol", Plain)()) == ()


def test_string_split_token_dimensions() -> None:
    """Verify StringSplit output shape with dynamic and explicit token dimensions."""
    op_split = StringSplit()

    t = _make_tensor((8,))

    # Dynamic splits: returns (8, SymVar("num_tokens"))
    out_dynamic = cast("EagerCallableProtocol", op_split.infer_shape)(t)
    assert cast("EagerTensorProtocol", getattr(out_dynamic, "__getitem__")(0)) == 8
    assert isinstance(cast("EagerTensorProtocol", getattr(out_dynamic, "__getitem__")(1)), SymVar)

    # Explicit max_splits=3 -> (8, 4)
    out_bounded = cast("EagerCallableProtocol", op_split.infer_shape)(t, max_splits=3)
    assert out_bounded == (8, 4)

    # Fallback
    assert op_split.infer_shape() == ()


def test_string_lookups_and_conversions() -> None:
    """Verify Lookup, StringLookup, IntegerLookup, StringToNumber, and AsString."""
    op_lookup = Lookup()
    op_str_lookup = StringLookup()
    op_int_lookup = IntegerLookup()
    op_to_num = StringToNumber()
    op_as_str = AsString()

    t = _make_tensor((2, 5, 10))

    assert cast("EagerCallableProtocol", op_lookup.infer_shape)(t) == (2, 5, 10)
    assert cast("EagerCallableProtocol", op_str_lookup.infer_shape)(t) == (2, 5, 10)
    assert cast("EagerCallableProtocol", op_int_lookup.infer_shape)(t) == (2, 5, 10)
    assert cast("EagerCallableProtocol", op_to_num.infer_shape)(t) == (2, 5, 10)
    assert cast("EagerCallableProtocol", op_as_str.infer_shape)(t) == (2, 5, 10)


def test_string_manipulation_ops() -> None:
    """Verify StringLower, StringUpper, StringLength, StringSubstr, and StringJoin."""
    op_lower = StringLower()
    op_upper = StringUpper()
    op_len = StringLength()
    op_sub = StringSubstr()
    op_join = StringJoin()

    t = _make_tensor((3, 6))

    assert cast("EagerCallableProtocol", op_lower.infer_shape)(t) == (3, 6)
    assert cast("EagerCallableProtocol", op_upper.infer_shape)(t) == (3, 6)
    assert cast("EagerCallableProtocol", op_len.infer_shape)(t) == (3, 6)
    assert cast("EagerCallableProtocol", op_sub.infer_shape)(t) == (3, 6)

    # StringJoin reducing axis 1
    assert cast("EagerCallableProtocol", op_join.infer_shape)(t, axis=1) == (3,)
    assert cast("EagerCallableProtocol", op_join.infer_shape)(t, axis=-1) == (3,)
    assert cast("EagerCallableProtocol", op_join.infer_shape)(t, axis=10) == (3, 6)
    assert cast("EagerCallableProtocol", op_join.infer_shape)([]) == ()

    # StringJoin list of tensors
    assert cast("EagerCallableProtocol", op_join.infer_shape)([t, t]) == (3, 6)

    # Fallbacks
    assert op_join.infer_shape() == ()


def test_edit_distance_and_vectorization() -> None:
    """Verify EditDistance sequence reduction and TextVectorization dimensions."""
    op_edit = EditDistance()
    op_vec = TextVectorization()

    hypo = _make_tensor((4, 20))
    truth = _make_tensor((4, 25))

    # EditDistance reduces sequence dimension: (4, 20) -> (4,)
    assert cast("EagerCallableProtocol", op_edit.infer_shape)(hypo, truth) == (4,)
    assert cast("EagerCallableProtocol", op_edit.infer_shape)(cast("EagerCallableProtocol", _make_tensor)(()), cast("EagerCallableProtocol", _make_tensor)(())) == ()

    # TextVectorization with explicit output_sequence_length
    sentences = _make_tensor((8,))
    assert cast("EagerCallableProtocol", op_vec.infer_shape)(sentences, output_sequence_length=128) == (8, 128)

    # TextVectorization with count mode and max_tokens
    assert cast("EagerCallableProtocol", op_vec.infer_shape)(sentences, output_mode="count", max_tokens=1000) == (1000,)
    assert cast("EagerCallableProtocol", op_vec.infer_shape)(cast("EagerCallableProtocol", _make_tensor)((4, 8)), output_mode="count", max_tokens=500) == (4, 500)

    # TextVectorization default with SymVar
    out_default = cast("EagerCallableProtocol", op_vec.infer_shape)(sentences)
    assert cast("EagerTensorProtocol", getattr(out_default, "__getitem__")(0)) == 8
    assert isinstance(cast("EagerTensorProtocol", getattr(out_default, "__getitem__")(1)), SymVar)

    # TextVectorization empty fallbacks
    assert op_vec.infer_shape() == ()
    assert cast("EagerCallableProtocol", op_vec.infer_shape)(()) == ()


def test_array_repr_and_str() -> None:
    """Verify ArrayRepr and ArrayStr scalar string shapes."""
    op_repr = ArrayRepr()
    op_str = ArrayStr()

    t = _make_tensor((2, 3))
    assert cast("EagerCallableProtocol", op_repr.infer_shape)(t) == ()
    assert cast("EagerCallableProtocol", op_str.infer_shape)(t) == ()
