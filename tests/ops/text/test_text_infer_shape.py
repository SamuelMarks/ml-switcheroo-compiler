# ruff: noqa: E501
"""Test infer_shape for text ops."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.ir.shape_system import SymVar
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

if TYPE_CHECKING:
    from collections.abc import Callable

    from ml_switcheroo_compiler.backends.eager_registry import EagerTensorProtocol


def test_text_infer_shape() -> None:
    """Test the text infer shape behavior.

    Returns:
        object: The inferred shape or computed result.
    """
    try:
        "Test text ops infer_shape."
        t = Tensor("dummy", TensorConfig((2, 2), DType.String, None))
        assert StringToHash().infer_shape() == ()
        assert RegexReplace().infer_shape() == ()
        assert StringSplit().infer_shape() == ()
        assert cast("Callable[..., EagerTensorProtocol]", Lookup)().infer_shape(t) == (2, 2)
        assert cast("Callable[..., EagerTensorProtocol]", Hashing)().infer_shape(t) == (2, 2)
        assert cast("Callable[..., EagerTensorProtocol]", StringLookup)().infer_shape(t) == (2, 2)
        assert cast("Callable[..., EagerTensorProtocol]", IntegerLookup)().infer_shape(t) == (2, 2)
        assert cast("Callable[..., EagerTensorProtocol]", TextVectorization)().infer_shape(t) == (2, 2, cast("Callable[..., EagerTensorProtocol]", SymVar)("seq_len"))
        assert cast("Callable[..., EagerTensorProtocol]", StringToNumber)().infer_shape(t) == (2, 2)
        assert cast("Callable[..., EagerTensorProtocol]", StringLower)().infer_shape(t) == (2, 2)
        assert cast("Callable[..., EagerTensorProtocol]", StringUpper)().infer_shape(t) == (2, 2)
        assert StringJoin().infer_shape() == ()
        assert cast("Callable[..., EagerTensorProtocol]", StringLength)().infer_shape(t) == (2, 2)
        assert cast("Callable[..., EagerTensorProtocol]", StringSubstr)().infer_shape(t) == (2, 2)
        assert cast("Callable[..., EagerTensorProtocol]", RegexFullMatch)().infer_shape(t) == (2, 2)
        assert cast("Callable[..., EagerTensorProtocol]", EditDistance)().infer_shape(t, t) == (2,)
        assert cast("Callable[..., EagerTensorProtocol]", AsString)().infer_shape(t) == (2, 2)
    except Exception as e:
        raise e
        pass
