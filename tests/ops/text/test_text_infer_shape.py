"""Test infer_shape for text ops."""

from __future__ import annotations

import typing

from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.ir.shape_system import SymVar
from ml_switcheroo_compiler.ops.text.ops import AsString, EditDistance, Hashing, IntegerLookup, Lookup, RegexFullMatch, RegexReplace, StringJoin, StringLength, StringLookup, StringLower, StringSplit, StringSubstr, StringToHash, StringToNumber, StringUpper, TextVectorization

if typing.TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import EagerCallableProtocol


def test_text_infer_shape() -> None:
    """Test the text infer shape behavior.

    Returns:
        TensorOrScalar: The inferred shape or computed result.
    """
    try:
        t = Tensor("dummy", TensorConfig((2, 2), DType.String, None))
        assert StringToHash().infer_shape() == ()
        assert RegexReplace().infer_shape() == ()
        assert StringSplit().infer_shape() == ()
        assert typing.cast("EagerCallableProtocol", Lookup)().infer_shape(t) == (2, 2)
        assert typing.cast("EagerCallableProtocol", Hashing)().infer_shape(t) == (2, 2)
        assert typing.cast("EagerCallableProtocol", StringLookup)().infer_shape(t) == (2, 2)
        assert typing.cast("EagerCallableProtocol", IntegerLookup)().infer_shape(t) == (2, 2)
        assert typing.cast("EagerCallableProtocol", TextVectorization)().infer_shape(t) == (2, 2, typing.cast("EagerCallableProtocol", SymVar)("seq_len"))
        assert typing.cast("EagerCallableProtocol", StringToNumber)().infer_shape(t) == (2, 2)
        assert typing.cast("EagerCallableProtocol", StringLower)().infer_shape(t) == (2, 2)
        assert typing.cast("EagerCallableProtocol", StringUpper)().infer_shape(t) == (2, 2)
        assert StringJoin().infer_shape() == ()
        assert typing.cast("EagerCallableProtocol", StringLength)().infer_shape(t) == (2, 2)
        assert typing.cast("EagerCallableProtocol", StringSubstr)().infer_shape(t) == (2, 2)
        assert typing.cast("EagerCallableProtocol", RegexFullMatch)().infer_shape(t) == (2, 2)
        assert typing.cast("EagerCallableProtocol", EditDistance)().infer_shape(t, t) == (2,)
        assert typing.cast("EagerCallableProtocol", AsString)().infer_shape(t) == (2, 2)
    except Exception as e:
        raise e
        pass
