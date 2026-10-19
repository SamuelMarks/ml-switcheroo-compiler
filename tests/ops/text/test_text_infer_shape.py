"""Test infer_shape for text ops."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.ir.shape_system import SymVar
from ml_switcheroo_compiler.ops.text.ops import AsString, EditDistance, Hashing, IntegerLookup, Lookup, RegexFullMatch, RegexReplace, StringJoin, StringLength, StringLookup, StringLower, StringSplit, StringSubstr, StringToHash, StringToNumber, StringUpper, TextVectorization

if TYPE_CHECKING:
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
        assert cast("EagerCallableProtocol", Lookup)().infer_shape(t) == (2, 2)
        assert cast("EagerCallableProtocol", Hashing)().infer_shape(t) == (2, 2)
        assert cast("EagerCallableProtocol", StringLookup)().infer_shape(t) == (2, 2)
        assert cast("EagerCallableProtocol", IntegerLookup)().infer_shape(t) == (2, 2)
        assert cast("EagerCallableProtocol", TextVectorization)().infer_shape(t) == (2, 2, cast("EagerCallableProtocol", SymVar)("seq_len"))
        assert cast("EagerCallableProtocol", StringToNumber)().infer_shape(t) == (2, 2)
        assert cast("EagerCallableProtocol", StringLower)().infer_shape(t) == (2, 2)
        assert cast("EagerCallableProtocol", StringUpper)().infer_shape(t) == (2, 2)
        assert StringJoin().infer_shape() == ()
        assert cast("EagerCallableProtocol", StringLength)().infer_shape(t) == (2, 2)
        assert cast("EagerCallableProtocol", StringSubstr)().infer_shape(t) == (2, 2)
        assert cast("EagerCallableProtocol", RegexFullMatch)().infer_shape(t) == (2, 2)
        assert cast("EagerCallableProtocol", EditDistance)().infer_shape(t, t) == (2,)
        assert cast("EagerCallableProtocol", AsString)().infer_shape(t) == (2, 2)
    except Exception as e:
        raise e
        pass
