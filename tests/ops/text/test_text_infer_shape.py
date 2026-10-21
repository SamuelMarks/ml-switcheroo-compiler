"""Test infer_shape for text ops."""

from __future__ import annotations

import typing

from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.ir.shape_system import SymVar
from ml_switcheroo_compiler.ops.text.ops import AsString, EditDistance, Hashing, IntegerLookup, Lookup, RegexFullMatch, RegexReplace, StringJoin, StringLength, StringLookup, StringLower, StringSplit, StringSubstr, StringToHash, StringToNumber, StringUpper, TextVectorization
from tests.testing_utils import as_callable

if typing.TYPE_CHECKING:
    pass


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
        assert as_callable(Lookup)().infer_shape(t) == (2, 2)
        assert as_callable(Hashing)().infer_shape(t) == (2, 2)
        assert as_callable(StringLookup)().infer_shape(t) == (2, 2)
        assert as_callable(IntegerLookup)().infer_shape(t) == (2, 2)
        assert as_callable(TextVectorization)().infer_shape(t) == (2, 2, as_callable(SymVar)("seq_len"))
        assert as_callable(StringToNumber)().infer_shape(t) == (2, 2)
        assert as_callable(StringLower)().infer_shape(t) == (2, 2)
        assert as_callable(StringUpper)().infer_shape(t) == (2, 2)
        assert StringJoin().infer_shape() == ()
        assert as_callable(StringLength)().infer_shape(t) == (2, 2)
        assert as_callable(StringSubstr)().infer_shape(t) == (2, 2)
        assert as_callable(RegexFullMatch)().infer_shape(t) == (2, 2)
        assert as_callable(EditDistance)().infer_shape(t, t) == (2,)
        assert as_callable(AsString)().infer_shape(t) == (2, 2)
    except Exception as e:
        raise e
        pass
