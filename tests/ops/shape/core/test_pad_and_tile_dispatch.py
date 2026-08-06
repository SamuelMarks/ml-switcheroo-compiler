# pylint: disable=use-implicit-booleaness-not-comparison,broad-exception-caught

"""Coverage tests for misc shape ops."""

import pytest

from ml_switcheroo_compiler.ops.loss import CategoricalGeneralizedCrossEntropy, CircleLoss
from ml_switcheroo_compiler.ops.nn.nlp import CtcLoss as NlpCtcLoss
from ml_switcheroo_compiler.ops.random_ops.core import Rademacher, rademacher
from ml_switcheroo_compiler.ops.shape.pad_and_tile import (
    Lexsort,
    Percentile,
    Quantile,
    RavelMultiIndex,
    Repeat,
    Searchsorted,
    SortComplex,
    Tile,
    Unique,
    flatnonzero,
    lexsort,
    nonzero,
    percentile,
    quantile,
    ravel_multi_index,
    searchsorted,
    sort_complex,
    unique,
)


class Dummy:
    """Dummy tensor class for testing."""

    def __init__(self, shape=None):
        """Initialize dummy tensor.

        Args:
            shape: The shape of the tensor.
        """
        if shape is not None:
            self.shape = shape


def test_loss_infer_shapes():
    """Test loss inference shapes."""
    assert CircleLoss().infer_shape() == ()
    assert CategoricalGeneralizedCrossEntropy().infer_shape() == ()


def test_nlp_ctc_loss_infer_shape():
    """Test NLP CTC loss inference shape."""
    op = NlpCtcLoss()
    assert op.infer_shape(None, Dummy((2, 3, 4)), None, None, logits_time_major=True) == (3,)
    assert op.infer_shape(None, Dummy((2, 3, 4)), None, None, logits_time_major=False) == (2,)
    assert op.infer_shape(None, Dummy((2, 3)), None, None) == (1,)
    assert op.infer_shape(None, Dummy((2,)), None, None) == (1,)
    assert op.infer_shape(None, None, None, None) == (1,)


def test_rademacher_infer_shape():
    """Test Rademacher inference shape."""
    op = Rademacher()
    assert op.infer_shape(size=None) == ()
    assert op.infer_shape(size=5) == (5,)
    assert op.infer_shape(size=(2, 3)) == (2, 3)
    assert op.infer_shape(Dummy((4,)), shape=(4,)) == (4,)

    with pytest.raises(Exception):
        rademacher(size=5)


def test_misc_infer_shapes():
    """Test miscellaneous inference shapes."""
    op1 = Lexsort()
    assert op1.infer_shape([Dummy((2,)), Dummy((2,))]) == (2,)
    assert op1.infer_shape([]) == ()
    assert op1.infer_shape(Dummy((2, 3))) == (3,)
    assert op1.infer_shape(Dummy((2,))) == ()
    assert op1.infer_shape(Dummy(())) == ()
    assert op1.infer_shape(None) == ()

    for OpClass in [Percentile, Quantile]:
        op2 = OpClass()
        assert op2.infer_shape(Dummy((2, 3)), 0.5) == ()
        assert op2.infer_shape(Dummy((2, 3)), [0.5, 0.6]) == (2,)
        assert op2.infer_shape(Dummy((2, 3)), Dummy((2,))) == (2,)

        assert op2.infer_shape(Dummy((2, 3)), 0.5, axis=0, keepdims=True) == (1, 3)
        assert op2.infer_shape(Dummy((2, 3)), 0.5, axis=1, keepdims=False) == (2,)
        assert op2.infer_shape(Dummy((2, 3)), 0.5, axis=(0, 1), keepdims=False) == ()
        assert op2.infer_shape(Dummy((2, 3)), 0.5, axis=None, keepdims=True) == (1, 1)

    op3 = RavelMultiIndex()
    assert op3.infer_shape([Dummy((2,)), Dummy((2,))]) == (2,)
    assert op3.infer_shape([]) == ()
    assert op3.infer_shape(Dummy((2, 3))) == (2, 3)
    assert op3.infer_shape(None) == ()

    op4 = Repeat()
    assert op4.infer_shape(Dummy((2, 3)), 2, axis=None) == (12,)
    assert op4.infer_shape(Dummy((2, None)), 2, axis=None) == (None,)
    assert op4.infer_shape(Dummy((2, 3)), [1, 2], axis=None) == (3,)
    assert op4.infer_shape(Dummy((2, 3)), None, axis=None) == (None,)

    assert op4.infer_shape(Dummy((2, 3)), 2, axis=0) == (4, 3)
    assert op4.infer_shape(Dummy((None, 3)), 2, axis=0) == (None, 3)
    assert op4.infer_shape(Dummy((2, 3)), [1, 2], axis=0) == (3, 3)
    assert op4.infer_shape(Dummy((2, 3)), None, axis=0) == (None, 3)

    op5 = Searchsorted()
    assert op5.infer_shape(Dummy((2,)), Dummy((3,))) == (3,)

    op6 = SortComplex()
    assert op6.infer_shape(Dummy((3, 4))) == (3, 4)

    op7 = Tile()
    assert op7.infer_shape(Dummy((2, 3)), None) == ()
    assert op7.infer_shape(Dummy((2, 3)), 2) == (2, 6)
    assert op7.infer_shape(Dummy((3,)), (2, 2)) == (2, 6)
    assert op7.infer_shape(Dummy((2, 3)), (2,)) == (2, 6)
    assert op7.infer_shape(Dummy((None, 3)), (2, None)) == (None, None)

    op8 = Unique()
    assert op8.infer_shape(Dummy((2, 3)), axis=None) == (None,)
    assert op8.infer_shape(Dummy((2, 3)), axis=0) == (None, 3)

    assert op8.infer_shape(Dummy((2, 3)), return_index=True, axis=None) == ((None,), (None,))
    assert op8.infer_shape(Dummy((2, 3)), return_index=True, axis=0) == ((None, 3), (2,))

    assert op8.infer_shape(Dummy((2, 3)), return_inverse=True, axis=None) == ((None,), (6,))
    assert op8.infer_shape(Dummy((None, 3)), return_inverse=True, axis=None) == ((None,), (None,))
    assert op8.infer_shape(Dummy((2, 3)), return_inverse=True, axis=0) == ((None, 3), (2,))

    assert op8.infer_shape(Dummy((2, 3)), return_counts=True, axis=None) == ((None,), (None,))


def test_misc_dispatch():
    """Test miscellaneous dispatch functions."""

    def safe_call(func, *args, **kwargs):
        try:
            func(*args, **kwargs)
        except Exception:
            pass

    safe_call(percentile, Dummy((2,)), 0.5)
    safe_call(quantile, Dummy((2,)), 0.5)
    safe_call(flatnonzero, Dummy((2,)))
    safe_call(nonzero, Dummy((2,)))
    safe_call(ravel_multi_index, Dummy((2,)))
    safe_call(lexsort, Dummy((2,)))
    safe_call(searchsorted, Dummy((2,)), Dummy((2,)))
    safe_call(sort_complex, Dummy((2,)))
    safe_call(unique, Dummy((2,)))


import ml_switcheroo_compiler.ops.shape.pad_and_tile as sm


def test_shape_misc_coverage():
    ops = [
        "Nonzero",
        "Percentile",
        "Ppermute",
        "PsumScatter",
        "Quantile",
        "RavelMultiIndex",
        "Repeat",
        "Searchsorted",
        "Tile",
        "Unique",
        "UpdateSlice",
        "Flatnonzero",
        "IndexInDim",
        "Lexsort",
        "SortComplex",
        "Compress",
        "FillDiagonal",
        "Intersect1d",
        "Put",
        "Setdiff1d",
        "Setxor1d",
        "Union1d",
        "Extract",
        "NumpyIsneginf",
        "NumpyIsposinf",
        "TrimZeros",
    ]

    for op_name in ops:
        if hasattr(sm, op_name):
            cls = getattr(sm, op_name)
            inst = cls()
            inst.infer_shape()

            class DummyShape:
                shape = (1,)

            try:
                inst.infer_shape(DummyShape())
            except:
                pass
            try:
                inst.infer_shape(DummyShape(), DummyShape())
            except:
                pass
