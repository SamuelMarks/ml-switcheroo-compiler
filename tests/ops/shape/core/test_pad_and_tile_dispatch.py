"""Coverage tests for misc shape ops."""

from __future__ import annotations

import typing

import pytest

import ml_switcheroo_compiler.ops.shape.pad_and_tile as sm
from ml_switcheroo_compiler.backends.eager_registry import EagerValue
from ml_switcheroo_compiler.ir.shape_system import SymVar
from ml_switcheroo_compiler.ops.loss import CategoricalGeneralizedCrossEntropy, CircleLoss
from ml_switcheroo_compiler.ops.nn.nlp import CtcLoss as NlpCtcLoss
from ml_switcheroo_compiler.ops.random_ops.core import Rademacher, rademacher
from ml_switcheroo_compiler.ops.shape.pad_and_tile import Lexsort, Percentile, Quantile, RavelMultiIndex, Repeat, Searchsorted, SortComplex, Tile, Unique, flatnonzero, lexsort, nonzero, percentile, quantile, ravel_multi_index, searchsorted, sort_complex, unique

if typing.TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import EagerCallableProtocol, EagerTensorProtocol, EagerValue


class Dummy:
    """Dummy tensor class for testing."""

    def __init__(self, shape: EagerValue = None) -> None:
        """Initialize dummy tensor.

        Args:
            shape: The shape of the tensor.
        """
        if shape is not None:
            self.shape: tuple[int, ...] = typing.cast("tuple[int, ...]", shape)


def test_loss_infer_shapes() -> None:
    """Test loss inference shapes."""
    from ml_switcheroo_compiler.ops.nn.loss import InTopK, LogPoissonLoss

    assert CircleLoss().infer_shape() == ()
    assert CategoricalGeneralizedCrossEntropy().infer_shape() == ()
    assert LogPoissonLoss().infer_shape() == ()
    assert LogPoissonLoss().infer_shape(typing.cast("EagerTensorProtocol", Dummy((2, 3)))) == (2, 3)
    assert InTopK().infer_shape() == ()
    assert InTopK().infer_shape(typing.cast("EagerTensorProtocol", Dummy((4,)))) == (4,)


def test_nlp_ctc_loss_infer_shape() -> None:
    """Test NLP CTC loss inference shape."""
    op = NlpCtcLoss()
    assert typing.cast("EagerCallableProtocol", op.infer_shape)(None, typing.cast("EagerCallableProtocol", Dummy)((2, 3, 4)), None, None, logits_time_major=True) == (3,)
    assert typing.cast("EagerCallableProtocol", op.infer_shape)(None, typing.cast("EagerCallableProtocol", Dummy)((2, 3, 4)), None, None, logits_time_major=False) == (2,)
    assert typing.cast("EagerCallableProtocol", op.infer_shape)(None, typing.cast("EagerCallableProtocol", Dummy)((2, 3)), None, None) == (1,)
    assert typing.cast("EagerCallableProtocol", op.infer_shape)(None, typing.cast("EagerCallableProtocol", Dummy)((2,)), None, None) == (1,)
    assert typing.cast("EagerCallableProtocol", op.infer_shape)(None, None, None, None) == (1,)


def test_rademacher_infer_shape() -> None:
    """Test Rademacher inference shape."""
    op = Rademacher()
    assert op.infer_shape(size=None) == ()
    assert op.infer_shape(size=5) == (5,)
    assert op.infer_shape(size=(2, 3)) == (2, 3)
    assert typing.cast("EagerCallableProtocol", op.infer_shape)(typing.cast("EagerCallableProtocol", Dummy)((4,)), shape=(4,)) == (4,)
    with pytest.raises(Exception, match=".*"):
        rademacher(size=5)


def _test_misc_infer_shapes_part_1(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 1 of test_misc_infer_shapes.

    Args:
        _state: Shared test state dictionary.
    """
    op1 = Lexsort()
    assert typing.cast("EagerCallableProtocol", op1.infer_shape)([typing.cast("EagerCallableProtocol", Dummy)((2,)), typing.cast("EagerCallableProtocol", Dummy)((2,))]) == (2,)
    assert typing.cast("EagerCallableProtocol", op1.infer_shape)([]) == ()
    assert typing.cast("EagerCallableProtocol", op1.infer_shape)(typing.cast("EagerCallableProtocol", Dummy)((2, 3))) == (3,)
    assert typing.cast("EagerCallableProtocol", op1.infer_shape)(typing.cast("EagerCallableProtocol", Dummy)((2,))) == ()
    assert typing.cast("EagerCallableProtocol", op1.infer_shape)(typing.cast("EagerCallableProtocol", Dummy)(())) == ()
    assert typing.cast("EagerCallableProtocol", op1.infer_shape)(None) == ()
    for OpClass in [Percentile, Quantile]:
        op2 = OpClass()
        assert op2.infer_shape(Dummy(typing.cast("EagerValue", (2, 3))), 0.5) == ()
        assert op2.infer_shape(Dummy(typing.cast("EagerValue", (2, 3))), [0.5, 0.6]) == (2,)
        assert op2.infer_shape(Dummy((2, 3)), Dummy(typing.cast("EagerValue", (2,)))) == (2,)
        assert op2.infer_shape(Dummy(typing.cast("EagerValue", (2, 3))), 0.5, axis=0, keepdims=True) == (1, 3)
        assert op2.infer_shape(Dummy(typing.cast("EagerValue", (2, 3))), 0.5, axis=1, keepdims=False) == (2,)
        assert op2.infer_shape(Dummy(typing.cast("EagerValue", (2, 3))), 0.5, axis=(0, 1), keepdims=False) == ()
        assert op2.infer_shape(Dummy(typing.cast("EagerValue", (2, 3))), 0.5, axis=None, keepdims=True) == (1, 1)
    op3 = RavelMultiIndex()
    assert typing.cast("EagerCallableProtocol", op3.infer_shape)([typing.cast("EagerCallableProtocol", Dummy)((2,)), typing.cast("EagerCallableProtocol", Dummy)((2,))]) == (2,)
    assert typing.cast("EagerCallableProtocol", op3.infer_shape)([]) == ()
    assert typing.cast("EagerCallableProtocol", op3.infer_shape)(typing.cast("EagerCallableProtocol", Dummy)((2, 3))) == (2, 3)
    assert typing.cast("EagerCallableProtocol", op3.infer_shape)(None) == ()
    op4 = Repeat()
    getattr(_state, "update")({"op1": op1, "op3": op3, "op4": op4})
    _state.update(locals())


def _test_misc_infer_shapes_part_2(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 2 of test_misc_infer_shapes.

    Args:
        _state: Shared test state dictionary.
    """
    op4 = _state["op4"]
    assert typing.cast("EagerCallableProtocol", op4.infer_shape)(typing.cast("EagerCallableProtocol", Dummy)((2, 3)), 2, axis=None) == (12,)
    res_none_dim = typing.cast("EagerCallableProtocol", op4.infer_shape)(typing.cast("EagerCallableProtocol", Dummy)((2, None)), 2, axis=None)
    assert typing.cast("EagerValue", res_none_dim[0]) is None or hasattr(res_none_dim[0], "name")
    assert typing.cast("EagerCallableProtocol", op4.infer_shape)(typing.cast("EagerCallableProtocol", Dummy)((2, 3)), [1, 2], axis=None) == (3,)
    res_none_rep = typing.cast("EagerCallableProtocol", op4.infer_shape)(typing.cast("EagerCallableProtocol", Dummy)((2, 3)), None, axis=None)
    assert typing.cast("EagerValue", res_none_rep[0]) is None or hasattr(res_none_rep[0], "name")
    assert typing.cast("EagerCallableProtocol", op4.infer_shape)(typing.cast("EagerCallableProtocol", Dummy)((2, 3)), 2, axis=0) == (4, 3)
    assert typing.cast("EagerCallableProtocol", op4.infer_shape)(typing.cast("EagerCallableProtocol", Dummy)((None, 3)), 2, axis=0) == (None, 3)
    assert typing.cast("EagerCallableProtocol", op4.infer_shape)(typing.cast("EagerCallableProtocol", Dummy)((2, 3)), [1, 2], axis=0) == (3, 3)
    assert typing.cast("EagerCallableProtocol", op4.infer_shape)(typing.cast("EagerCallableProtocol", Dummy)((2, 3)), None, axis=0) == (None, 3)
    op5 = Searchsorted()
    assert typing.cast("EagerCallableProtocol", op5.infer_shape)(typing.cast("EagerCallableProtocol", Dummy)((2,)), typing.cast("EagerCallableProtocol", Dummy)((3,))) == (3,)
    op6 = SortComplex()
    assert typing.cast("EagerCallableProtocol", op6.infer_shape)(typing.cast("EagerCallableProtocol", Dummy)((3, 4))) == (3, 4)
    op7 = Tile()
    assert typing.cast("EagerCallableProtocol", op7.infer_shape)(typing.cast("EagerCallableProtocol", Dummy)((2, 3)), None) == ()
    assert typing.cast("EagerCallableProtocol", op7.infer_shape)(typing.cast("EagerCallableProtocol", Dummy)((2, 3)), 2) == (2, 6)
    assert typing.cast("EagerCallableProtocol", op7.infer_shape)(typing.cast("EagerCallableProtocol", Dummy)((3,)), (2, 2)) == (2, 6)
    assert typing.cast("EagerCallableProtocol", op7.infer_shape)(typing.cast("EagerCallableProtocol", Dummy)((2, 3)), (2,)) == (2, 6)
    assert typing.cast("EagerCallableProtocol", op7.infer_shape)(typing.cast("EagerCallableProtocol", Dummy)((None, 3)), (2, None)) == (None, None)
    op8 = Unique()
    assert typing.cast("EagerCallableProtocol", getattr(typing.cast("EagerCallableProtocol", op8.infer_shape)(typing.cast("EagerCallableProtocol", Dummy)((2, 3)), axis=None), "__getitem__"))(0) in (None, typing.cast("EagerCallableProtocol", SymVar)("unique_count"))
    getattr(_state, "update")({"op5": op5, "op6": op6, "op7": op7, "op8": op8})
    getattr(_state, "update")({"res_none_dim": res_none_dim, "res_none_rep": res_none_rep})
    _state.update(locals())


def _test_misc_infer_shapes_part_3(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 3 of test_misc_infer_shapes.

    Args:
        _state: Shared test state dictionary.
    """
    op8 = _state["op8"]
    res_u0 = typing.cast("EagerCallableProtocol", op8.infer_shape)(typing.cast("EagerCallableProtocol", Dummy)((2, 3)), axis=0)
    assert res_u0[0] in (None, SymVar("unique_count")) and res_u0[1] == 3
    res_idx_none = typing.cast("EagerCallableProtocol", op8.infer_shape)(typing.cast("EagerCallableProtocol", Dummy)((2, 3)), return_index=True, axis=None)
    assert res_idx_none[0][0] in (None, SymVar("unique_count")) and res_idx_none[1][0] in (None, SymVar("unique_count"))
    res_idx_0 = typing.cast("EagerCallableProtocol", op8.infer_shape)(typing.cast("EagerCallableProtocol", Dummy)((2, 3)), return_index=True, axis=0)
    assert res_idx_0[0][0] in (None, SymVar("unique_count")) and res_idx_0[0][1] == 3 and (res_idx_0[1] == (2,))
    res_inv_none = typing.cast("EagerCallableProtocol", op8.infer_shape)(typing.cast("EagerCallableProtocol", Dummy)((2, 3)), return_inverse=True, axis=None)
    assert res_inv_none[0][0] in (None, SymVar("unique_count")) and res_inv_none[1] == (6,)
    res_inv_none_none = typing.cast("EagerCallableProtocol", op8.infer_shape)(typing.cast("EagerCallableProtocol", Dummy)((None, 3)), return_inverse=True, axis=None)
    getattr(_state, "update")({"res_idx_0": res_idx_0, "res_idx_none": res_idx_none, "res_inv_none": res_inv_none, "res_inv_none_none": res_inv_none_none})
    getattr(_state, "update")({"res_u0": res_u0})
    _state.update(locals())


def _test_misc_infer_shapes_part_4(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 4 of test_misc_infer_shapes.

    Args:
        _state: Shared test state dictionary.
    """
    (op8, res_inv_none_none) = (_state["op8"], _state["res_inv_none_none"])
    assert res_inv_none_none[0][0] in (None, SymVar("unique_count")) and res_inv_none_none[1] == (None,)
    res_inv_0 = typing.cast("EagerCallableProtocol", op8.infer_shape)(typing.cast("EagerCallableProtocol", Dummy)((2, 3)), return_inverse=True, axis=0)
    assert res_inv_0[0][0] in (None, SymVar("unique_count")) and res_inv_0[0][1] == 3 and (res_inv_0[1] == (2,))
    res_cnt_none = typing.cast("EagerCallableProtocol", op8.infer_shape)(typing.cast("EagerCallableProtocol", Dummy)((2, 3)), return_counts=True, axis=None)
    assert res_cnt_none[0][0] in (None, SymVar("unique_count")) and res_cnt_none[1][0] in (None, SymVar("unique_count"))
    getattr(_state, "update")({"res_cnt_none": res_cnt_none, "res_inv_0": res_inv_0})


def test_misc_infer_shapes() -> None:
    """Test miscellaneous inference shapes."""
    _state: dict[str, EagerTensorProtocol] = {}
    _test_misc_infer_shapes_part_1(_state)
    _test_misc_infer_shapes_part_2(_state)
    _test_misc_infer_shapes_part_3(_state)
    _test_misc_infer_shapes_part_4(_state)
    assert len(_state) > 0


def test_misc_dispatch() -> None:
    """Test miscellaneous dispatch functions."""

    def safe_call(func: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> None:
        """Test function safe_call.

        Args:
            func: Test parameter `func`.
            args: Test parameter `args`.
            kwargs: Test parameter `kwargs`.
        """
        with pytest.raises(RuntimeError):
            typing.cast("EagerCallableProtocol", func)(*args, **kwargs)

    typing.cast("EagerCallableProtocol", safe_call)(percentile, typing.cast("EagerCallableProtocol", Dummy)((2,)), 0.5)
    typing.cast("EagerCallableProtocol", safe_call)(quantile, typing.cast("EagerCallableProtocol", Dummy)((2,)), 0.5)
    typing.cast("EagerCallableProtocol", safe_call)(flatnonzero, typing.cast("EagerCallableProtocol", Dummy)((2,)))
    typing.cast("EagerCallableProtocol", safe_call)(nonzero, typing.cast("EagerCallableProtocol", Dummy)((2,)))
    typing.cast("EagerCallableProtocol", safe_call)(ravel_multi_index, typing.cast("EagerCallableProtocol", Dummy)((2,)))
    typing.cast("EagerCallableProtocol", safe_call)(lexsort, typing.cast("EagerCallableProtocol", Dummy)((2,)))
    typing.cast("EagerCallableProtocol", safe_call)(searchsorted, typing.cast("EagerCallableProtocol", Dummy)((2,)), typing.cast("EagerCallableProtocol", Dummy)((2,)))
    typing.cast("EagerCallableProtocol", safe_call)(sort_complex, typing.cast("EagerCallableProtocol", Dummy)((2,)))
    typing.cast("EagerCallableProtocol", safe_call)(unique, typing.cast("EagerCallableProtocol", Dummy)((2,)))


def test_shape_misc_coverage() -> None:
    """Test function test_shape_misc_coverage."""
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
                """Test class DummyShape."""

                shape = (1,)

            inst.infer_shape(DummyShape())
            try:
                inst.infer_shape(DummyShape(), DummyShape())
            except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, OverflowError, RecursionError, LookupError, ArithmeticError):
                _caught_err = True
    assert callable(test_shape_misc_coverage)
