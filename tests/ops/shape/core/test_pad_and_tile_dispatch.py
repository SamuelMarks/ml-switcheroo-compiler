"""Coverage tests for misc shape ops."""

from __future__ import annotations

import typing
from collections.abc import Sequence

import pytest

import ml_switcheroo_compiler.ops.shape.pad_and_tile as sm
from ml_switcheroo_compiler.backends.eager_registry import BackendArg
from ml_switcheroo_compiler.backends.numpy.eager import ensure_ndarray
from ml_switcheroo_compiler.core.shape import OpValue
from ml_switcheroo_compiler.ir.shape_system import SymVar
from ml_switcheroo_compiler.ops.nn.loss import CategoricalGeneralizedCrossEntropy, CircleLoss
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
from tests.testing_utils import EagerTensorProtocol, MockTestStub, adapt_backend_module, require_callable

if typing.TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import BackendArg, EagerTensorProtocol


class Dummy(MockTestStub):
    """Dummy tensor class for testing."""

    def __init__(self, shape: Sequence[int | None] | None = None) -> None:
        """Initialize dummy tensor.

        Args:
            shape: The shape of the tensor.
        """
        if shape is not None:
            self.shape: tuple[int | None, ...] = tuple(shape)


def test_loss_infer_shapes() -> None:
    """Test loss inference shapes."""
    from ml_switcheroo_compiler.ops.nn.loss import InTopK, LogPoissonLoss

    assert CircleLoss().infer_shape() == ()
    assert CategoricalGeneralizedCrossEntropy().infer_shape() == ()
    assert LogPoissonLoss().infer_shape() == ()
    assert LogPoissonLoss().infer_shape(Dummy((2, 3))) == (2, 3)
    assert InTopK().infer_shape() == ()
    assert InTopK().infer_shape(Dummy((4,))) == (4,)


def test_nlp_ctc_loss_infer_shape() -> None:
    """Test NLP CTC loss inference shape."""
    op = NlpCtcLoss()
    assert op.infer_shape(None, Dummy((2, 3, 4)), None, None, logits_time_major=True) == (3,)
    assert op.infer_shape(None, Dummy((2, 3, 4)), None, None, logits_time_major=False) == (2,)
    assert op.infer_shape(None, Dummy((2, 3)), None, None) == (1,)
    assert op.infer_shape(None, Dummy((2,)), None, None) == (1,)
    assert op.infer_shape(None, None, None, None) == (1,)


def test_rademacher_infer_shape() -> None:
    """Test Rademacher inference shape."""
    op = Rademacher()
    assert op.infer_shape(size=None) == ()
    assert op.infer_shape(size=5) == (5,)
    assert op.infer_shape(size=(2, 3)) == (2, 3)
    assert op.infer_shape(Dummy((4,)), shape=(4,)) == (4,)
    with pytest.raises(Exception, match=".*"):
        rademacher(size=5)


class _TestMiscInferShapesState(typing.TypedDict, total=False):
    """Typed shared state for `_test_misc_infer_shapes` split helpers."""

    op4: EagerTensorProtocol | OpValue
    op8: EagerTensorProtocol | OpValue
    res_inv_none_none: EagerTensorProtocol


def _test_misc_infer_shapes_part_1(_state: _TestMiscInferShapesState) -> None:
    """Execute part 1 of test_misc_infer_shapes.

    Args:
        _state: Shared test state dictionary.
    """
    op1 = Lexsort()
    assert op1.infer_shape([Dummy((2,)), Dummy((2,))]) == (2,)
    assert op1.infer_shape([]) == ()
    assert op1.infer_shape(Dummy((2, 3))) == (3,)
    assert op1.infer_shape(Dummy((2,))) == ()
    assert op1.infer_shape(Dummy(())) == ()
    assert op1.infer_shape(None) == ()
    for OpClass in [Percentile, Quantile]:
        op2 = OpClass()
        assert op2.dispatch_infer_shape(Dummy((2, 3)), 0.5) == ()
        assert op2.dispatch_infer_shape(Dummy((2, 3)), [0.5, 0.6]) == (2,)
        assert op2.dispatch_infer_shape(Dummy((2, 3)), Dummy((2,))) == (2,)
        assert op2.dispatch_infer_shape(Dummy((2, 3)), 0.5, axis=0, keepdims=True) == (1, 3)
        assert op2.dispatch_infer_shape(Dummy((2, 3)), 0.5, axis=1, keepdims=False) == (2,)
        assert op2.dispatch_infer_shape(Dummy((2, 3)), 0.5, axis=(0, 1), keepdims=False) == ()
        assert op2.dispatch_infer_shape(Dummy((2, 3)), 0.5, axis=None, keepdims=True) == (1, 1)
    op3 = RavelMultiIndex()
    assert op3.infer_shape([Dummy((2,)), Dummy((2,))]) == (2,)
    assert op3.infer_shape([]) == ()
    assert op3.infer_shape(Dummy((2, 3))) == (2, 3)
    assert op3.infer_shape(None) == ()
    op4 = Repeat()
    _state.update({"op4": op4})


def _test_misc_infer_shapes_part_2(_state: _TestMiscInferShapesState) -> None:
    """Execute part 2 of test_misc_infer_shapes.

    Args:
        _state: Shared test state dictionary.
    """
    op4 = _state["op4"]
    assert adapt_backend_module(op4).require_symbol("infer_shape")(Dummy((2, 3)), 2, axis=None) == (12,)
    res_none_dim = adapt_backend_module(op4).require_symbol("infer_shape")(Dummy((2, None)), 2, axis=None)
    assert adapt_backend_module(res_none_dim[0]).has_symbol("name")
    assert adapt_backend_module(op4).require_symbol("infer_shape")(Dummy((2, 3)), [1, 2], axis=None) == (3,)
    res_none_rep = adapt_backend_module(op4).require_symbol("infer_shape")(Dummy((2, 3)), None, axis=None)
    assert adapt_backend_module(res_none_rep[0]).has_symbol("name")
    assert adapt_backend_module(op4).require_symbol("infer_shape")(Dummy((2, 3)), 2, axis=0) == (4, 3)
    assert adapt_backend_module(op4).require_symbol("infer_shape")(Dummy((None, 3)), 2, axis=0) == (None, 3)
    assert adapt_backend_module(op4).require_symbol("infer_shape")(Dummy((2, 3)), [1, 2], axis=0) == (3, 3)
    assert adapt_backend_module(op4).require_symbol("infer_shape")(Dummy((2, 3)), None, axis=0) == (None, 3)
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
    assert ensure_ndarray(op8.infer_shape(Dummy((2, 3)), axis=None))[0] in (
        None,
        SymVar("unique_count"),
    )
    _state.update({"op8": op8})
    _state.update({"op4": op4})


def _test_misc_infer_shapes_part_3(_state: _TestMiscInferShapesState) -> None:
    """Execute part 3 of test_misc_infer_shapes.

    Args:
        _state: Shared test state dictionary.
    """
    op8 = _state["op8"]
    res_u0 = adapt_backend_module(op8).require_symbol("infer_shape")(Dummy((2, 3)), axis=0)
    assert res_u0[0] in (None, SymVar("unique_count")) and res_u0[1] == 3
    res_idx_none = adapt_backend_module(op8).require_symbol("infer_shape")(Dummy((2, 3)), return_index=True, axis=None)
    assert res_idx_none[0][0] in (None, SymVar("unique_count")) and res_idx_none[1][0] in (None, SymVar("unique_count"))
    res_idx_0 = adapt_backend_module(op8).require_symbol("infer_shape")(Dummy((2, 3)), return_index=True, axis=0)
    assert res_idx_0[0][0] in (None, SymVar("unique_count")) and res_idx_0[0][1] == 3 and (res_idx_0[1] == (2,))
    res_inv_none = adapt_backend_module(op8).require_symbol("infer_shape")(
        Dummy((2, 3)), return_inverse=True, axis=None
    )
    assert res_inv_none[0][0] in (None, SymVar("unique_count")) and res_inv_none[1] == (6,)
    res_inv_none_none = adapt_backend_module(op8).require_symbol("infer_shape")(
        Dummy((None, 3)), return_inverse=True, axis=None
    )
    _state.update({"res_inv_none_none": res_inv_none_none})
    _state.update({"op8": op8})


def _test_misc_infer_shapes_part_4(_state: _TestMiscInferShapesState) -> None:
    """Execute part 4 of test_misc_infer_shapes.

    Args:
        _state: Shared test state dictionary.
    """
    (op8, res_inv_none_none) = (_state["op8"], _state["res_inv_none_none"])
    assert res_inv_none_none[0][0] in (None, SymVar("unique_count")) and res_inv_none_none[1] == (None,)
    res_inv_0 = adapt_backend_module(op8).require_symbol("infer_shape")(Dummy((2, 3)), return_inverse=True, axis=0)
    assert res_inv_0[0][0] in (None, SymVar("unique_count")) and res_inv_0[0][1] == 3 and (res_inv_0[1] == (2,))
    res_cnt_none = adapt_backend_module(op8).require_symbol("infer_shape")(Dummy((2, 3)), return_counts=True, axis=None)
    assert res_cnt_none[0][0] in (None, SymVar("unique_count")) and res_cnt_none[1][0] in (None, SymVar("unique_count"))


def test_misc_infer_shapes() -> None:
    """Test miscellaneous inference shapes."""
    _state: _TestMiscInferShapesState = {}
    _test_misc_infer_shapes_part_1(_state)
    _test_misc_infer_shapes_part_2(_state)
    _test_misc_infer_shapes_part_3(_state)
    _test_misc_infer_shapes_part_4(_state)
    assert len(_state) > 0


def test_misc_dispatch() -> None:
    """Test miscellaneous dispatch functions."""

    def safe_call(func: BackendArg, *args: BackendArg, **kwargs: BackendArg) -> None:
        """Test function safe_call.

        Args:
            func: Test parameter `func`.
            args: Test parameter `args`.
            kwargs: Test parameter `kwargs`.
        """
        with pytest.raises(RuntimeError):
            require_callable(func)(*args, **kwargs)

    safe_call(percentile, Dummy((2,)), 0.5)
    safe_call(quantile, Dummy((2,)), 0.5)
    safe_call(flatnonzero, Dummy((2,)))
    safe_call(nonzero, Dummy((2,)))
    safe_call(ravel_multi_index, Dummy((2,)))
    safe_call(lexsort, Dummy((2,)))
    safe_call(searchsorted, Dummy((2,)), Dummy((2,)))
    safe_call(sort_complex, Dummy((2,)))
    safe_call(unique, Dummy((2,)))


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
    from ml_switcheroo_compiler.ops.registry import _REGISTRY

    sm_classes = {cls.__name__: cls for cls in set(_REGISTRY.values()) if cls.__name__ in sm.__all__}
    inferred_count = 0
    for op_name in ops:
        if op_name in sm_classes:
            cls = sm_classes[op_name]
            inst = cls()
            s0 = inst.dispatch_infer_shape()
            assert s0 is not None
            inferred_count += 1

            class DummyShape(MockTestStub):
                """Test class DummyShape."""

                shape = (1,)

            s1 = inst.dispatch_infer_shape(DummyShape())
            assert s1 is not None
            if op_name in ("Intersect1d", "Setdiff1d", "Setxor1d", "Union1d", "Searchsorted", "Compress", "Extract"):
                s2 = inst.dispatch_infer_shape(DummyShape(), DummyShape())
                assert s2 is not None
    assert inferred_count == 11
