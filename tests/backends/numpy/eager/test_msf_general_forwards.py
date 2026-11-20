# ruff: noqa: C901, PLR0912, PLR0915, F841
"""Tests for NumPy eager general math forwarding, debug NaN/Inf guards, and dot_general."""

from __future__ import annotations

import typing

import numpy as np
import pytest

import tests.backends.numpy.eager._math_advanced_test_ns as _mod_raw
from ml_switcheroo_compiler.backends.eager_registry import BackendArg
from ml_switcheroo_compiler.backends.numpy.eager import ensure_ndarray
from tests.backends.numpy.eager._msf_forwards_stub import _SimpleForwardsDummyBackend
from tests.testing_utils import (
    EagerBoundCallable,
    EagerTensorProtocol,
    MockTestStub,
    coerce_operand_list,
    extract_test_shape,
    require_callable,
)

if typing.TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import EagerTensorProtocol

mod = _mod_raw


class _TestMathGeneralSimpleForwardsState(typing.TypedDict, total=False):
    """Typed shared state for `_test_math_general_simple_forwards` split helpers."""

    DummyBackend: type
    _np_apply_over_axes: EagerBoundCallable
    _np_array_repr_: EagerBoundCallable
    _np_array_str_: EagerBoundCallable
    _np_atleast_1d: EagerBoundCallable
    _np_atleast_2d: EagerBoundCallable
    _np_atleast_3d: EagerBoundCallable
    _np_average: EagerBoundCallable
    _np_betainc: EagerBoundCallable
    _np_block: EagerBoundCallable
    _np_callable: EagerBoundCallable
    _np_clip: EagerBoundCallable
    _np_corrcoef: EagerBoundCallable
    _np_cov: EagerBoundCallable
    _np_debuginfs: EagerBoundCallable
    _np_debugnans: EagerBoundCallable
    _np_descriptive_2: EagerBoundCallable
    _np_diff_: EagerBoundCallable
    _np_digitize_: EagerBoundCallable
    _np_dotgeneral: EagerBoundCallable
    _np_ediff1d_: EagerBoundCallable
    _np_einsum_path_: EagerBoundCallable
    _np_fabs_: EagerBoundCallable
    _np_interp_: EagerBoundCallable
    _np_iterable_: EagerBoundCallable
    _np_ix__: EagerBoundCallable
    _np_key: EagerBoundCallable
    _np_kron_: EagerBoundCallable
    _np_load_: EagerBoundCallable
    _np_modf_: EagerBoundCallable
    _np_one_hot: EagerBoundCallable
    _np_raggeddot: EagerBoundCallable
    _np_ravel_multi_index_: EagerBoundCallable
    _np_rem_2: EagerBoundCallable
    _np_serialize_tensor: EagerBoundCallable
    _np_serialize_tensor_camel: EagerBoundCallable
    _np_tensor: EagerBoundCallable
    _np_trapz: EagerBoundCallable
    _np_trapz_: EagerBoundCallable
    _np_truncate_div: EagerBoundCallable
    _np_truncate_mod: EagerBoundCallable
    _np_uint: EagerBoundCallable
    _np_uint8: EagerBoundCallable
    _np_unravelindex: EagerBoundCallable
    _np_unwrap: EagerBoundCallable
    _np_vectorize: EagerBoundCallable
    db: EagerTensorProtocol


def _test_math_general_simple_forwards_sub_1(_state: _TestMathGeneralSimpleForwardsState) -> None:
    """Execute sub-part 1 of test_math_general_simple_forwards.

    Args:
        _state: Shared test state dictionary.
    """
    from ml_switcheroo_compiler.backends.eager.reference_math.math_general import (
        _np_apply_over_axes,
        _np_array_repr_,
        _np_array_str_,
        _np_atleast_1d,
        _np_atleast_2d,
        _np_atleast_3d,
        _np_average,
        _np_block,
        _np_callable,
        _np_clip,
        _np_corrcoef,
        _np_cov,
        _np_debuginfs,
        _np_debugnans,
        _np_descriptive_2,
        _np_diff_,
        _np_digitize_,
        _np_dotgeneral,
        _np_ediff1d_,
        _np_fabs_,
        _np_interp_,
        _np_iterable_,
        _np_ix__,
        _np_key,
        _np_kron_,
        _np_load_,
        _np_modf_,
        _np_one_hot,
        _np_raggeddot,
        _np_ravel_multi_index_,
        _np_rem_2,
        _np_serialize_tensor,
        _np_serialize_tensor_camel,
        _np_tensor,
        _np_trapz,
        _np_trapz_,
        _np_truncate_div,
        _np_truncate_mod,
        _np_uint,
        _np_uint8,
        _np_unravelindex,
        _np_unwrap,
        _np_vectorize,
    )
    from ml_switcheroo_compiler.backends.numpy.eager.scipy_special_kernels import _np_betainc, _np_einsum_path_

    _state.update(
        {
            "_np_apply_over_axes": require_callable(_np_apply_over_axes),
            "_np_array_repr_": require_callable(_np_array_repr_),
            "_np_array_str_": require_callable(_np_array_str_),
            "_np_atleast_1d": require_callable(_np_atleast_1d),
        }
    )
    _state.update(
        {
            "_np_atleast_2d": require_callable(_np_atleast_2d),
            "_np_atleast_3d": require_callable(_np_atleast_3d),
            "_np_average": require_callable(_np_average),
            "_np_betainc": require_callable(_np_betainc),
        }
    )
    _state.update(
        {
            "_np_block": require_callable(_np_block),
            "_np_callable": require_callable(_np_callable),
            "_np_clip": require_callable(_np_clip),
            "_np_corrcoef": require_callable(_np_corrcoef),
        }
    )
    _state.update(
        {
            "_np_cov": require_callable(_np_cov),
            "_np_debuginfs": require_callable(_np_debuginfs),
            "_np_debugnans": require_callable(_np_debugnans),
            "_np_descriptive_2": require_callable(_np_descriptive_2),
        }
    )
    _state.update(
        {
            "_np_diff_": require_callable(_np_diff_),
            "_np_digitize_": require_callable(_np_digitize_),
            "_np_dotgeneral": require_callable(_np_dotgeneral),
            "_np_ediff1d_": require_callable(_np_ediff1d_),
        }
    )
    _state.update(
        {
            "_np_einsum_path_": require_callable(_np_einsum_path_),
            "_np_fabs_": require_callable(_np_fabs_),
            "_np_interp_": require_callable(_np_interp_),
            "_np_iterable_": require_callable(_np_iterable_),
        }
    )
    _state.update(
        {
            "_np_ix__": require_callable(_np_ix__),
            "_np_key": require_callable(_np_key),
            "_np_kron_": require_callable(_np_kron_),
            "_np_load_": require_callable(_np_load_),
        }
    )
    _state.update(
        {
            "_np_modf_": require_callable(_np_modf_),
            "_np_one_hot": require_callable(_np_one_hot),
            "_np_raggeddot": require_callable(_np_raggeddot),
            "_np_ravel_multi_index_": require_callable(_np_ravel_multi_index_),
        }
    )
    _state.update(
        {
            "_np_rem_2": require_callable(_np_rem_2),
            "_np_serialize_tensor": require_callable(_np_serialize_tensor),
            "_np_serialize_tensor_camel": require_callable(_np_serialize_tensor_camel),
            "_np_tensor": require_callable(_np_tensor),
        }
    )
    _state.update(
        {
            "_np_trapz": require_callable(_np_trapz),
            "_np_trapz_": require_callable(_np_trapz_),
            "_np_truncate_div": require_callable(_np_truncate_div),
            "_np_truncate_mod": require_callable(_np_truncate_mod),
        }
    )
    _state.update(
        {
            "_np_uint": require_callable(_np_uint),
            "_np_uint8": require_callable(_np_uint8),
            "_np_unravelindex": require_callable(_np_unravelindex),
            "_np_unwrap": require_callable(_np_unwrap),
        }
    )
    _state.update({"_np_vectorize": require_callable(_np_vectorize)})


def _test_math_general_simple_forwards_sub_2(_state: _TestMathGeneralSimpleForwardsState) -> None:
    """Execute sub-part 2 of test_math_general_simple_forwards.

    Args:
        _state: Shared test state dictionary.
    """
    _state.update({"DummyBackend": _SimpleForwardsDummyBackend})


def _test_math_general_simple_forwards_sub_3(_state: _TestMathGeneralSimpleForwardsState) -> None:
    """Execute sub-part 3 of test_math_general_simple_forwards.

    Args:
        _state: Shared test state dictionary.
    """
    (DummyBackend, _np_apply_over_axes, _np_array_repr_, _np_array_str_) = (
        _state["DummyBackend"],
        _state["_np_apply_over_axes"],
        _state["_np_array_repr_"],
        _state["_np_array_str_"],
    )
    _np_atleast_1d = _state["_np_atleast_1d"]
    db = DummyBackend()
    with pytest.raises(AttributeError):
        assert _np_apply_over_axes(db) == 0
    with pytest.raises(AttributeError):
        assert _np_array_repr_(db) == 1
    with pytest.raises(AttributeError):
        assert _np_array_str_(db) == 2
    with pytest.raises(AttributeError):
        assert _np_atleast_1d(db) == 3
    _state.update({"db": db})
    _state.update(
        {
            "DummyBackend": DummyBackend,
            "_np_apply_over_axes": require_callable(_np_apply_over_axes),
            "_np_array_repr_": require_callable(_np_array_repr_),
        }
    )
    _state.update(
        {"_np_array_str_": require_callable(_np_array_str_), "_np_atleast_1d": require_callable(_np_atleast_1d)}
    )


def _test_math_general_simple_forwards_sub_4(_state: _TestMathGeneralSimpleForwardsState) -> None:
    """Execute sub-part 4 of test_math_general_simple_forwards.

    Args:
        _state: Shared test state dictionary.
    """
    (_np_atleast_2d, _np_atleast_3d, _np_average, _np_betainc) = (
        _state["_np_atleast_2d"],
        _state["_np_atleast_3d"],
        _state["_np_average"],
        _state["_np_betainc"],
    )
    db = _state["db"]
    with pytest.raises(AttributeError):
        assert _np_atleast_2d(db) == 4
    with pytest.raises(AttributeError):
        assert _np_atleast_3d(db) == 5
    assert _np_average(db) == 6
    with pytest.raises(AttributeError):
        assert _np_betainc(db) == 7
    _state.update(
        {
            "_np_atleast_2d": require_callable(_np_atleast_2d),
            "_np_atleast_3d": require_callable(_np_atleast_3d),
            "_np_average": require_callable(_np_average),
        }
    )
    _state.update({"_np_betainc": require_callable(_np_betainc), "db": db})


def _test_math_general_simple_forwards_sub_5(_state: _TestMathGeneralSimpleForwardsState) -> None:
    """Execute sub-part 5 of test_math_general_simple_forwards.

    Args:
        _state: Shared test state dictionary.
    """
    (_np_block, _np_callable, _np_clip, _np_corrcoef) = (
        _state["_np_block"],
        _state["_np_callable"],
        _state["_np_clip"],
        _state["_np_corrcoef"],
    )
    db = _state["db"]
    assert _np_block(db) == 8
    assert _np_callable(db) != 9
    assert _np_clip(db) != 10
    assert _np_corrcoef(db) == 11
    _state.update(
        {
            "_np_block": require_callable(_np_block),
            "_np_callable": require_callable(_np_callable),
            "_np_clip": require_callable(_np_clip),
        }
    )
    _state.update({"_np_corrcoef": require_callable(_np_corrcoef), "db": db})


def _test_math_general_simple_forwards_sub_6(_state: _TestMathGeneralSimpleForwardsState) -> None:
    """Execute sub-part 6 of test_math_general_simple_forwards.

    Args:
        _state: Shared test state dictionary.
    """
    (_np_cov, _np_debuginfs, _np_debugnans, _np_descriptive_2) = (
        _state["_np_cov"],
        _state["_np_debuginfs"],
        _state["_np_debugnans"],
        _state["_np_descriptive_2"],
    )
    db = _state["db"]
    assert _np_cov(db) == 12
    with pytest.raises(IndexError):
        assert _np_debuginfs(db) == 13
    with pytest.raises(IndexError):
        assert _np_debugnans(db) == 14
    assert _np_descriptive_2(db) == 15
    _state.update(
        {
            "_np_cov": require_callable(_np_cov),
            "_np_debuginfs": require_callable(_np_debuginfs),
            "_np_debugnans": require_callable(_np_debugnans),
        }
    )
    _state.update({"_np_descriptive_2": require_callable(_np_descriptive_2), "db": db})


def _test_math_general_simple_forwards_sub_7(_state: _TestMathGeneralSimpleForwardsState) -> None:
    """Execute sub-part 7 of test_math_general_simple_forwards.

    Args:
        _state: Shared test state dictionary.
    """
    (_np_descriptive_2, _np_diff_, _np_digitize_, _np_dotgeneral) = (
        _state["_np_descriptive_2"],
        _state["_np_diff_"],
        _state["_np_digitize_"],
        _state["_np_dotgeneral"],
    )
    db = _state["db"]
    assert _np_descriptive_2(db) != 16
    assert _np_diff_(db) == 17
    assert _np_digitize_(db) == 18
    with pytest.raises(IndexError):
        assert _np_dotgeneral(db) == 19
    _state.update(
        {
            "_np_descriptive_2": require_callable(_np_descriptive_2),
            "_np_diff_": require_callable(_np_diff_),
            "_np_digitize_": require_callable(_np_digitize_),
        }
    )
    _state.update({"_np_dotgeneral": require_callable(_np_dotgeneral), "db": db})


def _test_math_general_simple_forwards_sub_8(_state: _TestMathGeneralSimpleForwardsState) -> None:
    """Execute sub-part 8 of test_math_general_simple_forwards.

    Args:
        _state: Shared test state dictionary.
    """
    (_np_ediff1d_, _np_einsum_path_, _np_fabs_, _np_interp_) = (
        _state["_np_ediff1d_"],
        _state["_np_einsum_path_"],
        _state["_np_fabs_"],
        _state["_np_interp_"],
    )
    db = _state["db"]
    assert _np_ediff1d_(db) == 20
    with pytest.raises(AttributeError):
        assert _np_einsum_path_(db) == 21
    assert _np_fabs_(db) == 22
    assert _np_interp_(db) == 23
    _state.update(
        {
            "_np_ediff1d_": require_callable(_np_ediff1d_),
            "_np_einsum_path_": require_callable(_np_einsum_path_),
            "_np_fabs_": require_callable(_np_fabs_),
        }
    )
    _state.update({"_np_interp_": require_callable(_np_interp_), "db": db})


def _test_math_general_simple_forwards_sub_9(_state: _TestMathGeneralSimpleForwardsState) -> None:
    """Execute sub-part 9 of test_math_general_simple_forwards.

    Args:
        _state: Shared test state dictionary.
    """
    (_np_iterable_, _np_ix__, _np_key, _np_kron_) = (
        _state["_np_iterable_"],
        _state["_np_ix__"],
        _state["_np_key"],
        _state["_np_kron_"],
    )
    db = _state["db"]
    assert _np_iterable_(db) == 24
    with pytest.raises(AttributeError):
        assert _np_ix__(db) == 25
    with pytest.raises(ValueError):
        assert _np_key(db) == 26
    assert _np_kron_(db) == 27
    _state.update(
        {
            "_np_iterable_": require_callable(_np_iterable_),
            "_np_ix__": require_callable(_np_ix__),
            "_np_key": require_callable(_np_key),
        }
    )
    _state.update({"_np_kron_": require_callable(_np_kron_), "db": db})


def _test_math_general_simple_forwards_sub_10(_state: _TestMathGeneralSimpleForwardsState) -> None:
    """Execute sub-part 10 of test_math_general_simple_forwards.

    Args:
        _state: Shared test state dictionary.
    """
    (_np_load_, _np_modf_, _np_one_hot, _np_raggeddot) = (
        _state["_np_load_"],
        _state["_np_modf_"],
        _state["_np_one_hot"],
        _state["_np_raggeddot"],
    )
    db = _state["db"]
    assert _np_load_(db) == 28
    assert _np_modf_(db) == 29
    assert _np_one_hot(db) != 30
    with pytest.raises(AttributeError):
        assert _np_raggeddot(db) == 31
    _state.update(
        {
            "_np_load_": require_callable(_np_load_),
            "_np_modf_": require_callable(_np_modf_),
            "_np_one_hot": require_callable(_np_one_hot),
        }
    )
    _state.update({"_np_raggeddot": require_callable(_np_raggeddot), "db": db})


def _test_math_general_simple_forwards_sub_11(_state: _TestMathGeneralSimpleForwardsState) -> None:
    """Execute sub-part 11 of test_math_general_simple_forwards.

    Args:
        _state: Shared test state dictionary.
    """
    (_np_ravel_multi_index_, _np_rem_2, db) = (_state["_np_ravel_multi_index_"], _state["_np_rem_2"], _state["db"])
    with pytest.raises(AttributeError):
        assert _np_ravel_multi_index_(db) == 32
    assert _np_rem_2(db) == 33
    assert _np_rem_2(db) != 34
    assert _np_rem_2(db) != 35
    _state.update(
        {
            "_np_ravel_multi_index_": require_callable(_np_ravel_multi_index_),
            "_np_rem_2": require_callable(_np_rem_2),
            "db": db,
        }
    )


def _test_math_general_simple_forwards_sub_12(_state: _TestMathGeneralSimpleForwardsState) -> None:
    """Execute sub-part 12 of test_math_general_simple_forwards.

    Args:
        _state: Shared test state dictionary.
    """
    (_np_serialize_tensor, _np_serialize_tensor_camel, _np_tensor, _np_trapz) = (
        _state["_np_serialize_tensor"],
        _state["_np_serialize_tensor_camel"],
        _state["_np_tensor"],
        _state["_np_trapz"],
    )
    db = _state["db"]
    assert _np_serialize_tensor(db) != 36
    assert _np_serialize_tensor_camel(db) != 37
    with pytest.raises(AttributeError):
        assert _np_tensor(db) == 38
    with pytest.raises(IndexError):
        assert _np_trapz(db) == 39
    _state.update(
        {
            "_np_serialize_tensor": require_callable(_np_serialize_tensor),
            "_np_serialize_tensor_camel": require_callable(_np_serialize_tensor_camel),
            "_np_tensor": require_callable(_np_tensor),
        }
    )
    _state.update({"_np_trapz": require_callable(_np_trapz), "db": db})


def _test_math_general_simple_forwards_sub_13(_state: _TestMathGeneralSimpleForwardsState) -> None:
    """Execute sub-part 13 of test_math_general_simple_forwards.

    Args:
        _state: Shared test state dictionary.
    """
    (_np_trapz_, _np_truncate_div, _np_truncate_mod, _np_uint) = (
        _state["_np_trapz_"],
        _state["_np_truncate_div"],
        _state["_np_truncate_mod"],
        _state["_np_uint"],
    )
    db = _state["db"]
    assert _np_trapz_(db) != 40
    with pytest.raises(IndexError):
        assert _np_truncate_div(db) == 41
    with pytest.raises(IndexError):
        assert _np_truncate_mod(db) == 42
    with pytest.raises(IndexError):
        assert _np_uint(db) == 43
    _state.update(
        {
            "_np_trapz_": require_callable(_np_trapz_),
            "_np_truncate_div": require_callable(_np_truncate_div),
            "_np_truncate_mod": require_callable(_np_truncate_mod),
        }
    )
    _state.update({"_np_uint": require_callable(_np_uint), "db": db})


def _test_math_general_simple_forwards_sub_14(_state: _TestMathGeneralSimpleForwardsState) -> None:
    """Execute sub-part 14 of test_math_general_simple_forwards.

    Args:
        _state: Shared test state dictionary.
    """
    (_np_uint8, _np_unravelindex, _np_unwrap, _np_vectorize) = (
        _state["_np_uint8"],
        _state["_np_unravelindex"],
        _state["_np_unwrap"],
        _state["_np_vectorize"],
    )
    db = _state["db"]
    with pytest.raises(IndexError):
        assert _np_uint8(db) == 44
    with pytest.raises(IndexError):
        assert _np_unravelindex(db) == 45
    with pytest.raises(IndexError):
        assert _np_unwrap(db) == 46
    with pytest.raises(IndexError):
        assert _np_vectorize(db) == 47
    _state.update(
        {
            "_np_uint8": require_callable(_np_uint8),
            "_np_unravelindex": require_callable(_np_unravelindex),
            "_np_unwrap": require_callable(_np_unwrap),
        }
    )
    _state.update({"_np_vectorize": require_callable(_np_vectorize), "db": db})


def test_math_general_simple_forwards() -> None:
    """Test function test_math_general_simple_forwards."""
    _state: _TestMathGeneralSimpleForwardsState = {}
    for _sub_fn in (
        _test_math_general_simple_forwards_sub_1,
        _test_math_general_simple_forwards_sub_2,
        _test_math_general_simple_forwards_sub_3,
        _test_math_general_simple_forwards_sub_4,
        _test_math_general_simple_forwards_sub_5,
        _test_math_general_simple_forwards_sub_6,
        _test_math_general_simple_forwards_sub_7,
        _test_math_general_simple_forwards_sub_8,
        _test_math_general_simple_forwards_sub_9,
        _test_math_general_simple_forwards_sub_10,
        _test_math_general_simple_forwards_sub_11,
        _test_math_general_simple_forwards_sub_12,
        _test_math_general_simple_forwards_sub_13,
        _test_math_general_simple_forwards_sub_14,
    ):
        _sub_fn(_state)
    assert len(_state) > 0


def test_math_general_missing_forwards() -> None:
    """Test function test_math_general_missing_forwards."""
    from ml_switcheroo_compiler.backends.eager.reference_math.math_general import (
        _np_clip,
        _np_dotgeneral,
        _np_key,
        _np_one_hot,
        _np_raggeddot,
        _np_serialize_tensor,
        _np_serialize_tensor_camel,
        _np_truncate_div,
        _np_truncate_mod,
    )

    class DummyBackend(MockTestStub):
        """Test class DummyBackend."""

        def matmul(self, *args: BackendArg, **kwargs: BackendArg) -> BackendArg:
            """Test function matmul.

            Args:
                args: Test parameter `args`.
                kwargs: Test parameter `kwargs`.

            Returns:
                BackendArg: Computed output value from `matmul`.
            """
            return 1

    db = DummyBackend()
    assert _np_truncate_div(db, 5.0, 2.0) == 2.0
    assert _np_truncate_mod(db, 5.0, 2.0) == 1.0
    assert _np_dotgeneral(db, np.ones((2, 2)), np.ones((2, 2)), dimension_numbers=(((1,), (0,)), ((), ()))) is not None
    assert _np_raggeddot(db, np.ones((2, 2)), np.ones((2, 2))) is not None
    assert _np_serialize_tensor(db, 1) is not None
    assert _np_serialize_tensor_camel(db, 1) is not None
    assert np.array_equal(ensure_ndarray(_np_key(db, 1)), [1, 0])
    assert np.array_equal(ensure_ndarray(_np_clip(db, np.array([5.0]), 1.0, 3.0)), [3.0])
    res = _np_one_hot(db, np.array([0]), 2)
    assert extract_test_shape(res) == (1, 2)


def test_math_general_debuginfs_callable() -> None:
    """Test function test_math_general_debuginfs_callable."""
    from ml_switcheroo_compiler.backends.eager.reference_math.math_general import (
        _np_callable,
        _np_debuginfs,
        _np_debugnans,
        _np_descriptive_2,
        _np_key,
        _np_rem_2,
        _np_tensor,
    )

    assert _np_callable(np, lambda x: x) is True
    assert _np_callable(np) is False
    import pytest

    with pytest.raises(ValueError):
        _np_debuginfs(np, np.array([np.inf]))
    assert np.array_equal(ensure_ndarray(_np_debuginfs(np, np.array([1.0]))), [1.0])
    with pytest.raises(ValueError):
        _np_debugnans(np, np.array([np.nan]))
    assert np.array_equal(ensure_ndarray(_np_debugnans(np, np.array([1.0]))), [1.0])
    assert extract_test_shape(_np_tensor(np)) == (0,)
    assert extract_test_shape(_np_tensor(np, [1, 2])) == (2,)
    res = _np_descriptive_2(np, [1.0, 2.0])
    assert len(coerce_operand_list(res)) == 3
    assert len(coerce_operand_list(_np_descriptive_2(np))) == 3
    assert np.array_equal(ensure_ndarray(_np_key(np)), [0, 0])
    assert _np_rem_2(np) is None


def test_dotgeneral_batches() -> None:
    """Test function test_dotgeneral_batches."""
    from ml_switcheroo_compiler.backends.eager.reference_math.math_general import _np_dotgeneral

    class DummyBackend(MockTestStub):
        """Test class DummyBackend."""

        pass

    assert tuple(
        extract_test_shape(
            _np_dotgeneral(
                DummyBackend(),
                np.ones((2, 2)),
                np.ones((2, 2)),
                dimension_numbers=(((1,), (1,)), ((0,), (0,))),
            )
        )
    ) == (2,)


def test_math_general_remaining() -> None:
    """Test function test_math_general_remaining."""
    from ml_switcheroo_compiler.backends.eager.reference_math.math_general import (
        _np_clip,
        _np_key,
        _np_one_hot,
        _np_rem_2,
        _np_serialize_tensor_camel,
    )
    from ml_switcheroo_compiler.backends.numpy.eager.scipy_special_kernels import _get_sc

    assert _np_rem_2(np, 5, 2) == 1
    res = _np_key(np, 1)
    assert res is not None
    assert _get_sc() is not None
    assert _np_clip(np, np.array([5.0]), 1.0, 3.0) is not None
    assert _np_clip(np) is None
    res2 = _np_one_hot(np, np.array([0]), 2)
    assert res2 is not None
    assert _np_one_hot(np) is None
    assert _np_serialize_tensor_camel(np, np.array([1])) is not None

    class DummyBackendRem(MockTestStub):
        """Test class DummyBackendRem."""

        pass

    assert _np_rem_2(DummyBackendRem(), 5, 2) == 1


def test_math_general_final() -> None:
    """Test function test_math_general_final."""
    from unittest.mock import patch

    from ml_switcheroo_compiler.backends.eager.reference_math.math_general import (
        _np_one_hot,
        _np_serialize_tensor_camel,
    )
    from ml_switcheroo_compiler.backends.numpy.eager.scipy_special_kernels import _get_sc

    with patch("importlib.import_module", side_effect=ImportError("boom")):
        assert _get_sc() is None
    res = _np_one_hot(np, np.array([0]), 2, axis=0)
    assert extract_test_shape(res) == (2, 1)
    with patch("pickle.dumps", side_effect=RuntimeError("boom")):
        res2 = _np_serialize_tensor_camel(np, np.array([1]))
        assert res2 is not None
