"""Tests for test_core_math_internal."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, cast

import numpy as np
import pytest

from ml_switcheroo_compiler.backends.eager.core_math_ops.math_internal import (
    _copysign,
    _getprintoptions,
    _np_tensorarrayread,
    _np_tensorarraywrite,
    _np_topk,
)
from ml_switcheroo_compiler.backends.eager_registry import global_eager_registry

if TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import EagerCallableProtocol, EagerTensorProtocol, EagerValue


class _EmptyBackend:
    """Mock backend without any attributes."""

    pass


def _test_math_internal_all_branches_part_1(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 1 of test_math_internal_all_branches.

    Args:
        _state: Shared test state dictionary.
    """
    empty_backend = _EmptyBackend()

    # 1. _copysign
    # Case A: backend has copysign
    class BackendCopysign:
        """Backend with copysign."""

        def copysign(self, x: float, y: float) -> float:
            """Copysign.

            Args:
                x: Test parameter `x`.
                y: Test parameter `y`.
            """
            return math.copysign(x, y)

    assert _copysign(BackendCopysign(), -5.0, 1.0) == 5.0

    # Case B: backend lacks copysign -> fallback: abs(x) * sign(y)
    class BackendAbsSign:
        """Backend with abs and sign."""

        def abs(self, val: EagerValue) -> EagerValue:
            """Abs.

            Args:
                val: Test parameter `val`.
            """
            return cast("EagerValue", cast("EagerCallableProtocol", np.abs)(val))

        def sign(self, val: EagerValue) -> EagerValue:
            """Sign.

            Args:
                val: Test parameter `val`.
            """
            return cast("EagerValue", cast("EagerCallableProtocol", np.sign)(val))

    res_cs = _copysign(BackendAbsSign(), np.array([5.0, -3.0]), np.array([-1.0, 1.0]))
    assert cast("EagerCallableProtocol", list)(res_cs) == [-5.0, 3.0]

    # 2. _getprintoptions
    class BackendPrintOptions:
        """Backend with get_printoptions."""

        def get_printoptions(self, *args: EagerValue, **kwargs: EagerValue) -> dict[str, EagerValue]:
            """Get print options.

            Args:
                args: Test parameter `args`.
                kwargs: Test parameter `kwargs`.
            """
            return {"precision": 4}

    getattr(_state, "update")({"BackendAbsSign": BackendAbsSign, "BackendCopysign": BackendCopysign, "BackendPrintOptions": BackendPrintOptions, "empty_backend": empty_backend})
    getattr(_state, "update")({"res_cs": res_cs})
    _state.update(locals())


def _test_math_internal_all_branches_part_2(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 2 of test_math_internal_all_branches.

    Args:
        _state: Shared test state dictionary.
    """
    BackendPrintOptions, empty_backend = _state["BackendPrintOptions"], _state["empty_backend"]
    assert _getprintoptions(BackendPrintOptions()) == {"precision": 4}

    # 3. _np_tensorarrayread
    # Case A: no args or args[0] is None
    assert _np_tensorarrayread(empty_backend) == 0
    assert _np_tensorarrayread(empty_backend, None) == 0
    sample_list = [10, 20, 30]
    assert _np_tensorarrayread(empty_backend, sample_list, 1) == 20
    assert _np_tensorarrayread(empty_backend, sample_list, index=2) == 30

    with pytest.raises(IndexError, match="out of bounds"):
        _np_tensorarrayread(empty_backend, sample_list, -1)
    with pytest.raises(IndexError, match="out of bounds"):
        _np_tensorarrayread(empty_backend, sample_list, 5)

    # Case C: arr has shape and __getitem__ (e.g. ndarray)
    sample_arr = np.array([100, 200, 300])
    assert _np_tensorarrayread(empty_backend, sample_arr, 0) == 100
    with pytest.raises(IndexError, match="out of bounds for axis 0"):
        _np_tensorarrayread(empty_backend, sample_arr, -1)
    with pytest.raises(IndexError, match="out of bounds for axis 0"):
        _np_tensorarrayread(empty_backend, sample_arr, 10)

    # Case D: arr is a dict
    sample_dict = {0: "a", 1: "b"}
    assert _np_tensorarrayread(empty_backend, sample_dict, 0) == "a"
    with pytest.raises(IndexError, match="not found in sparse handle"):
        _np_tensorarrayread(empty_backend, sample_dict, 2)

    # Case E: arr is neither list, array, nor dict
    assert _np_tensorarrayread(empty_backend, 12345, 0) == 0

    # 4. _np_tensorarraywrite
    # Case A: no args
    assert _np_tensorarraywrite(empty_backend) == 0
    getattr(_state, "update")({"sample_arr": sample_arr, "sample_dict": sample_dict, "sample_list": sample_list})
    _state.update(locals())


def _test_math_internal_all_branches_part_3(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 3 of test_math_internal_all_branches.

    Args:
        _state: Shared test state dictionary.
    """
    empty_backend = _state["empty_backend"]
    with pytest.raises(IndexError, match="Negative TensorArray index"):
        _np_tensorarraywrite(empty_backend, [], -1, 10)

    # Case C: arr is a list (index within length)
    w_list1 = _np_tensorarraywrite(empty_backend, [1, 2, 3], 1, 99)
    assert w_list1 == [1, 99, 3]

    # Case D: arr is a list (index exceeds length -> padded with None)
    w_list2 = _np_tensorarraywrite(empty_backend, [1], 3, 55)
    assert w_list2 == [1, None, None, 55]

    # Case E: arr is None -> initialized as empty list and expanded
    w_list_none = _np_tensorarraywrite(empty_backend, None, 2, 42)
    assert w_list_none == [None, None, 42]

    # Case F: arr is numpy.ndarray (index < shape[0])
    np_arr1 = np.array([1, 2, 3])
    w_arr1 = _np_tensorarraywrite(empty_backend, np_arr1, 1, 99)
    assert cast("EagerCallableProtocol", list)(w_arr1) == [1, 99, 3]

    # Case G: arr is numpy.ndarray (index >= shape[0] -> padded with zeros)
    np_arr2 = np.array([1, 2])
    w_arr2 = _np_tensorarraywrite(empty_backend, np_arr2, 4, 100, index=4)
    assert cast("EagerCallableProtocol", list)(w_arr2) == [1, 2, 0, 0, 100]

    # Case H: arr is dict
    w_dict = _np_tensorarraywrite(empty_backend, {0: "a"}, 1, "b")
    assert w_dict == {0: "a", 1: "b"}

    # Case I: arr is other type
    assert _np_tensorarraywrite(empty_backend, 12345, 0, "val") == 0

    # 5. _np_topk
    # Case A: arr is list/ndarray
    vals, indices = cast("EagerCallableProtocol", _np_topk)(empty_backend, [10, 50, 20, 40, 30], 2)
    assert cast("EagerCallableProtocol", list)(vals) == [50, 40]
    assert cast("EagerCallableProtocol", list)(indices) == [1, 3]

    # Default k=1
    vals_def, indices_def = cast("EagerCallableProtocol", _np_topk)(empty_backend, [10, 50, 20])
    assert cast("EagerCallableProtocol", list)(vals_def) == [50]
    getattr(_state, "update")({"indices": indices, "indices_def": indices_def, "np_arr1": np_arr1, "np_arr2": np_arr2})
    getattr(_state, "update")({"vals": vals, "vals_def": vals_def, "w_arr1": w_arr1, "w_arr2": w_arr2})
    getattr(_state, "update")({"w_dict": w_dict, "w_list1": w_list1, "w_list2": w_list2, "w_list_none": w_list_none})
    _state.update(locals())


def _test_math_internal_all_branches_part_4(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 4 of test_math_internal_all_branches.

    Args:
        _state: Shared test state dictionary.
    """
    empty_backend, indices_def = _state["empty_backend"], _state["indices_def"]
    assert list(indices_def) == [1]

    # Case B: arr is not list or ndarray
    v_fb, i_fb = cast("EagerCallableProtocol", _np_topk)(empty_backend, "not_an_array")
    assert v_fb == [0]
    assert i_fb == [0]

    # Registration checks
    for op_name in ["Copysign", "GetPrintoptions", "NpTensorarrayread", "NpTensorarraywrite", "NpTopk"]:
        assert global_eager_registry.get(op_name) is not None
    getattr(_state, "update")({"i_fb": i_fb, "v_fb": v_fb})


def test_math_internal_all_branches() -> None:
    """Test 100% lines and branches of math_internal.py.

    Returns:
        None
    """
    _state: dict[str, EagerTensorProtocol] = {}
    _test_math_internal_all_branches_part_1(_state)
    _test_math_internal_all_branches_part_2(_state)
    _test_math_internal_all_branches_part_3(_state)
    _test_math_internal_all_branches_part_4(_state)
    assert len(_state) > 0
