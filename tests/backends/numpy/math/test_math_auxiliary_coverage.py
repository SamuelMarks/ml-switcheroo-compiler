"""Tests for test_math_auxiliary_coverage."""

from __future__ import annotations

import types
import typing

import numpy as np
import pytest

from ml_switcheroo_compiler.backends.eager_registry import EagerValue, numpy_eager_registry
from ml_switcheroo_compiler.backends.numpy.eager import math_auxiliary as aux_module
from ml_switcheroo_compiler.backends.protocols import BackendModuleProtocol
from ml_switcheroo_compiler.core.dtype import DType

if typing.TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import EagerCallableProtocol, EagerTensorProtocol


def _test_math_auxiliary_ops_part_1(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 1 of test_math_auxiliary_ops.

    Args:
        _state: Shared test state dictionary.
    """
    monkeypatch = _state["monkeypatch"]
    c_arr = aux_module._np_constant_of_shape(np, (2, 2), value=3.0)
    assert typing.cast("EagerCallableProtocol", np.allclose)(c_arr, typing.cast("EagerCallableProtocol", np.full)((2, 2), 3.0))
    reg_cos = numpy_eager_registry.get("ConstantOfShape")
    assert typing.cast("EagerCallableProtocol", np.allclose)(typing.cast("EagerCallableProtocol", reg_cos)(np, (2, 2), value=3.0), typing.cast("EagerCallableProtocol", np.full)((2, 2), 3.0))

    # ReduceWindow
    monkeypatch.setattr(aux_module, "_reduce_window", lambda *args, **kwargs: np.array([42.0]))
    rw_res = aux_module._np_reduce_window(np, np.ones((4, 4)))
    assert typing.cast("EagerTensorProtocol", getattr(rw_res, "__getitem__")(0)) == 42.0
    reg_rw = numpy_eager_registry.get("ReduceWindow")
    assert reg_rw is not None

    # TestEagerOp
    assert typing.cast("EagerCallableProtocol", np.allclose)(typing.cast("EagerCallableProtocol", aux_module._np_test_eager_op)(np), [1.0, 2.0, 3.0])
    assert typing.cast("EagerCallableProtocol", np.allclose)(typing.cast("EagerCallableProtocol", numpy_eager_registry.get)("TestEagerOp")(np), [1.0, 2.0, 3.0])

    # Unknown
    assert aux_module._np_unknown(np) == 0.0
    assert typing.cast("EagerCallableProtocol", numpy_eager_registry.get)("Unknown")(np) == 0.0

    # Rand with dtype branches
    r_bf = aux_module._np_rand(np, 2, 3, dtype="bfloat16")
    assert typing.cast("EagerTensorProtocol", getattr(r_bf, "dtype")) == np.float32

    r_f8 = aux_module._np_rand(np, 2, 3, dtype="float8_e4m3")
    assert typing.cast("EagerTensorProtocol", getattr(r_f8, "dtype")) == np.float32

    r_i4 = aux_module._np_rand(np, 2, 3, dtype="int4")
    assert typing.cast("EagerTensorProtocol", getattr(r_i4, "dtype")) == np.int8

    r_std = aux_module._np_rand(np, 2, 3, dtype=np.float64)
    assert typing.cast("EagerTensorProtocol", getattr(r_std, "dtype")) == np.float64

    mock_rand_mod = types.SimpleNamespace(
        random=types.SimpleNamespace(rand=lambda *a: np.array([0.5])),
        array=lambda a: a,
    )
    getattr(_state, "update")({"c_arr": c_arr, "mock_rand_mod": mock_rand_mod, "r_bf": r_bf, "r_f8": r_f8})
    getattr(_state, "update")({"r_i4": r_i4, "r_std": r_std, "reg_cos": reg_cos, "reg_rw": reg_rw})
    getattr(_state, "update")({"rw_res": rw_res})
    _state.update(locals())


def _test_math_auxiliary_ops_part_2(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 2 of test_math_auxiliary_ops.

    Args:
        _state: Shared test state dictionary.
    """
    mock_rand_mod = _state["mock_rand_mod"]
    r_str = aux_module._np_rand(mock_rand_mod, 1, dtype="int64")
    assert typing.cast("EagerTensorProtocol", getattr(r_str, "dtype")) == np.int64

    reg_rand = numpy_eager_registry.get("Rand")
    assert typing.cast("EagerTensorProtocol", getattr(typing.cast("EagerCallableProtocol", reg_rand)(np, 2, 2), "shape")) == (2, 2)

    # IsNonDecreasing branches
    mock_diff_mod = types.SimpleNamespace(
        size=lambda a: np.size(a),
        diff=lambda a: a[1:] - a[:-1],
        all=lambda a: bool(all(x for x in a.flat)),
        array=lambda a: np.array(a),
    )
    assert aux_module._np_is_non_decreasing(typing.cast("BackendModuleProtocol", mock_diff_mod), np.array([5]))
    assert aux_module._np_is_non_decreasing(typing.cast("BackendModuleProtocol", mock_diff_mod), np.array([1, 2, 2, 3]))
    assert not aux_module._np_is_non_decreasing(typing.cast("BackendModuleProtocol", mock_diff_mod), np.array([1, 3, 2]))
    reg_ind = numpy_eager_registry.get("IsNonDecreasing")
    assert typing.cast("EagerCallableProtocol", reg_ind)(mock_diff_mod, typing.cast("EagerCallableProtocol", np.array)([1, 2, 3]))

    # IsStrictlyIncreasing branches
    assert aux_module._np_is_strictly_increasing(typing.cast("BackendModuleProtocol", mock_diff_mod), np.array([5]))
    assert aux_module._np_is_strictly_increasing(typing.cast("BackendModuleProtocol", mock_diff_mod), np.array([1, 2, 3]))
    assert not aux_module._np_is_strictly_increasing(typing.cast("BackendModuleProtocol", mock_diff_mod), np.array([1, 2, 2]))
    reg_isi = numpy_eager_registry.get("IsStrictlyIncreasing")
    assert typing.cast("EagerCallableProtocol", reg_isi)(mock_diff_mod, typing.cast("EagerCallableProtocol", np.array)([1, 2, 3]))

    # L2Normalize
    mat = np.array([[3.0, 4.0]])
    normed = aux_module._np_l2_normalize(np, mat, axis=-1)
    assert typing.cast("EagerCallableProtocol", np.allclose)(normed, [[0.6, 0.8]])
    reg_l2 = numpy_eager_registry.get("L2Normalize")
    assert typing.cast("EagerCallableProtocol", np.allclose)(typing.cast("EagerCallableProtocol", reg_l2)(np, mat, axis=-1), [[0.6, 0.8]])

    # ReduceEuclideanNorm
    assert typing.cast("EagerCallableProtocol", np.isclose)(typing.cast("EagerCallableProtocol", aux_module._np_reduce_euclidean_norm)(np, typing.cast("EagerCallableProtocol", np.array)([3.0, 4.0])), 5.0)
    reg_ren = numpy_eager_registry.get("ReduceEuclideanNorm")
    getattr(_state, "update")({"mat": mat, "mock_diff_mod": mock_diff_mod, "normed": normed, "r_str": r_str})
    getattr(_state, "update")({"reg_ind": reg_ind, "reg_isi": reg_isi, "reg_l2": reg_l2, "reg_rand": reg_rand})
    getattr(_state, "update")({"reg_ren": reg_ren})
    _state.update(locals())


def _test_math_auxiliary_ops_part_3(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 3 of test_math_auxiliary_ops.

    Args:
        _state: Shared test state dictionary.
    """
    reg_ren = _state["reg_ren"]
    assert typing.cast("EagerCallableProtocol", np.isclose)(typing.cast("EagerCallableProtocol", reg_ren)(np, typing.cast("EagerCallableProtocol", np.array)([3.0, 4.0])), 5.0)

    # Clamp
    assert aux_module._clamp(np, 0.0, 5.0, 2.0) == 2.0
    reg_clamp = numpy_eager_registry.get("Clamp")
    assert typing.cast("EagerCallableProtocol", reg_clamp)(np, 0.0, 5.0, 2.0) == 2.0

    class SpaceConfig:
        """Mock SpaceConfig class."""

        def __init__(self) -> None:
            """Initialize mock SpaceConfig."""
            self.num = 4
            self.endpoint = True
            self.base = 10.0
            self.dtype = float
            self.axis = 0

    ls_cfg = aux_module._logspace(np, 1.0, 3.0, typing.cast("EagerValue", SpaceConfig()))
    assert typing.cast("EagerCallableProtocol", len)(ls_cfg) == 4

    ls_normal = aux_module._logspace(np, 1.0, 3.0, num=4)
    assert typing.cast("EagerCallableProtocol", len)(ls_normal) == 4
    reg_ls = numpy_eager_registry.get("Logspace")
    assert typing.cast("EagerCallableProtocol", len)(typing.cast("EagerCallableProtocol", reg_ls)(np, 1.0, 3.0, num=4)) == 4

    # FromBuffer
    buf = b"\x00\x00\x80?\x00\x00\x00@"
    fb = aux_module._np_frombuffer(np, buf, dtype="float32")
    assert typing.cast("EagerCallableProtocol", len)(fb) == 2
    getattr(_state, "update")({"SpaceConfig": SpaceConfig, "buf": buf, "fb": fb, "ls_cfg": ls_cfg})
    getattr(_state, "update")({"ls_normal": ls_normal, "reg_clamp": reg_clamp, "reg_ls": reg_ls})
    _state.update(locals())


def _test_math_auxiliary_ops_part_4(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 4 of test_math_auxiliary_ops.

    Args:
        _state: Shared test state dictionary.
    """
    buf = _state["buf"]
    reg_fb = numpy_eager_registry.get("FromBuffer")
    assert typing.cast("EagerCallableProtocol", len)(typing.cast("EagerCallableProtocol", reg_fb)(np, buf, dtype="float32")) == 2

    # DType branches
    assert isinstance(aux_module._np_dtype_op(np, "float32"), DType)
    assert isinstance(aux_module._np_dtype_op(np, np.array([1.0])), DType)
    assert isinstance(aux_module._np_dtype_op(np, [1, 2]), DType)
    reg_dt = numpy_eager_registry.get("DType")
    assert isinstance(typing.cast("EagerCallableProtocol", reg_dt)(np, "int32"), DType)
    assert aux_module._eager_Gradient(np, np.array([1.0, 2.0, 4.0])) is not None
    assert aux_module._eager_I0(np, np.array([0.0])) is not None
    assert typing.cast("EagerTensorProtocol", getattr(aux_module._eager_BroadcastedIota(np, 3, (2, 3)), "shape")) == (2, 3)

    # Gradient & I0 fallback branch
    no_grad_mod = types.SimpleNamespace()
    assert aux_module._eager_Gradient(typing.cast("BackendModuleProtocol", no_grad_mod), 42) == 42
    assert aux_module._eager_I0(typing.cast("BackendModuleProtocol", no_grad_mod), 42) == 42
    getattr(_state, "update")({"no_grad_mod": no_grad_mod, "reg_dt": reg_dt, "reg_fb": reg_fb})


def test_math_auxiliary_ops(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test auxiliary math operations directly and through eager registry.

    Args:
        monkeypatch (pytest.MonkeyPatch): Pytest fixture.

    Returns:
        None
    """
    _state: dict[str, EagerTensorProtocol] = typing.cast("dict[str, EagerTensorProtocol]", {"monkeypatch": monkeypatch})
    _test_math_auxiliary_ops_part_1(_state)
    _test_math_auxiliary_ops_part_2(_state)
    _test_math_auxiliary_ops_part_3(_state)
    _test_math_auxiliary_ops_part_4(_state)
    assert len(_state) > 0
