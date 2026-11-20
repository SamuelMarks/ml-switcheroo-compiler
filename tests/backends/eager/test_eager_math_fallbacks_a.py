"""Direct behavioral and fallback-branch tests for `core_math_ops` general, linalg, logical, and manipulation."""

from __future__ import annotations

import types

import numpy as np
import pytest

from ml_switcheroo_compiler.backends.eager.core_math_ops import (
    math_general as mg,
)
from ml_switcheroo_compiler.backends.eager.core_math_ops import (
    math_linalg as mla,
)
from ml_switcheroo_compiler.backends.eager.core_math_ops import (
    math_logical as mlog,
)
from ml_switcheroo_compiler.backends.eager.core_math_ops import (
    math_manipulation as mmanip,
)
from ml_switcheroo_compiler.backends.eager_registry import BackendArg
from ml_switcheroo_compiler.backends.numpy.eager import ensure_ndarray
from ml_switcheroo_compiler.core.shape import VariadicCallableProtocol
from tests.testing_utils import make_mock_module


def _mod(name: str, attrs: dict[str, BackendArg]) -> types.ModuleType:
    """Create a mock module with `attrs` bound via `make_mock_module`.

    Args:
        name (str): Module name.
        attrs (dict[str, BackendArg]): Symbol table to attach.

    Returns:
        types.ModuleType: Constructed module stub.
    """
    del name
    return make_mock_module(**attrs)


def _tag_fn(tag: str) -> VariadicCallableProtocol:
    """Build a deterministic callable returning `tag` for custom backend-symbol dispatch tests.

    Args:
        tag (str): Expected marker string.

    Returns:
        VariadicCallableProtocol: Variadic marker function.
    """

    def _fn(*args: BackendArg, **kwargs: BackendArg) -> str:
        """Return `tag` marker string.

        Args:
            *args (BackendArg): Positional arguments.
            **kwargs (BackendArg): Keyword arguments.

        Returns:
            str: Marker string `tag`.
        """
        del args, kwargs
        return tag

    return _fn


class _CustomPopQueue:
    """Custom non-list queue object exposing `.pop(index)` and `.shape` (`BackendArg`)."""

    def __init__(self, items: list[int]) -> None:
        """Initialize backing queue items and `.shape`.

        Args:
            items (list[int]): Initial integer batch values.
        """
        self.shape: tuple[int, ...] = ()
        self._items: list[int] = list(items)

    def pop(self, index: int) -> BackendArg:
        """Pop element at `index`.

        Args:
            index (int): Element index to remove.

        Returns:
            BackendArg: Popped integer item.
        """
        return self._items.pop(index)


def test_math_general_direct_and_symbol_fallbacks() -> None:
    """Verify shadowed direct helpers and custom/fallback branches in `core_math_ops/math_general.py`."""
    arr = np.array([1.0, 2.0, 3.0])
    assert mg._apply_over_axes(np, _tag_fn("unused"), arr) is arr
    assert mg._apply_over_axes(np, _tag_fn("only_one")) is None
    assert "array" in str(mg._array_repr(np, np.array([1, 2])))
    assert "[1 2]" in str(mg._array_str(np, np.array([1, 2])))
    assert mg._assign(np, arr, 42.0) == 42.0
    assert mg._assign_variable(np, arr, 99.0) == 99.0

    empty_mod = types.ModuleType("empty_mod")
    assert ensure_ndarray(mg._atleast_1d(np, 5.0)).ndim == 1
    assert mg._atleast_1d(empty_mod, 5.0) == 5.0
    assert ensure_ndarray(mg._atleast_2d(np, 5.0)).ndim == 2
    assert mg._atleast_2d(empty_mod, 5.0) == 5.0
    assert ensure_ndarray(mg._atleast_3d(np, 5.0)).ndim == 3
    assert mg._atleast_3d(empty_mod, 5.0) == 5.0

    adj_mod = _mod("adj_mod", {"adjoint": _tag_fn("adj_hit")})
    assert mg._adjoint(adj_mod, arr) == "adj_hit"

    np.testing.assert_allclose(ensure_ndarray(mg._ediff1d(np, np.array([1.0, 4.0, 9.0]))), np.array([3.0, 5.0]))
    np.testing.assert_allclose(ensure_ndarray(mg._gradient(np, np.array([1.0, 3.0, 5.0]))), np.array([2.0, 2.0, 2.0]))
    assert mg._interp(np, 1.5, [1.0, 2.0], [10.0, 20.0]) == 15.0
    assert bool(mg._iterable(np, [1, 2])) is True
    ix_grid = mg._ix(np, [0, 1], [2, 3])
    assert isinstance(ix_grid, tuple) and len(ix_grid) == 2
    np.testing.assert_allclose(ensure_ndarray(mg._kron(np, [1.0, 2.0], [3.0, 4.0])), np.array([3.0, 4.0, 6.0, 8.0]))

    r_mod = _mod("r_mod", {"r_": _tag_fn("r_hit")})
    assert mg._r(r_mod, arr) == "r_hit"


def test_math_general_trapezoid_variance_vectorize_and_zerofraction() -> None:
    """Verify trapezoid, variance, vectorize, and zero-fraction branches in `core_math_ops/math_general.py`."""
    y = np.array([1.0, 2.0, 3.0])
    trap_mod = _mod("trap_mod", {"trapezoid": _tag_fn("trapezoid_direct")})
    trapz_only_mod = _mod("trapz_only_mod", {"trapz": lambda a: float(np.sum(a))})
    empty_mod = types.ModuleType("empty_mod")

    assert mg._trapezoid(trap_mod, y) == "trapezoid_direct"
    assert mg._trapezoid(trapz_only_mod, y) == 6.0
    vec_mod = _mod("vec_mod", {"vectorize": _tag_fn("vec_direct")})
    assert mg._vectorize(vec_mod, _tag_fn("f")) == "vec_direct"

    assert mg._np_trapezoid(trap_mod, y) == "trapezoid_direct"
    assert float(ensure_ndarray(mg._np_trapezoid(empty_mod, y))) == 4.0

    ti_mod = _mod("ti_mod", {"trapezoidalintegral": _tag_fn("ti_direct")})
    assert mg._np_trapezoidalintegral(ti_mod, y) == "ti_direct"
    assert float(ensure_ndarray(mg._np_trapezoidalintegral(empty_mod, y))) == 4.0

    var_mod = _mod("var_mod", {"variance": _tag_fn("var_direct")})
    assert mg._np_variance(var_mod, y) == "var_direct"
    np.testing.assert_allclose(ensure_ndarray(mg._np_variance(empty_mod, np.array([1.0, 3.0]))), 1.0)

    assert mg._np_vectorize(_mod("v2", {"vectorize": _tag_fn("v2_hit")}), _tag_fn("f")) == "v2_hit"
    v_fn = mg._np_vectorize(empty_mod, lambda x: x + 1)
    assert callable(v_fn)
    np.testing.assert_allclose(ensure_ndarray(v_fn(np.array([1, 2]))), np.array([2, 3]))

    zf_mod = _mod("zf_mod", {"zerofraction": _tag_fn("zf_hit")})
    assert mg._np_zerofraction(zf_mod, y) == "zf_hit"
    assert float(ensure_ndarray(mg._np_zerofraction(empty_mod, np.array([0.0, 2.0])))) == 0.5


def test_math_linalg_top_level_and_np_fallbacks() -> None:
    """Verify top-level symbol hits and fallback branches in `core_math_ops/math_linalg.py`."""
    eye2 = np.eye(2)
    top_linalg_mod = _mod(
        "top_linalg_mod",
        {
            "det": _tag_fn("det_top"),
            "norm": _tag_fn("norm_top"),
            "pinv": _tag_fn("pinv_top"),
            "qr": _tag_fn("qr_top"),
            "svd": _tag_fn("svd_top"),
            "tensorinv": _tag_fn("tinv_top"),
            "inv": _tag_fn("inv_top"),
            "vander": _tag_fn("vander_top"),
        },
    )
    assert mla._det(top_linalg_mod, eye2) == "det_top"
    assert mla._norm(top_linalg_mod, eye2) == "norm_top"
    assert mla._pinv(top_linalg_mod, eye2) == "pinv_top"
    assert mla._qr(top_linalg_mod, eye2) == "qr_top"
    assert mla._svd(top_linalg_mod, eye2) == "svd_top"
    assert mla._tensorinv(top_linalg_mod, eye2) == "tinv_top"
    assert mla._inv(top_linalg_mod, eye2) == "inv_top"
    assert mla._vander(top_linalg_mod, np.array([1.0, 2.0])) == "vander_top"

    empty_mod = types.ModuleType("empty_mod")
    assert mla._np_svd(top_linalg_mod, eye2) == "svd_top"
    svd_out = mla._np_svd(empty_mod, eye2)
    assert isinstance(svd_out, tuple) and len(svd_out) == 3

    eye_4d = np.eye(4).reshape(2, 2, 2, 2)
    assert mla._np_tensorinv(top_linalg_mod, eye_4d) == "tinv_top"
    assert ensure_ndarray(mla._np_tensorinv(empty_mod, eye_4d)).shape == (2, 2, 2, 2)

    assert mla._np_vander(top_linalg_mod, np.array([1, 2, 3])) == "vander_top"
    assert ensure_ndarray(mla._np_vander(empty_mod, np.array([1, 2, 3]))).shape == (3, 3)


def test_math_logical_branches() -> None:
    """Verify uncovered branches in `core_math_ops/math_logical.py`."""
    sub_math = _mod(
        "sub_math",
        {
            "divide_no_nan": _tag_fn("div_no_nan_hit"),
            "multiply_no_nan": _tag_fn("mul_no_nan_hit"),
        },
    )
    tf_like = _mod("tf_like", {"math": sub_math})
    assert mlog._divide_no_nan(tf_like, 4.0, 2.0) == "div_no_nan_hit"
    assert mlog._multiply_no_nan(tf_like, 4.0, 2.0) == "mul_no_nan_hit"

    hist_out = mlog._histogram(np, np.array([1.0, 2.0, 3.0]), bins=2)
    assert isinstance(hist_out, tuple) and len(hist_out) == 2

    mlog.clear_infeed_queues()
    custom_q = _CustomPopQueue([321])
    mlog.register_infeed_queue("pop_proto_q", custom_q)
    try:
        assert mlog._infeed(np, "pop_proto_q") == 321
        assert mlog._infeed(np, "pop_proto_q") == 0
        mlog.register_infeed_queue("non_pop_q", 42)
        assert mlog._infeed(np, "non_pop_q") == 0
    finally:
        mlog.clear_infeed_queues()

    empty_mod = types.ModuleType("empty_mod")
    custom_pred_mod = _mod(
        "custom_pred_mod",
        {
            "iscomplexobj": _tag_fn("iscomplexobj_hit"),
            "isrealobj": _tag_fn("isrealobj_hit"),
            "isscalar": _tag_fn("isscalar_hit"),
            "piecewise": _tag_fn("piecewise_hit"),
        },
    )
    assert mlog._iscomplexobj(custom_pred_mod, 1 + 2j) == "iscomplexobj_hit"
    assert bool(mlog._iscomplexobj(empty_mod, np.array([1 + 2j]))) is True
    assert mlog._isrealobj(custom_pred_mod, 1.0) == "isrealobj_hit"
    assert bool(mlog._isrealobj(empty_mod, np.array([1.0]))) is True
    assert mlog._isscalar(custom_pred_mod, 3.0) == "isscalar_hit"
    assert bool(mlog._isscalar(empty_mod, 3.0)) is True
    assert bool(mlog._issubdtype(empty_mod, np.int32, np.integer)) is True

    mish_val = ensure_ndarray(mlog._mish(np, np.array([0.0])))
    np.testing.assert_allclose(mish_val, np.array([0.0]), atol=1e-6)
    assert mlog._piecewise(custom_pred_mod, np.array([1.0]), [True], [1.0]) == "piecewise_hit"


def test_math_manipulation_branches() -> None:
    """Verify uncovered branches in `core_math_ops/math_manipulation.py`."""
    empty_mod = types.ModuleType("empty_mod")
    mat2 = np.array([[1, 2], [3, 4]])
    assert ensure_ndarray(mmanip._rot90(np, mat2)).shape == (2, 2)
    nz = mmanip._nonzero(np, np.array([0, 5, 0]))
    assert isinstance(nz, tuple) and len(nz) == 1
    assert ensure_ndarray(mmanip._tile(np, np.array([1, 2]), 2)).tolist() == [1, 2, 1, 2]
    with pytest.raises(RuntimeError, match="Expected numpy-like backend"):
        mmanip._updateslice(empty_mod, mat2, np.array([[9]]), (0, 0))

    manip_custom = _mod(
        "manip_custom",
        {
            "hsplit": _tag_fn("hsplit_hit"),
            "scatternd": _tag_fn("scatternd_hit"),
            "tensorarraystack": _tag_fn("tastack_hit"),
            "tensorscatterupdate": _tag_fn("tsupdate_hit"),
            "unstack": _tag_fn("unstack_hit"),
            "vsplit": _tag_fn("vsplit_hit"),
        },
    )
    assert mmanip._np_hsplit(manip_custom, mat2, 2) == "hsplit_hit"
    hsplit_out = mmanip._np_hsplit(empty_mod, mat2, 2)
    assert isinstance(hsplit_out, list) and len(hsplit_out) == 2
    assert mmanip._np_scatternd(manip_custom, mat2) == "scatternd_hit"
    assert mmanip._np_scatternd(empty_mod, 123) == 123
    assert mmanip._np_tensorarraystack(manip_custom, [mat2]) == "tastack_hit"
    assert ensure_ndarray(mmanip._np_tensorarraystack(empty_mod, [np.array([1]), np.array([2])])).shape == (2, 1)
    assert mmanip._np_tensorscatterupdate(manip_custom, mat2) == "tsupdate_hit"
    assert mmanip._np_tensorscatterupdate(empty_mod, 456) == 456
    assert mmanip._np_unstack(manip_custom, mat2) == "unstack_hit"
    unstacked = mmanip._np_unstack(empty_mod, mat2, axis=0)
    assert isinstance(unstacked, list) and len(unstacked) == 2
    assert mmanip._np_vsplit(manip_custom, mat2, 2) == "vsplit_hit"
    vsplit_out = mmanip._np_vsplit(empty_mod, mat2, 2)
    assert isinstance(vsplit_out, list) and len(vsplit_out) == 2
