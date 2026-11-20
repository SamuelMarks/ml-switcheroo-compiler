# ruff: noqa: C901, PLR0912, PLR0915, F841
"""Tests for NumPy eager statistical distributions and SciPy fallback branches."""

from __future__ import annotations

import importlib
import types

import numpy as np
import pytest

import tests.backends.numpy.eager._math_advanced_test_ns as _mod_raw
from ml_switcheroo_compiler.backends.eager.utils import _resolve_eager_mod
from ml_switcheroo_compiler.backends.eager_registry import BackendArg, BackendOperand
from tests.backends.numpy.eager._msf_forwards_stub import BaseMathMiscState, _make_base_math_state
from tests.testing_utils import (
    EagerBoundCallable,
    EagerTensorProtocol,
    MockTestStub,
    adapt_backend_module,
    require_callable,
)

mod = _mod_raw


class _TestMathMiscCoverageState(BaseMathMiscState, total=False):
    """Typed shared state for `_test_math_misc_coverage` split helpers."""

    ExplodingFunc: type
    FakeDescBackend: type[MockTestStub]
    FakeDescBackendHas: type[MockTestStub]
    MockDistributionsFunc: EagerBoundCallable
    _ops: types.ModuleType
    _orig_desc: EagerTensorProtocol | None
    _orig_dist: EagerTensorProtocol | None
    old_scipy: types.ModuleType | None
    raise_err: EagerBoundCallable
    sys: types.ModuleType
    test_desc_func: EagerBoundCallable


def _test_math_misc_coverage_part_376(_state: _TestMathMiscCoverageState) -> None:
    """Execute part 376 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    bk = _state["bk"]
    mod._np_customlinearsolve(bk, lambda x: x, np.array([1.0]), solve=lambda f, x: x)
    mod._np_descriptive(bk, np.array([1.0]))
    mod._np_distributions(bk, np.array([1.0]))
    mod._np_decode_csv(bk, [b"1,2"], record_defaults=[[0.0], [0.0]])
    import sys

    class FakeScipy:
        """Test class FakeScipy."""

        i1 = None
        k0 = None
        k1 = None

    old_scipy = sys.modules.get("scipy.special")
    pytest.MonkeyPatch().setitem(sys.modules, "scipy.special", None)
    mod._np_modified_bessel_i1(bk, np.array([1.0]))
    mod._np_modified_bessel_k0(bk, np.array([1.0]))
    _state.update({"old_scipy": old_scipy, "sys": sys})
    _state.update({"bk": bk})


def _test_math_misc_coverage_part_377(_state: _TestMathMiscCoverageState) -> None:
    """Execute part 377 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (bk, old_scipy, sys) = (_state["bk"], _state["old_scipy"], _state["sys"])
    mod._np_modified_bessel_k1(bk, np.array([1.0]))
    if old_scipy is not None:
        pytest.MonkeyPatch().setitem(sys.modules, "scipy.special", old_scipy)
    else:
        sys.modules.pop("scipy.special", None)
    with pytest.raises(IndexError):
        mod._np_bessel_i0e(bk)
    with pytest.raises(IndexError):
        mod._np_bessel_i1e(bk)
    mod._np_modified_bessel_i0(bk)
    mod._np_modified_bessel_i1(bk)
    mod._np_modified_bessel_k0(bk)
    _state.update({"bk": bk, "old_scipy": old_scipy, "sys": sys})


def _test_math_misc_coverage_part_378(_state: _TestMathMiscCoverageState) -> None:
    """Execute part 378 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    bk = _state["bk"]
    mod._np_modified_bessel_k1(bk)
    from tests.backends.numpy.eager._math_advanced_test_ns import math_matrix_utils, math_poly

    math_general = importlib.import_module("tests.backends.numpy.eager._math_advanced_test_ns").math_general
    assert math_matrix_utils._np_diag_indices_(np, 2) is not None
    assert math_matrix_utils._np_diag_indices_from_(np, np.eye(2)) is not None
    assert math_matrix_utils._np_diagflat_(np, np.array([1, 2])) is not None
    assert math_matrix_utils._np_diagonal_(np, np.eye(2)) is not None
    assert math_matrix_utils._np_indices_(np, (2, 2)) is not None
    assert math_matrix_utils._np_mask_indices_(np, 2, np.triu) is not None
    assert math_general._np_apply_over_axes(np, np.sum, np.ones((2, 2)), [0]) is not None
    assert math_general._np_corrcoef(np, np.array([1, 2, 3])) is not None
    assert math_general._np_cov(np, np.array([1, 2, 3])) is not None
    assert math_poly._np_shifted_chebyshev_polynomial_t(np, np.array([1]), np.array([1])) is not None
    assert math_poly._np_shifted_chebyshev_polynomial_u(np, np.array([1]), np.array([1])) is not None
    assert math_poly._np_shifted_chebyshev_polynomial_v(np, np.array([1]), np.array([1])) is not None
    assert math_poly._np_shifted_chebyshev_polynomial_w(np, np.array([1]), np.array([1])) is not None
    assert math_poly._np_shifted_chebyshev_polynomial_t(np) is None
    assert math_poly._np_shifted_chebyshev_polynomial_u(np) is None
    assert math_poly._np_shifted_chebyshev_polynomial_v(np) is None
    assert math_poly._np_shifted_chebyshev_polynomial_w(np) is None
    _state.update({"bk": bk})


def _test_math_misc_coverage_part_379(_state: _TestMathMiscCoverageState) -> None:
    """Execute part 379 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    bk = _state["bk"]
    with pytest.raises(TypeError):
        mod._np_descriptive(bk, None)
    with pytest.raises(TypeError):
        mod._np_distributions(bk, None)
    with pytest.raises(TypeError):
        mod._np_rrelu(bk, None)
    mod._get_csv_data((), _resolve_eager_mod(np))
    mod._np_decode_image_camel(bk)
    _state.update({"bk": bk})


def _test_math_misc_coverage_part_380(_state: _TestMathMiscCoverageState) -> None:
    """Execute part 380 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    bk = _state["bk"]
    with pytest.raises(RuntimeError):
        mod._np_decode_image_camel(bk, np.array("invalid str"))

    class DummyFeature:
        """Test class DummyFeature."""

        shape = (1,)
        dtype = np.float32

    mod._np_serialize_tensor_camel(bk)
    _state.update({"bk": bk})


def _test_math_misc_coverage_part_381(_state: _TestMathMiscCoverageState) -> None:
    """Execute part 381 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    bk = _state["bk"]

    class Unpicklable:
        """Test class Unpicklable."""

        def __reduce__(self) -> str:
            """Test function __reduce__.

            Returns:
                str: Computed output value from `__reduce__`.

            Raises:
                ValueError: When raised during `__reduce__` (not picklable).
            """
            raise ValueError("not picklable")

    mod._np_serialize_tensor_camel(bk, np.array([Unpicklable()], dtype="O"))
    pass
    pass
    _state.update({"bk": bk})


def _test_math_misc_coverage_part_382(_state: _TestMathMiscCoverageState) -> None:
    """Execute part 382 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    bk = _state["bk"]

    class DummyBkVecdot(MockTestStub):
        """Test class DummyBkVecdot."""

        @staticmethod
        def iscomplexobj(x: BackendArg) -> BackendArg:
            """Test function iscomplexobj.

            Args:
                x: Test parameter `x`.

            Returns:
                BackendArg: Computed output value from `iscomplexobj`.
            """
            return True

        @staticmethod
        def conj(x: BackendArg) -> BackendArg:
            """Test function conj.

            Args:
                x: Test parameter `x`.

            Returns:
                BackendArg: Computed output value from `conj`.
            """
            return x

        @staticmethod
        def sum(x: BackendArg, axis: BackendArg) -> BackendArg:
            """Test function sum.

            Args:
                x: Test parameter `x`.
                axis: Test parameter `axis`.

            Returns:
                BackendArg: Computed output value from `sum`.
            """
            return x

    mod._np_vecdot(DummyBkVecdot(), np.array([1.0]), np.array([1.0]))
    with pytest.raises(RuntimeError):
        mod._np_parse_tensor_camel(bk, np.array([1.0]))
    _ops = importlib.import_module("ml_switcheroo_compiler.ops")

    def test_desc_func(*args: EagerTensorProtocol, **kwargs: EagerTensorProtocol) -> BackendArg:
        """Test function test_desc_func.

        Args:
            args: Test parameter `args`.
            kwargs: Test parameter `kwargs`.

        Returns:
            BackendArg: Computed output value from `test_desc_func`.
        """
        assert len(args) >= 1
        return args[0]

    _state.update({"_ops": _ops, "test_desc_func": require_callable(test_desc_func)})
    _state.update({"bk": bk})


def _test_math_misc_coverage_part_383(_state: _TestMathMiscCoverageState) -> None:
    """Execute part 383 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (_ops, bk, dummy_bk, test_desc_func) = (_state["_ops"], _state["bk"], _state["dummy_bk"], _state["test_desc_func"])
    mp = pytest.MonkeyPatch()
    _orig_desc = adapt_backend_module(_ops).get_symbol("descriptive", None)
    mp.setattr(_ops, "descriptive", test_desc_func, raising=False)
    mod._np_descriptive(bk, np.array([1.0]))
    if _orig_desc is not None:
        mp.setattr(_ops, "descriptive", _orig_desc, raising=False)
    else:
        mp.delattr(_ops, "descriptive", raising=False)
    _orig_dist = adapt_backend_module(_ops).get_symbol("distributions", None)
    mp.setattr(_ops, "distributions", test_desc_func, raising=False)
    mod._np_distributions(bk, np.array([1.0]))
    if _orig_dist is not None:
        mp.setattr(_ops, "distributions", _orig_dist, raising=False)
    else:
        mp.delattr(_ops, "distributions", raising=False)
    mod._np_descriptive(dummy_bk, np.array([1.0]))
    _state.update({"_orig_desc": _orig_desc, "_orig_dist": _orig_dist})
    _state.update({"_ops": _ops, "bk": bk, "dummy_bk": dummy_bk})
    _state.update({"test_desc_func": require_callable(test_desc_func)})


def _test_math_misc_coverage_part_384(_state: _TestMathMiscCoverageState) -> None:
    """Execute part 384 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (bk, dummy_bk) = (_state["bk"], _state["dummy_bk"])
    mod._np_distributions(dummy_bk, np.array([1.0]))

    class EmptyOps:
        """Test class EmptyOps."""

        class OpDef:
            """Test class OpDef."""

            pass

    import sys

    pass
    mod._np_descriptive(bk, np.array([1.0]))
    mod._np_distributions(bk, np.array([1.0]))
    pass
    _state.update({"sys": sys})
    _state.update({"bk": bk, "dummy_bk": dummy_bk})


def _test_math_misc_coverage_part_385(_state: _TestMathMiscCoverageState) -> None:
    """Execute part 385 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    dummy_bk = _state["dummy_bk"]

    class FakeDescBackend(MockTestStub):
        """Test class FakeDescBackend."""

        @staticmethod
        def descriptive(*args: BackendArg, **kwargs: BackendArg) -> BackendArg:
            """Test function descriptive.

            Args:
                args: Test parameter `args`.
                kwargs: Test parameter `kwargs`.

            Returns:
                BackendArg: Computed output value from `descriptive`.
            """
            return "desc_bk"

        @staticmethod
        def distributions(*args: BackendArg, **kwargs: BackendArg) -> BackendArg:
            """Test function distributions.

            Args:
                args: Test parameter `args`.
                kwargs: Test parameter `kwargs`.

            Returns:
                BackendArg: Computed output value from `distributions`.
            """
            return "dist_bk"

    mod._np_descriptive(FakeDescBackend(), np.array([1.0]))
    mod._np_distributions(FakeDescBackend(), np.array([1.0]))
    _ops = importlib.import_module("ml_switcheroo_compiler.ops")
    pytest.MonkeyPatch().setattr(_ops, "descriptive", None, raising=False)
    pytest.MonkeyPatch().setattr(_ops, "distributions", None, raising=False)
    mod._np_descriptive(dummy_bk, np.array([1.0]))
    _state.update({"FakeDescBackend": FakeDescBackend, "_ops": _ops})
    _state.update({"dummy_bk": dummy_bk})


def _test_math_misc_coverage_part_386(_state: _TestMathMiscCoverageState) -> None:
    """Execute part 386 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (_ops, bk, dummy_bk) = (_state["_ops"], _state["bk"], _state["dummy_bk"])
    mod._np_distributions(dummy_bk, np.array([1.0]))
    mp = pytest.MonkeyPatch()
    mp.delattr(_ops, "descriptive", raising=False)
    mp.delattr(_ops, "distributions", raising=False)
    _ops = importlib.import_module("ml_switcheroo_compiler.ops")

    def test_func(*a: EagerTensorProtocol, **k: EagerTensorProtocol) -> BackendArg:
        """Test function test_func.

        Args:
            a: Test parameter `a`.
            k: Test parameter `k`.

        Returns:
            BackendArg: Computed output value from `test_func`.
        """
        assert len(a) >= 1
        return a[0]

    _orig_desc = adapt_backend_module(_ops).get_symbol("descriptive", None)
    mp.setattr(_ops, "descriptive", test_func, raising=False)
    mod._np_descriptive(bk, np.array([1.0]))
    if _orig_desc is not None:
        mp.setattr(_ops, "descriptive", _orig_desc, raising=False)
    else:
        mp.delattr(_ops, "descriptive", raising=False)
    _orig_dist = adapt_backend_module(_ops).get_symbol("distributions", None)
    mp.setattr(_ops, "distributions", test_func, raising=False)
    _state.update({"_ops": _ops, "_orig_desc": _orig_desc, "_orig_dist": _orig_dist})
    _state.update({"bk": bk, "dummy_bk": dummy_bk})


def _test_math_misc_coverage_part_387(_state: _TestMathMiscCoverageState) -> None:
    """Execute part 387 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (_ops, _orig_dist, bk, dummy_bk) = (_state["_ops"], _state["_orig_dist"], _state["bk"], _state["dummy_bk"])
    mod._np_distributions(bk, np.array([1.0]))
    mp = pytest.MonkeyPatch()
    if _orig_dist is not None:
        mp.setattr(_ops, "distributions", _orig_dist, raising=False)
    else:
        mp.delattr(_ops, "distributions", raising=False)
    mod._np_fromdlpack(dummy_bk, np.array([1.0]))
    with pytest.raises(AttributeError):
        mod._np_randomcategorical(dummy_bk, np.array([1.0]))
    with pytest.raises(AttributeError):
        mod._np_vecdot(dummy_bk, np.array([1j]), np.array([1j]))
    _state.update({"_ops": _ops, "_orig_dist": _orig_dist, "bk": bk})
    _state.update({"dummy_bk": dummy_bk})


def _test_math_misc_coverage_part_388(_state: _TestMathMiscCoverageState) -> None:
    """Execute part 388 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    dummy_bk = _state["dummy_bk"]
    mod._np_distributions(dummy_bk, np.array([1.0]))
    _ops = importlib.import_module("ml_switcheroo_compiler.ops")

    def buggy_desc(*a: BackendArg, **k: BackendArg) -> None:
        """Test function buggy_desc.

        Args:
            a: Test parameter `a`.
            k: Test parameter `k`.

        Raises:
            ValueError: When raised during `buggy_desc` (buggy).
        """
        raise ValueError("buggy")

    mp = pytest.MonkeyPatch()
    _orig_desc = adapt_backend_module(_ops).get_symbol("descriptive", None)
    mp.setattr(_ops, "descriptive", buggy_desc, raising=False)
    mod._np_descriptive(dummy_bk, np.array([1.0]))
    if _orig_desc is not None:
        mp.setattr(_ops, "descriptive", _orig_desc, raising=False)
    else:
        mp.delattr(_ops, "descriptive", raising=False)
    _orig_dist = adapt_backend_module(_ops).get_symbol("distributions", None)
    mp.setattr(_ops, "distributions", buggy_desc, raising=False)
    mod._np_distributions(dummy_bk, np.array([1.0]))
    _state.update({"_ops": _ops, "_orig_desc": _orig_desc, "_orig_dist": _orig_dist})
    _state.update({"dummy_bk": dummy_bk})


def _test_math_misc_coverage_part_389(_state: _TestMathMiscCoverageState) -> None:
    """Execute part 389 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (_ops, _orig_dist, dummy_bk) = (_state["_ops"], _state["_orig_dist"], _state["dummy_bk"])
    mp = pytest.MonkeyPatch()
    if _orig_dist is not None:
        mp.setattr(_ops, "distributions", _orig_dist, raising=False)
    else:
        mp.delattr(_ops, "distributions", raising=False)
    _ops = importlib.import_module("ml_switcheroo_compiler.ops")
    _orig_desc = adapt_backend_module(_ops).get_symbol("descriptive", None)
    if _orig_desc is not None:
        mp.delattr(_ops, "descriptive", raising=False)
    mod._np_descriptive(dummy_bk)
    if _orig_desc is not None:
        mp.setattr(_ops, "descriptive", _orig_desc, raising=False)
    _orig_dist = adapt_backend_module(_ops).get_symbol("distributions", None)
    if _orig_dist is not None:
        mp.delattr(_ops, "distributions", raising=False)
    _state.update({"_ops": _ops, "_orig_desc": _orig_desc, "_orig_dist": _orig_dist})
    _state.update({"dummy_bk": dummy_bk})


def _test_math_misc_coverage_part_390(_state: _TestMathMiscCoverageState) -> None:
    """Execute part 390 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (_ops, _orig_dist, dummy_bk) = (_state["_ops"], _state["_orig_dist"], _state["dummy_bk"])
    mp = pytest.MonkeyPatch()
    mod._np_distributions(dummy_bk)
    if _orig_dist is not None:
        mp.setattr(_ops, "distributions", _orig_dist, raising=False)

    class FakeDescBackend(MockTestStub):
        """Test class FakeDescBackend."""

        pass

    _ops = importlib.import_module("ml_switcheroo_compiler.ops")
    _orig_desc = adapt_backend_module(_ops).get_symbol("descriptive", None)
    if _orig_desc is not None:
        mp.delattr(_ops, "descriptive", raising=False)
    mod._np_descriptive(FakeDescBackend(), np.array([1.0]))
    if _orig_desc is not None:
        mp.setattr(_ops, "descriptive", _orig_desc, raising=False)
    _orig_dist = adapt_backend_module(_ops).get_symbol("distributions", None)
    _state.update(
        {"FakeDescBackend": FakeDescBackend, "_ops": _ops, "_orig_desc": _orig_desc, "_orig_dist": _orig_dist}
    )
    _state.update({"dummy_bk": dummy_bk})


def _test_math_misc_coverage_part_391(_state: _TestMathMiscCoverageState) -> None:
    """Execute part 391 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (FakeDescBackend, _ops, _orig_dist) = (_state["FakeDescBackend"], _state["_ops"], _state["_orig_dist"])
    mp = pytest.MonkeyPatch()
    if _orig_dist is not None:
        mp.delattr(_ops, "distributions", raising=False)
    mod._np_distributions(FakeDescBackend(), np.array([1.0]))
    if _orig_dist is not None:
        mp.setattr(_ops, "distributions", _orig_dist, raising=False)

    def MockDescriptiveFunc(*args: BackendArg, **kwargs: BackendArg) -> BackendOperand:
        """Test function MockDescriptiveFunc.

        Args:
            args: Test parameter `args`.
            kwargs: Test parameter `kwargs`.

        Returns:
            EagerTensorProtocol: Computed output value from `MockDescriptiveFunc`.
        """
        return "mock_desc"

    def MockDistributionsFunc(*args: BackendArg, **kwargs: BackendArg) -> BackendOperand:
        """Test function MockDistributionsFunc.

        Args:
            args: Test parameter `args`.
            kwargs: Test parameter `kwargs`.

        Returns:
            EagerTensorProtocol: Computed output value from `MockDistributionsFunc`.
        """
        return "mock_dist"

    _ops = importlib.import_module("ml_switcheroo_compiler.ops")
    _orig_desc = adapt_backend_module(_ops).get_symbol("descriptive", None)
    mp.setattr(_ops, "descriptive", MockDescriptiveFunc, raising=False)
    _state.update(
        {"MockDistributionsFunc": require_callable(MockDistributionsFunc), "_ops": _ops, "_orig_desc": _orig_desc}
    )
    _state.update({"FakeDescBackend": FakeDescBackend, "_orig_dist": _orig_dist})


def _test_math_misc_coverage_part_392(_state: _TestMathMiscCoverageState) -> None:
    """Execute part 392 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (MockDistributionsFunc, _ops, _orig_desc, dummy_bk) = (
        _state["MockDistributionsFunc"],
        _state["_ops"],
        _state["_orig_desc"],
        _state["dummy_bk"],
    )
    mp = pytest.MonkeyPatch()
    mod._np_descriptive(dummy_bk, np.array([1.0]))
    if _orig_desc is not None:
        mp.setattr(_ops, "descriptive", _orig_desc, raising=False)
    else:
        mp.delattr(_ops, "descriptive", raising=False)
    _orig_dist = adapt_backend_module(_ops).get_symbol("distributions", None)
    mp.setattr(_ops, "distributions", MockDistributionsFunc, raising=False)
    mod._np_distributions(dummy_bk, np.array([1.0]))
    if _orig_dist is not None:
        mp.setattr(_ops, "distributions", _orig_dist, raising=False)
    else:
        mp.delattr(_ops, "distributions", raising=False)

    class ExplodingFunc:
        """Test class ExplodingFunc."""

        def __new__(cls, *args: BackendArg, **kwargs: BackendArg) -> ExplodingFunc:
            """Test function __new__.

            Args:
                args: Test parameter `args`.
                kwargs: Test parameter `kwargs`.

            Returns:
                ExplodingFunc: Computed output value from `__new__`.

            Raises:
                ValueError: When raised during `__new__` (exploding).
            """
            raise ValueError("exploding")

    _orig_desc = adapt_backend_module(_ops).get_symbol("descriptive", None)
    mp.setattr(_ops, "descriptive", ExplodingFunc, raising=False)
    _state.update({"ExplodingFunc": ExplodingFunc, "_orig_desc": _orig_desc, "_orig_dist": _orig_dist})
    _state.update(
        {"MockDistributionsFunc": require_callable(MockDistributionsFunc), "_ops": _ops, "dummy_bk": dummy_bk}
    )


def _test_math_misc_coverage_part_393(_state: _TestMathMiscCoverageState) -> None:
    """Execute part 393 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (ExplodingFunc, _ops, _orig_desc, dummy_bk) = (
        _state["ExplodingFunc"],
        _state["_ops"],
        _state["_orig_desc"],
        _state["dummy_bk"],
    )
    mp = pytest.MonkeyPatch()
    mod._np_descriptive(dummy_bk, np.array([1.0]))
    if _orig_desc is not None:
        mp.setattr(_ops, "descriptive", _orig_desc, raising=False)
    else:
        mp.delattr(_ops, "descriptive", raising=False)
    _orig_dist = adapt_backend_module(_ops).get_symbol("distributions", None)
    mp.setattr(_ops, "distributions", ExplodingFunc, raising=False)
    mod._np_distributions(dummy_bk, np.array([1.0]))
    if _orig_dist is not None:
        mp.setattr(_ops, "distributions", _orig_dist, raising=False)
    else:
        mp.delattr(_ops, "distributions", raising=False)
    _ops = importlib.import_module("ml_switcheroo_compiler.ops")

    def raise_err(*args: BackendArg, **kwargs: BackendArg) -> None:
        """Test function raise_err.

        Args:
            args: Test parameter `args`.
            kwargs: Test parameter `kwargs`.

        Raises:
            ValueError: When raised during `raise_err` (error).
        """
        raise ValueError("error")

    _orig_desc = adapt_backend_module(_ops).get_symbol("descriptive", None)
    mp.setattr(_ops, "descriptive", raise_err, raising=False)
    _state.update(
        {"_ops": _ops, "_orig_desc": _orig_desc, "_orig_dist": _orig_dist, "raise_err": require_callable(raise_err)}
    )
    _state.update({"ExplodingFunc": ExplodingFunc, "dummy_bk": dummy_bk})


def _test_math_misc_coverage_part_394(_state: _TestMathMiscCoverageState) -> None:
    """Execute part 394 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (_ops, _orig_desc, dummy_bk, raise_err) = (
        _state["_ops"],
        _state["_orig_desc"],
        _state["dummy_bk"],
        _state["raise_err"],
    )
    mp = pytest.MonkeyPatch()
    mod._np_descriptive(dummy_bk, np.array([1.0]))
    if _orig_desc is not None:
        mp.setattr(_ops, "descriptive", _orig_desc, raising=False)
    else:
        mp.delattr(_ops, "descriptive", raising=False)
    _orig_dist = adapt_backend_module(_ops).get_symbol("distributions", None)
    mp.setattr(_ops, "distributions", raise_err, raising=False)
    mod._np_distributions(dummy_bk, np.array([1.0]))
    if _orig_dist is not None:
        mp.setattr(_ops, "distributions", _orig_dist, raising=False)
    else:
        mp.delattr(_ops, "distributions", raising=False)

    class FakeDescBackendHas(MockTestStub):
        """Test class FakeDescBackendHas."""

        @staticmethod
        def descriptive(*args: BackendArg, **kwargs: BackendArg) -> BackendArg:
            """Test function descriptive.

            Args:
                args: Test parameter `args`.
                kwargs: Test parameter `kwargs`.

            Returns:
                BackendArg: Computed output value from `descriptive`.
            """
            return "desc_bk"

        @staticmethod
        def distributions(*args: BackendArg, **kwargs: BackendArg) -> BackendArg:
            """Test function distributions.

            Args:
                args: Test parameter `args`.
                kwargs: Test parameter `kwargs`.

            Returns:
                BackendArg: Computed output value from `distributions`.
            """
            return "dist_bk"

    _state.update({"FakeDescBackendHas": FakeDescBackendHas, "_orig_dist": _orig_dist})
    _state.update({"_ops": _ops, "_orig_desc": _orig_desc, "dummy_bk": dummy_bk})
    _state.update({"raise_err": require_callable(raise_err)})


def _test_math_misc_coverage_part_395(_state: _TestMathMiscCoverageState) -> None:
    """Execute part 395 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    FakeDescBackendHas = _state["FakeDescBackendHas"]
    _ops = importlib.import_module("ml_switcheroo_compiler.ops")

    class MagicOps:
        """Test class MagicOps."""

        @property
        def descriptive(self) -> None:
            """Test function descriptive.

            Raises:
                RuntimeError: When raised during `descriptive` (Boom).
            """
            raise RuntimeError("Boom")

        @property
        def distributions(self) -> None:
            """Test function distributions.

            Raises:
                RuntimeError: When raised during `distributions` (Boom).
            """
            raise RuntimeError("Boom")

    import sys

    pass
    mod._np_descriptive(FakeDescBackendHas(), np.array([1.0]))
    mod._np_distributions(FakeDescBackendHas(), np.array([1.0]))
    pass
    _state.update({"_ops": _ops, "sys": sys})
    _state.update({"FakeDescBackendHas": FakeDescBackendHas})


def test_msf_distributions_scipy() -> None:
    """Tests for NumPy eager statistical distributions and SciPy fallback branches runner."""
    _base = _make_base_math_state(376)
    _state: _TestMathMiscCoverageState = {
        "DummyBk": _base["DummyBk"],
        "arg1": _base["arg1"],
        "arg2": _base["arg2"],
        "arg_bool": _base["arg_bool"],
        "arg_int": _base["arg_int"],
        "bk": _base["bk"],
        "dummy_bk": _base["dummy_bk"],
    }
    for _part_fn in (
        _test_math_misc_coverage_part_376,
        _test_math_misc_coverage_part_377,
        _test_math_misc_coverage_part_378,
        _test_math_misc_coverage_part_379,
        _test_math_misc_coverage_part_380,
        _test_math_misc_coverage_part_381,
        _test_math_misc_coverage_part_382,
        _test_math_misc_coverage_part_383,
        _test_math_misc_coverage_part_384,
        _test_math_misc_coverage_part_385,
        _test_math_misc_coverage_part_386,
        _test_math_misc_coverage_part_387,
        _test_math_misc_coverage_part_388,
        _test_math_misc_coverage_part_389,
        _test_math_misc_coverage_part_390,
        _test_math_misc_coverage_part_391,
        _test_math_misc_coverage_part_392,
        _test_math_misc_coverage_part_393,
        _test_math_misc_coverage_part_394,
        _test_math_misc_coverage_part_395,
    ):
        _part_fn(_state)
    np.testing.assert_array_equal(np.asarray(_state["arg1"]), np.array([0.0, 0.0]))
