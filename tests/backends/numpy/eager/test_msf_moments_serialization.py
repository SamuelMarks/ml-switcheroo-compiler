# ruff: noqa: C901, PLR0912, PLR0915, F841
"""Tests for NumPy eager statistical moments, covariance, and tensor serialization."""

from __future__ import annotations

import importlib
import types

import numpy as np
import pytest

import tests.backends.numpy.eager._math_advanced_test_ns as _mod_raw
from ml_switcheroo_compiler.backends.eager_registry import BackendArg
from ml_switcheroo_compiler.backends.numpy.eager import ensure_ndarray
from tests.backends.numpy.eager._msf_forwards_stub import BaseMathMiscState, _make_base_math_state
from tests.testing_utils import EagerTensorProtocol, MockTestStub, adapt_backend_module

mod = _mod_raw


class _TestMathMiscCoverageState(BaseMathMiscState, total=False):
    """Typed shared state for `_test_math_misc_coverage` split helpers."""

    _ops: types.ModuleType
    _orig_cm: EagerTensorProtocol | None
    _orig_desc: EagerTensorProtocol | None
    _orig_dist: EagerTensorProtocol | None
    _orig_rem: EagerTensorProtocol | None
    dist: types.ModuleType


def _test_math_misc_coverage_part_353(_state: _TestMathMiscCoverageState) -> None:
    """Execute part 353 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, arg_bool, bk) = (_state["arg1"], _state["arg2"], _state["arg_bool"], _state["bk"])
    dummy_bk = _state["dummy_bk"]
    mod._np_decode_csv_camel(dummy_bk, arg1, arg2)
    with pytest.raises(RuntimeError):
        mod._np_decode_image_camel(bk, arg1)
    with pytest.raises(RuntimeError):
        mod._np_decode_image_camel(bk, arg1, arg2)
    with pytest.raises(RuntimeError):
        mod._np_decode_image_camel(bk, arg1, arg1, arg2)
    with pytest.raises(RuntimeError):
        mod._np_decode_image_camel(bk, arg_bool)
    _state.update({"arg1": arg1, "arg2": arg2, "arg_bool": arg_bool})
    _state.update({"bk": bk, "dummy_bk": dummy_bk})


def _test_math_misc_coverage_part_354(_state: _TestMathMiscCoverageState) -> None:
    """Execute part 354 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, arg_int, bk) = (_state["arg1"], _state["arg2"], _state["arg_int"], _state["bk"])
    dummy_bk = _state["dummy_bk"]
    with pytest.raises(RuntimeError):
        mod._np_decode_image_camel(bk, arg_int)
    with pytest.raises(RuntimeError):
        mod._np_decode_image_camel(dummy_bk, arg1)
    with pytest.raises(RuntimeError):
        mod._np_decode_image_camel(dummy_bk, arg1, arg2)
    _state.update({"arg1": arg1, "arg2": arg2, "arg_int": arg_int})
    _state.update({"bk": bk, "dummy_bk": dummy_bk})


def _test_math_misc_coverage_part_355(_state: _TestMathMiscCoverageState) -> None:
    """Execute part 355 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, arg_bool, arg_int) = (_state["arg1"], _state["arg2"], _state["arg_bool"], _state["arg_int"])
    (bk, dummy_bk) = (_state["bk"], _state["dummy_bk"])
    assert bk is not None
    _state.update({"arg1": arg1, "arg2": arg2, "arg_bool": arg_bool})
    _state.update({"arg_int": arg_int, "bk": bk, "dummy_bk": dummy_bk})


def _test_math_misc_coverage_part_356(_state: _TestMathMiscCoverageState) -> None:
    """Execute part 356 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, arg_bool, arg_int) = (_state["arg1"], _state["arg2"], _state["arg_bool"], _state["arg_int"])
    bk = _state["bk"]
    with pytest.raises(RuntimeError):
        mod._np_parse_tensor_camel(bk, arg1)
    with pytest.raises(RuntimeError):
        mod._np_parse_tensor_camel(bk, arg1, arg2)
    with pytest.raises(RuntimeError):
        mod._np_parse_tensor_camel(bk, arg1, arg1, arg2)
    with pytest.raises(RuntimeError):
        mod._np_parse_tensor_camel(bk, arg_bool)
    with pytest.raises(RuntimeError):
        mod._np_parse_tensor_camel(bk, arg_int)
    _state.update({"arg1": arg1, "arg2": arg2, "arg_bool": arg_bool})
    _state.update({"arg_int": arg_int, "bk": bk})


def _test_math_misc_coverage_part_357(_state: _TestMathMiscCoverageState) -> None:
    """Execute part 357 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, bk, dummy_bk) = (_state["arg1"], _state["arg2"], _state["bk"], _state["dummy_bk"])
    with pytest.raises(RuntimeError):
        mod._np_parse_tensor_camel(dummy_bk, arg1)
    with pytest.raises(RuntimeError):
        mod._np_parse_tensor_camel(dummy_bk, arg1, arg2)
    with pytest.raises(RuntimeError):
        mod._np_read_file_camel(bk, arg1)
    with pytest.raises(RuntimeError):
        mod._np_read_file_camel(bk, arg1, arg2)
    with pytest.raises(RuntimeError):
        mod._np_read_file_camel(bk, arg1, arg1, arg2)
    _state.update({"arg1": arg1, "arg2": arg2, "bk": bk})
    _state.update({"dummy_bk": dummy_bk})


def _test_math_misc_coverage_part_358(_state: _TestMathMiscCoverageState) -> None:
    """Execute part 358 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, arg_bool, arg_int) = (_state["arg1"], _state["arg2"], _state["arg_bool"], _state["arg_int"])
    (bk, dummy_bk) = (_state["bk"], _state["dummy_bk"])
    with pytest.raises(RuntimeError):
        mod._np_read_file_camel(bk, arg_bool)
    with pytest.raises(RuntimeError):
        mod._np_read_file_camel(bk, arg_int)
    with pytest.raises(RuntimeError):
        mod._np_read_file_camel(dummy_bk, arg1)
    with pytest.raises(RuntimeError):
        mod._np_read_file_camel(dummy_bk, arg1, arg2)
    with pytest.raises(AttributeError):
        adapt_backend_module(mod).require_symbol("_np_rem_2")(bk, arg1)
    _state.update({"arg1": arg1, "arg2": arg2, "arg_bool": arg_bool})
    _state.update({"arg_int": arg_int, "bk": bk, "dummy_bk": dummy_bk})


def _test_math_misc_coverage_part_359(_state: _TestMathMiscCoverageState) -> None:
    """Execute part 359 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, arg_bool, arg_int) = (_state["arg1"], _state["arg2"], _state["arg_bool"], _state["arg_int"])
    (bk, dummy_bk) = (_state["bk"], _state["dummy_bk"])
    with pytest.raises(AttributeError):
        adapt_backend_module(mod).require_symbol("_np_rem_2")(bk, arg1, arg2)
    with pytest.raises(AttributeError):
        adapt_backend_module(mod).require_symbol("_np_rem_2")(bk, arg1, arg1, arg2)
    with pytest.raises(AttributeError):
        adapt_backend_module(mod).require_symbol("_np_rem_2")(bk, arg_bool)
    with pytest.raises(AttributeError):
        adapt_backend_module(mod).require_symbol("_np_rem_2")(bk, arg_int)
    with pytest.raises(AttributeError):
        adapt_backend_module(mod).require_symbol("_np_rem_2")(dummy_bk, arg1)
    _state.update({"arg1": arg1, "arg2": arg2, "arg_bool": arg_bool})
    _state.update({"arg_int": arg_int, "bk": bk, "dummy_bk": dummy_bk})


def _test_math_misc_coverage_part_360(_state: _TestMathMiscCoverageState) -> None:
    """Execute part 360 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, arg_bool, bk) = (_state["arg1"], _state["arg2"], _state["arg_bool"], _state["bk"])
    dummy_bk = _state["dummy_bk"]
    with pytest.raises(AttributeError):
        adapt_backend_module(mod).require_symbol("_np_rem_2")(dummy_bk, arg1, arg2)
    mod._np_serialize_tensor_camel(bk, arg1)
    mod._np_serialize_tensor_camel(bk, arg1, arg2)
    mod._np_serialize_tensor_camel(bk, arg1, arg1, arg2)
    mod._np_serialize_tensor_camel(bk, arg_bool)
    _state.update({"arg1": arg1, "arg2": arg2, "arg_bool": arg_bool})
    _state.update({"bk": bk, "dummy_bk": dummy_bk})


def _test_math_misc_coverage_part_361(_state: _TestMathMiscCoverageState) -> None:
    """Execute part 361 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, arg_int, bk) = (_state["arg1"], _state["arg2"], _state["arg_int"], _state["bk"])
    dummy_bk = _state["dummy_bk"]
    mod._np_serialize_tensor_camel(bk, arg_int)
    mod._np_serialize_tensor_camel(dummy_bk, arg1)
    mod._np_serialize_tensor_camel(dummy_bk, arg1, arg2)
    pass
    pass
    _state.update({"arg1": arg1, "arg2": arg2, "arg_int": arg_int})
    _state.update({"bk": bk, "dummy_bk": dummy_bk})


def _test_math_misc_coverage_part_362(_state: _TestMathMiscCoverageState) -> None:
    """Execute part 362 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    pass
    pass
    pass
    pass
    pass


def _test_math_misc_coverage_part_363(_state: _TestMathMiscCoverageState) -> None:
    """Execute part 363 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, arg_bool, arg_int) = (_state["arg1"], _state["arg2"], _state["arg_bool"], _state["arg_int"])
    bk = _state["bk"]
    mod._np_frombuffer(bk, arg1)
    mod._np_frombuffer(bk, arg1, arg2)
    mod._np_frombuffer(bk, arg1, arg1, arg2)
    with pytest.raises(ValueError):
        mod._np_frombuffer(bk, arg_bool)
    mod._np_frombuffer(bk, arg_int)
    _state.update({"arg1": arg1, "arg2": arg2, "arg_bool": arg_bool})
    _state.update({"arg_int": arg_int, "bk": bk})


def _test_math_misc_coverage_part_364(_state: _TestMathMiscCoverageState) -> None:
    """Execute part 364 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, arg2, bk, dummy_bk) = (_state["arg1"], _state["arg2"], _state["bk"], _state["dummy_bk"])
    mod._np_frombuffer(dummy_bk, arg1)
    mod._np_frombuffer(dummy_bk, arg1, arg2)
    mod._build_dot_general_einsum_str(2, 2, (((1,), (0,)), ((), ())))
    mod._build_dot_general_einsum_str(2, 2, (((1,), (0,)), ((0,), (1,))))
    mod._np_logsumexp(bk, arg1, axis=0, keepdims=True)
    _state.update({"arg1": arg1, "arg2": arg2, "bk": bk})
    _state.update({"dummy_bk": dummy_bk})


def _test_math_misc_coverage_part_365(_state: _TestMathMiscCoverageState) -> None:
    """Execute part 365 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, bk) = (_state["arg1"], _state["bk"])
    mod._np_logsumexp(bk, arg1, axis=0, keepdims=False)
    mod._np_segment_sum(bk, arg1, np.array([0, 1]))
    mod._np_segment_sum(bk, arg1, np.array([0, 1]), num_segments=2)
    import ml_switcheroo_compiler.backends.numpy.eager.distributed as dist

    dist._tcp_dist_ctx.world_size = 2
    psum_res = mod._np_psum(bk, arg1)
    assert ensure_ndarray(psum_res).shape == arg1.shape
    dist._tcp_dist_ctx.world_size = 1
    dist._tcp_dist_ctx.world_size = 2
    with pytest.raises(AttributeError):
        adapt_backend_module(mod).require_symbol("_np_pmax")(bk, arg1)
    dist._tcp_dist_ctx.world_size = 1
    dist._tcp_dist_ctx.world_size = 2
    _state.update({"dist": dist})
    _state.update({"arg1": arg1, "bk": bk})


def _test_math_misc_coverage_part_366(_state: _TestMathMiscCoverageState) -> None:
    """Execute part 366 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, bk, dist) = (_state["arg1"], _state["bk"], _state["dist"])
    with pytest.raises(AttributeError):
        adapt_backend_module(mod).require_symbol("_np_pmin")(bk, arg1)
    pytest.MonkeyPatch().setattr(dist._tcp_dist_ctx, "world_size", 1, raising=False)
    mod._np_scanop(bk, lambda a, b: a + b, arg1, 0, True)
    with pytest.raises(TypeError):
        mod._np_scanop(bk, lambda a, b: a + b, arg1, None, False)
    mod._np_scanop(bk, None, arg1)
    mod._np_scanop(bk, lambda a, b: a + b, np.array([]), 0, True)
    _state.update({"arg1": arg1, "bk": bk, "dist": dist})


def _test_math_misc_coverage_part_367(_state: _TestMathMiscCoverageState) -> None:
    """Execute part 367 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (bk, dummy_bk) = (_state["bk"], _state["dummy_bk"])
    mod._np_sobolsample(bk, 2, 2)
    mod._np_sobolsample(bk, 2, 2, skip=5)
    with pytest.raises(ValueError):
        mod._np_debuginfs(dummy_bk, np.array([1.0, np.inf]))
    with pytest.raises(ValueError):
        mod._np_debugnans(dummy_bk, np.array([1.0, np.nan]))
    mod._np_confusion_matrix(bk, np.array([0, 1]), np.array([0, 1]))
    _ops = importlib.import_module("ml_switcheroo_compiler.ops")
    _orig_rem = adapt_backend_module(_ops).get_symbol("rem", None)
    _orig_cm = adapt_backend_module(_ops).get_symbol("confusion_matrix", None)
    _orig_desc = adapt_backend_module(_ops).get_symbol("descriptive", None)
    _orig_dist = adapt_backend_module(_ops).get_symbol("distributions", None)
    _state.update({"_ops": _ops, "_orig_cm": _orig_cm, "_orig_desc": _orig_desc, "_orig_dist": _orig_dist})
    _state.update({"_orig_rem": _orig_rem})
    _state.update({"bk": bk, "dummy_bk": dummy_bk})


def _test_math_misc_coverage_part_368(_state: _TestMathMiscCoverageState) -> None:
    """Execute part 368 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (_ops, arg1, arg2, arg_int) = (_state["_ops"], _state["arg1"], _state["arg2"], _state["arg_int"])
    bk = _state["bk"]

    def my_rem(*args: BackendArg, **kwargs: BackendArg) -> BackendArg:
        """Test function my_rem.

        Args:
            args: Test parameter `args`.
            kwargs: Test parameter `kwargs`.

        Returns:
            BackendArg: Computed output value from `my_rem`.
        """
        return "mock"

    def my_cm(*args: BackendArg, **kwargs: BackendArg) -> BackendArg:
        """Test function my_cm.

        Args:
            args: Test parameter `args`.
            kwargs: Test parameter `kwargs`.

        Returns:
            BackendArg: Computed output value from `my_cm`.
        """
        return "mock"

    def my_desc(*args: BackendArg, **kwargs: BackendArg) -> BackendArg:
        """Test function my_desc.

        Args:
            args: Test parameter `args`.
            kwargs: Test parameter `kwargs`.

        Returns:
            BackendArg: Computed output value from `my_desc`.
        """
        return "mock"

    def my_dist(*args: BackendArg, **kwargs: BackendArg) -> BackendArg:
        """Test function my_dist.

        Args:
            args: Test parameter `args`.
            kwargs: Test parameter `kwargs`.

        Returns:
            BackendArg: Computed output value from `my_dist`.
        """
        return "mock"

    pytest.MonkeyPatch().setattr(_ops, "rem", my_rem, raising=False)
    pytest.MonkeyPatch().setattr(_ops, "confusion_matrix", my_cm, raising=False)
    pytest.MonkeyPatch().setattr(_ops, "descriptive", my_desc, raising=False)
    pytest.MonkeyPatch().setattr(_ops, "distributions", my_dist, raising=False)
    with pytest.raises(AttributeError):
        adapt_backend_module(mod).require_symbol("_np_rem_2")(bk, arg1, arg2)
    mod._np_confusion_matrix(bk, arg_int, arg_int)
    _state.update({"_ops": _ops, "arg1": arg1, "arg2": arg2})
    _state.update({"arg_int": arg_int, "bk": bk})


def _test_math_misc_coverage_part_369(_state: _TestMathMiscCoverageState) -> None:
    """Execute part 369 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (_ops, _orig_cm, _orig_desc, _orig_rem) = (
        _state["_ops"],
        _state["_orig_cm"],
        _state["_orig_desc"],
        _state["_orig_rem"],
    )
    (arg1, bk) = (_state["arg1"], _state["bk"])
    mod._np_descriptive(bk, arg1)
    mod._np_distributions(bk, arg1)
    mp = pytest.MonkeyPatch()
    if _orig_rem is not None:
        mp.setattr(_ops, "rem", _orig_rem, raising=False)
    else:
        mp.delattr(_ops, "rem", raising=False)
    if _orig_cm is not None:
        mp.setattr(_ops, "confusion_matrix", _orig_cm, raising=False)
    else:
        mp.delattr(_ops, "confusion_matrix", raising=False)
    if _orig_desc is not None:
        mp.setattr(_ops, "descriptive", _orig_desc, raising=False)
    else:
        mp.delattr(_ops, "descriptive", raising=False)
    _state.update({"_ops": _ops, "_orig_cm": _orig_cm, "_orig_desc": _orig_desc})
    _state.update({"_orig_rem": _orig_rem, "arg1": arg1, "bk": bk})


def _test_math_misc_coverage_part_370(_state: _TestMathMiscCoverageState) -> None:
    """Execute part 370 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (_ops, _orig_dist, bk, dummy_bk) = (_state["_ops"], _state["_orig_dist"], _state["bk"], _state["dummy_bk"])
    mp = pytest.MonkeyPatch()
    if _orig_dist is not None:
        mp.setattr(_ops, "distributions", _orig_dist, raising=False)
    else:
        mp.delattr(_ops, "distributions", raising=False)
    mod._np_vecdot(bk, np.array([1j]), np.array([1j]))
    with pytest.raises(AttributeError):
        adapt_backend_module(mod).require_symbol("_np_qr")(dummy_bk, np.eye(2), mode="complete")
    with pytest.raises(AttributeError):
        adapt_backend_module(mod).require_symbol("_np_svd")(dummy_bk, np.eye(2), full_matrices=True, compute_uv=False)
    with pytest.raises(AttributeError):
        adapt_backend_module(mod).require_symbol("_np_logdet")(dummy_bk, np.eye(2))
    _state.update({"_ops": _ops, "_orig_dist": _orig_dist, "bk": bk})
    _state.update({"dummy_bk": dummy_bk})


def _test_math_misc_coverage_part_371(_state: _TestMathMiscCoverageState) -> None:
    """Execute part 371 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, bk, dummy_bk) = (_state["arg1"], _state["bk"], _state["dummy_bk"])
    with pytest.raises(AttributeError):
        adapt_backend_module(mod).require_symbol("_np_slogdet")(dummy_bk, np.eye(2))
    mod._np_tensorarraywrite(bk, [1, 2], 5, arg1)
    mod._np_tensorconfig(bk, (), "float32", None)
    funcs2 = [
        "BitwiseCount",
        "BitwiseLeftShift",
        "BitwiseRightShift",
        "Bincount",
        "Clip",
        "Clz",
        "Erf",
        "Erfc",
        "Erfinv",
        "Igamma",
        "Igammac",
        "PopulationCount",
        "ScatterApply",
        "ScatterMax",
        "ScatterMin",
        "ScatterMul",
        "ScatterNd",
        "Select",
        "BitwiseNot",
        "BitwiseOr",
        "BitwiseXor",
    ]
    for _f in funcs2:
        with pytest.raises(TypeError):
            mod._np_bitwise_count(bk, arg1)
    _state.update({"arg1": arg1, "bk": bk, "dummy_bk": dummy_bk})


def _test_math_misc_coverage_part_372(_state: _TestMathMiscCoverageState) -> None:
    """Execute part 372 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (arg1, bk, dummy_bk) = (_state["arg1"], _state["bk"], _state["dummy_bk"])
    mod._np_fromdlpack(dummy_bk, arg1)
    with pytest.raises(AttributeError):
        mod._np_randomcategorical(dummy_bk, np.array([[1.0, 2.0]]))
    with pytest.raises(AttributeError):
        mod._np_randompermutation(dummy_bk, np.array([1, 2]))
    with pytest.raises(AttributeError):
        mod._np_randompermutation(dummy_bk, np.array(5))
    mod._np_serialize_tensor(bk, np.array([1.0]), test=True)
    _state.update({"arg1": arg1, "bk": bk, "dummy_bk": dummy_bk})


def _test_math_misc_coverage_part_373(_state: _TestMathMiscCoverageState) -> None:
    """Execute part 373 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    bk = _state["bk"]
    pass
    mod._np_customlinearsolve(bk, lambda x: x, np.array([1.0]))

    class DummyDesc(MockTestStub):
        """Test class DummyDesc."""

        @staticmethod
        def descriptive(x: BackendArg) -> BackendArg:
            """Test function descriptive.

            Args:
                x: Test parameter `x`.

            Returns:
                BackendArg: Computed output value from `descriptive`.
            """
            return x

    mod._np_descriptive(DummyDesc(), np.array([1.0]))

    class DummyDist(MockTestStub):
        """Test class DummyDist."""

        @staticmethod
        def distributions(x: BackendArg) -> BackendArg:
            """Test function distributions.

            Args:
                x: Test parameter `x`.

            Returns:
                BackendArg: Computed output value from `distributions`.
            """
            return x

    mod._np_distributions(DummyDist(), np.array([1.0]))
    _state.update({"bk": bk})


def _test_math_misc_coverage_part_374(_state: _TestMathMiscCoverageState) -> None:
    """Execute part 374 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    bk = _state["bk"]
    _ops = importlib.import_module("ml_switcheroo_compiler.ops")

    class DummyDescClass:
        """Test class DummyDescClass."""

        def __init__(self, *args: BackendArg, **kwargs: BackendArg) -> None:
            """Test function __init__.

            Args:
                args: Test parameter `args`.
                kwargs: Test parameter `kwargs`.
            """
            pass

    _orig_desc = adapt_backend_module(_ops).get_symbol("descriptive", None)
    mp = pytest.MonkeyPatch()
    mp.setattr(_ops, "descriptive", DummyDescClass, raising=False)
    mod._np_descriptive(bk, np.array([1.0]))
    if _orig_desc is not None:
        mp.setattr(_ops, "descriptive", _orig_desc, raising=False)
    else:
        mp.delattr(_ops, "descriptive", raising=False)

    class DummyDistClass:
        """Test class DummyDistClass."""

        def __init__(self, *args: BackendArg, **kwargs: BackendArg) -> None:
            """Test function __init__.

            Args:
                args: Test parameter `args`.
                kwargs: Test parameter `kwargs`.
            """
            pass

    _orig_dist = adapt_backend_module(_ops).get_symbol("distributions", None)
    mp.setattr(_ops, "distributions", DummyDistClass, raising=False)
    _state.update({"_ops": _ops, "_orig_desc": _orig_desc})
    _state.update({"_orig_dist": _orig_dist})
    _state.update({"bk": bk})


def _test_math_misc_coverage_part_375(_state: _TestMathMiscCoverageState) -> None:
    """Execute part 375 of test_math_misc_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (_ops, _orig_dist, bk) = (_state["_ops"], _state["_orig_dist"], _state["bk"])
    mod._np_distributions(bk, np.array([1.0]))
    mp = pytest.MonkeyPatch()
    if _orig_dist is not None:
        mp.setattr(_ops, "distributions", _orig_dist, raising=False)
    else:
        mp.delattr(_ops, "distributions", raising=False)
    mod._np_customlinearsolve(bk, lambda x: x, np.array([1.0]), lambda f, x: x)
    mod._np_customlinearsolve(bk, lambda x: x, np.array([1.0]), solve=lambda f, x: x)
    mod._np_customlinearsolve(bk, lambda x: x, np.array([1.0]), lambda f, x: x)
    _state.update({"_ops": _ops, "_orig_dist": _orig_dist, "bk": bk})


def test_msf_moments_serialization() -> None:
    """Tests for NumPy eager statistical moments, covariance, and tensor serialization runner."""
    _base = _make_base_math_state(353)
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
        _test_math_misc_coverage_part_353,
        _test_math_misc_coverage_part_354,
        _test_math_misc_coverage_part_355,
        _test_math_misc_coverage_part_356,
        _test_math_misc_coverage_part_357,
        _test_math_misc_coverage_part_358,
        _test_math_misc_coverage_part_359,
        _test_math_misc_coverage_part_360,
        _test_math_misc_coverage_part_361,
        _test_math_misc_coverage_part_362,
        _test_math_misc_coverage_part_363,
        _test_math_misc_coverage_part_364,
        _test_math_misc_coverage_part_365,
        _test_math_misc_coverage_part_366,
        _test_math_misc_coverage_part_367,
        _test_math_misc_coverage_part_368,
        _test_math_misc_coverage_part_369,
        _test_math_misc_coverage_part_370,
        _test_math_misc_coverage_part_371,
        _test_math_misc_coverage_part_372,
        _test_math_misc_coverage_part_373,
        _test_math_misc_coverage_part_374,
        _test_math_misc_coverage_part_375,
    ):
        _part_fn(_state)
    np.testing.assert_array_equal(np.asarray(_state["arg1"]), np.array([0.0, 0.0]))
