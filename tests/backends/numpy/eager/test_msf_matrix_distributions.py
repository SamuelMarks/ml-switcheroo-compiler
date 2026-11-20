# ruff: noqa: C901, PLR0912, PLR0915, F841
"""Tests for NumPy eager matrix utilities, confusion matrix, and distribution fallbacks."""

from __future__ import annotations

import importlib
import types
import typing
from collections.abc import Mapping, Sequence

import numpy as np
import pytest

import ml_switcheroo_compiler.ops as ops
import tests.backends.numpy.eager._math_advanced_test_ns as _mod_raw
from ml_switcheroo_compiler.backends.eager_registry import BackendArg
from ml_switcheroo_compiler.backends.numpy.eager import ensure_ndarray
from tests.backends.numpy.eager._math_advanced_test_ns import global_eager_registry
from tests.testing_utils import (
    EagerBoundCallable,
    MockTestStub,
    adapt_backend_module,
    extract_test_shape,
    require_callable,
)

mod = _mod_raw


class DummyBackend(MockTestStub):
    """Test class DummyBackend."""

    pass


class MockCallableClass:
    """Test class MockCallableClass."""

    def __init__(self, *args: BackendArg, **kwargs: BackendArg) -> None:
        """Test function __init__.

        Args:
            args: Test parameter `args`.
            kwargs: Test parameter `kwargs`.
        """
        self.called = True


def test_math_misc_missing_branches_mocked() -> None:
    """Test function test_math_misc_missing_branches_mocked."""
    import ml_switcheroo_compiler.ops as ops

    orig_rem = adapt_backend_module(ops).get_symbol("rem", None)
    orig_cm = adapt_backend_module(ops).get_symbol("confusion_matrix", None)
    orig_desc = adapt_backend_module(ops).get_symbol("descriptive", None)
    orig_dist = adapt_backend_module(ops).get_symbol("distributions", None)
    if orig_rem is not None:
        pytest.MonkeyPatch().delattr(ops, "rem")
    if orig_cm is not None:
        pytest.MonkeyPatch().delattr(ops, "confusion_matrix")
    if orig_desc is not None:
        pytest.MonkeyPatch().delattr(ops, "descriptive")
    if orig_dist is not None:
        pytest.MonkeyPatch().delattr(ops, "distributions")
    try:
        res = global_eager_registry._registry["RawMatMul"](None, np.array([[1]]), np.array([[2]]))
        assert res.shape == (1, 1)
        res = global_eager_registry._registry["SparseDenseMatMul"](None, np.array([[1]]), np.array([[2]]))
        assert res.shape == (1, 1)
        res = global_eager_registry._registry["rem"](None, np.array([5]), np.array([2]))
        assert res[0] == 1
        res = global_eager_registry._registry["confusion_matrix"](None, np.array([0]), np.array([0]), num_classes=2)
        assert res.shape == (2, 2)
        res = global_eager_registry._registry["descriptive"](None, np.array([1, 2]))
        assert len(res) == 3
        res = global_eager_registry._registry["distributions"](None, np.array([1, 2]))
        assert len(res) == 2
    finally:
        if orig_rem is not None:
            pytest.MonkeyPatch().setattr(ops, "rem", orig_rem, raising=False)
        if orig_cm is not None:
            pytest.MonkeyPatch().setattr(ops, "confusion_matrix", orig_cm, raising=False)
        if orig_desc is not None:
            pytest.MonkeyPatch().setattr(ops, "descriptive", orig_desc, raising=False)
        if orig_dist is not None:
            pytest.MonkeyPatch().setattr(ops, "distributions", orig_dist, raising=False)


def test_math_advanced_oserror() -> None:
    """Test function test_math_advanced_oserror."""
    from unittest.mock import mock_open, patch

    import pytest

    from tests.backends.numpy.eager._math_advanced_test_ns import _np_write_file as _np_save_img

    with patch("builtins.open", mock_open()) as mocked_file:
        mocked_file.side_effect = RuntimeError("test")
        with pytest.raises(OSError):
            require_callable(_np_save_img)("dummy", "dummy.png", b"dummy")


def test_math_advanced_mocked_fallbacks_2() -> None:
    """Test function test_math_advanced_mocked_fallbacks_2."""
    from unittest.mock import MagicMock

    mock_np = MagicMock()
    mock_np.rem.return_value = "rem"
    mock_np.confusion_matrix.return_value = "cm"
    _np_rem_lower = global_eager_registry._registry["rem"]
    assert _np_rem_lower(mock_np, 1, 2) == "rem"
    _np_cm_lower = global_eager_registry._registry["confusion_matrix"]
    assert _np_cm_lower(mock_np, 1, 2) == "cm"


class _TestMathMatrixUtilsDistributionsSuccessState(typing.TypedDict, total=False):
    """Typed shared state for `_test_math_matrix_utils_distributions_success` split helpers."""

    FakeInst: type
    FakeOpDef: type
    _np_callable: EagerBoundCallable
    _np_clip: EagerBoundCallable
    _np_confusion_matrix: EagerBoundCallable
    _np_descriptive_2: EagerBoundCallable
    _np_distributions: EagerBoundCallable
    _np_key: EagerBoundCallable
    _np_one_hot: EagerBoundCallable
    _np_tensor: EagerBoundCallable
    _ops: types.ModuleType
    np: types.ModuleType


def _test_math_matrix_utils_distributions_success_part_1(_state: _TestMathMatrixUtilsDistributionsSuccessState) -> None:
    """Execute part 1 of test_math_matrix_utils_distributions_success.

    Args:
        _state: Shared test state dictionary.
    """
    import numpy as np

    from ml_switcheroo_compiler.backends.eager.reference_math.math_general import (
        _np_callable,
        _np_clip,
        _np_descriptive_2,
        _np_key,
        _np_one_hot,
        _np_rem_2,
        _np_tensor,
    )
    from ml_switcheroo_compiler.backends.eager.reference_math.math_matrix_utils import (
        _np_confusion_matrix,
        _np_distributions,
    )

    class FakeOpDef:
        """Test class FakeOpDef."""

        pass

    class FakeInst:
        """Test class FakeInst."""

        def __call__(self, *args: BackendArg, **kwargs: BackendArg) -> BackendArg:
            """Test function __call__.

            Args:
                args: Test parameter `args`.
                kwargs: Test parameter `kwargs`.

            Returns:
                BackendArg: Computed output value from `__call__`.
            """
            return 42

    from unittest.mock import patch

    _ops = importlib.import_module("ml_switcheroo_compiler.ops")
    with patch.object(_ops, "OpDef", FakeOpDef, create=True):
        with patch.object(_ops, "rem", FakeInst(), create=True):
            _np_rem_2(np, 1, 2)
        with patch.object(_ops, "rem", property(lambda _: int("invalid")), create=True):
            _np_rem_2(np, 1, 2)
    _state.update(
        {
            "FakeInst": FakeInst,
            "FakeOpDef": FakeOpDef,
            "_np_callable": require_callable(_np_callable),
            "_np_clip": require_callable(_np_clip),
        }
    )
    _state.update(
        {
            "_np_confusion_matrix": require_callable(_np_confusion_matrix),
            "_np_descriptive_2": require_callable(_np_descriptive_2),
            "_np_distributions": require_callable(_np_distributions),
            "_np_key": require_callable(_np_key),
        }
    )
    _state.update(
        {"_np_one_hot": require_callable(_np_one_hot), "_np_tensor": require_callable(_np_tensor), "_ops": _ops}
    )
    _state.update({"np": np})


def _test_math_matrix_utils_distributions_success_part_2(_state: _TestMathMatrixUtilsDistributionsSuccessState) -> None:
    """Execute part 2 of test_math_matrix_utils_distributions_success.

    Args:
        _state: Shared test state dictionary.
    """
    from unittest.mock import patch

    (FakeInst, FakeOpDef, _np_descriptive_2, _ops) = (
        _state["FakeInst"],
        _state["FakeOpDef"],
        _state["_np_descriptive_2"],
        _state["_ops"],
    )
    np = _state["np"]
    with patch.object(_ops, "OpDef", FakeOpDef, create=True):
        with patch.object(_ops, "descriptive", FakeInst(), create=True):
            _np_descriptive_2(np)
        with patch.object(_ops, "descriptive", property(lambda _: int("invalid")), create=True):
            _np_descriptive_2(np)
    _state.update(
        {"FakeInst": FakeInst, "FakeOpDef": FakeOpDef, "_np_descriptive_2": require_callable(_np_descriptive_2)}
    )
    _state.update({"_ops": _ops, "np": np})


def _test_math_matrix_utils_distributions_success_part_3(_state: _TestMathMatrixUtilsDistributionsSuccessState) -> None:
    """Execute part 3 of test_math_matrix_utils_distributions_success.

    Args:
        _state: Shared test state dictionary.
    """
    from unittest.mock import patch

    (FakeInst, FakeOpDef, _np_confusion_matrix, _ops) = (
        _state["FakeInst"],
        _state["FakeOpDef"],
        _state["_np_confusion_matrix"],
        _state["_ops"],
    )
    np = _state["np"]
    with patch.object(_ops, "OpDef", FakeOpDef, create=True):
        with patch.object(_ops, "confusion_matrix", FakeInst(), create=True):
            _np_confusion_matrix(np, [0], [0])
    _state.update(
        {"FakeInst": FakeInst, "FakeOpDef": FakeOpDef, "_np_confusion_matrix": require_callable(_np_confusion_matrix)}
    )
    _state.update({"_ops": _ops, "np": np})


def _test_math_matrix_utils_distributions_success_part_4(_state: _TestMathMatrixUtilsDistributionsSuccessState) -> None:
    """Execute part 4 of test_math_matrix_utils_distributions_success.

    Args:
        _state: Shared test state dictionary.
    """
    from unittest.mock import patch

    (FakeInst, FakeOpDef, _np_callable, _np_clip) = (
        _state["FakeInst"],
        _state["FakeOpDef"],
        _state["_np_callable"],
        _state["_np_clip"],
    )
    (_np_distributions, _np_key, _np_one_hot, _np_tensor) = (
        _state["_np_distributions"],
        _state["_np_key"],
        _state["_np_one_hot"],
        _state["_np_tensor"],
    )
    (_ops, np) = (_state["_ops"], _state["np"])
    with patch.object(_ops, "OpDef", FakeOpDef, create=True):
        with patch.object(_ops, "distributions", FakeInst(), create=True):
            _np_distributions(np, [0])
    assert _np_distributions is not None
    assert _np_callable(np, lambda x: x)
    assert not _np_callable(np)
    np.testing.assert_array_equal(_np_key(np, 1), np.array([1, 0], dtype=np.uint32))
    np.testing.assert_array_equal(_np_key(np), np.array([0, 0], dtype=np.uint32))
    assert _np_clip(np) is None
    res = _np_one_hot(np, [0, 1], depth=3, axis=0, on_value=1, off_value=0)
    expected = np.array([[1, 0], [0, 1], [0, 0]], dtype=float)
    np.testing.assert_array_equal(res, expected)
    res = _np_one_hot(np, [0, 1], depth=3, axis=-1, on_value=1, off_value=0)
    expected = np.array([[1, 0, 0], [0, 1, 0]], dtype=float)
    np.testing.assert_array_equal(res, expected)
    assert len(_np_tensor(np)) == 0


def test_math_matrix_utils_distributions_success() -> None:
    """Test function test_math_matrix_utils_distributions_success."""
    _state: _TestMathMatrixUtilsDistributionsSuccessState = {}
    _test_math_matrix_utils_distributions_success_part_1(_state)
    _test_math_matrix_utils_distributions_success_part_2(_state)
    _test_math_matrix_utils_distributions_success_part_3(_state)
    _test_math_matrix_utils_distributions_success_part_4(_state)
    assert len(_state) > 0


def test_fallback_branches_return(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test function test_fallback_branches_return.

    Args:
        monkeypatch: Test parameter `monkeypatch`.
    """
    from ml_switcheroo_compiler.backends.eager.reference_math.math_general import _np_descriptive_2, _np_rem_2
    from ml_switcheroo_compiler.backends.eager.reference_math.math_matrix_utils import (
        _np_confusion_matrix,
        _np_distributions,
    )

    class DummyOpDef:
        """Test class DummyOpDef."""

        pass

    class FakeDistributions(str):
        """Test class FakeDistributions."""

        def __new__(cls, *args: BackendArg, **kwargs: BackendArg) -> FakeDistributions:
            """Test function __new__.

            Args:
                args: Test parameter `args`.
                kwargs: Test parameter `kwargs`.

            Returns:
                FakeDistributions: Computed output value from `__new__`.
            """
            return super().__new__(cls, "dist2")

    class FakeCM(str):
        """Test class FakeCM."""

        def __new__(cls, *args: BackendArg, **kwargs: BackendArg) -> FakeCM:
            """Test function __new__.

            Args:
                args: Test parameter `args`.
                kwargs: Test parameter `kwargs`.

            Returns:
                FakeCM: Computed output value from `__new__`.
            """
            return super().__new__(cls, "cm2")

    class FakeDesc(str):
        """Test class FakeDesc."""

        def __new__(cls, *args: BackendArg, **kwargs: BackendArg) -> FakeDesc:
            """Test function __new__.

            Args:
                args: Test parameter `args`.
                kwargs: Test parameter `kwargs`.

            Returns:
                FakeDesc: Computed output value from `__new__`.
            """
            return super().__new__(cls, "desc2")

    class FakeRem(str):
        """Test class FakeRem."""

        def __new__(cls, *args: BackendArg, **kwargs: BackendArg) -> FakeRem:
            """Test function __new__.

            Args:
                args: Test parameter `args`.
                kwargs: Test parameter `kwargs`.

            Returns:
                FakeRem: Computed output value from `__new__`.
            """
            return super().__new__(cls, "rem2")

    monkeypatch.setattr(ops, "OpDef", DummyOpDef, raising=False)
    monkeypatch.setattr(ops, "distributions", FakeDistributions, raising=False)
    monkeypatch.setattr(ops, "confusion_matrix", FakeCM, raising=False)
    monkeypatch.setattr(ops, "descriptive", FakeDesc, raising=False)
    monkeypatch.setattr(ops, "rem", FakeRem, raising=False)

    class DummyBackend(MockTestStub):
        """Test class DummyBackend."""

        pass

    assert _np_distributions(DummyBackend(), 1) == "dist2"
    assert _np_confusion_matrix(DummyBackend(), 1) == "cm2"
    assert _np_descriptive_2(DummyBackend(), 1) == "desc2"
    assert _np_rem_2(DummyBackend(), 1) == "rem2"


def test_np_distributions_empty_args() -> None:
    """Test function test_np_distributions_empty_args."""
    from ml_switcheroo_compiler.backends.eager.reference_math.math_matrix_utils import _np_distributions

    class DummyBackend(MockTestStub):
        """Test class DummyBackend."""

        pass

    res = _np_distributions(DummyBackend())
    assert extract_test_shape(res) == (2,)


def test_np_confusion_matrix_cap_none() -> None:
    """Test function test_np_confusion_matrix_cap_none."""
    from ml_switcheroo_compiler.backends.eager.reference_math.math_matrix_utils import _np_confusion_matrix_cap

    class DummyBackend(MockTestStub):
        """Test class DummyBackend."""

        pass

    assert _np_confusion_matrix_cap(DummyBackend()) is None


def test_math_general_rem_3() -> None:
    """Test function test_math_general_rem_3."""
    from ml_switcheroo_compiler.backends.eager.reference_math.math_general import _np_rem_2

    class DummyBackend(MockTestStub):
        """Test class DummyBackend."""

        pass

    assert _np_rem_2(DummyBackend(), 5, 2) == 1
    assert _np_rem_2(DummyBackend()) is None


def test_math_general_scipy_import_error() -> None:
    """Test function test_math_general_scipy_import_error."""
    from ml_switcheroo_compiler.backends.numpy.eager.scipy_special_kernels import _np_betainc

    class DummyBackend(MockTestStub):
        """Test class DummyBackend."""

        pass

    import sys

    mp = pytest.MonkeyPatch()
    orig_scipy = sys.modules.get("scipy.special")
    mp.setitem(sys.modules, "scipy.special", None)
    try:
        with pytest.raises(ImportError):
            _np_betainc(DummyBackend(), 1)
    finally:
        if orig_scipy is not None:
            mp.setitem(sys.modules, "scipy.special", orig_scipy)
        else:
            sys.modules.pop("scipy.special", None)


def test_np_descriptive_2_empty_args() -> None:
    """Test function test_np_descriptive_2_empty_args."""
    from ml_switcheroo_compiler.backends.eager.reference_math.math_general import _np_descriptive_2

    class DummyBackend(MockTestStub):
        """Test class DummyBackend."""

        pass

    res = _np_descriptive_2(DummyBackend())
    assert extract_test_shape(res) == (3,)


def test_fallback_exception() -> None:
    """Test function test_fallback_exception."""
    from unittest.mock import patch

    from ml_switcheroo_compiler.backends.eager.reference_math.math_general import _np_descriptive_2, _np_rem_2
    from ml_switcheroo_compiler.backends.eager.reference_math.math_matrix_utils import (
        _np_confusion_matrix,
        _np_distributions,
    )

    class DummyBackend(MockTestStub):
        """Test class DummyBackend."""

        def distributions(self, *args: BackendArg, **kwargs: BackendArg) -> BackendArg:
            """Test function distributions.

            Args:
                args: Test parameter `args`.
                kwargs: Test parameter `kwargs`.

            Returns:
                BackendArg: Computed output value from `distributions`.
            """
            return "dist_exc"

        def confusion_matrix(self, *args: BackendArg, **kwargs: BackendArg) -> BackendArg:
            """Test function confusion_matrix.

            Args:
                args: Test parameter `args`.
                kwargs: Test parameter `kwargs`.

            Returns:
                BackendArg: Computed output value from `confusion_matrix`.
            """
            return "cm_exc"

        def descriptive(self, *args: BackendArg, **kwargs: BackendArg) -> BackendArg:
            """Test function descriptive.

            Args:
                args: Test parameter `args`.
                kwargs: Test parameter `kwargs`.

            Returns:
                BackendArg: Computed output value from `descriptive`.
            """
            return "desc_exc"

        def rem(self, *args: BackendArg, **kwargs: BackendArg) -> BackendArg:
            """Test function rem.

            Args:
                args: Test parameter `args`.
                kwargs: Test parameter `kwargs`.

            Returns:
                BackendArg: Computed output value from `rem`.
            """
            return "rem_exc"

    orig_import = __import__

    def mock_import(
        name: str,
        globals_map: Mapping[str, BackendArg] | None = None,
        locals_map: Mapping[str, BackendArg] | None = None,
        fromlist: Sequence[str] = (),
        level: int = 0,
    ) -> types.ModuleType:
        """Test function mock_import.

        Args:
            name (str): Module name to import.
            globals_map (Mapping[str, BackendArg] | None): Caller globals mapping.
            locals_map (Mapping[str, BackendArg] | None): Caller locals mapping.
            fromlist (Sequence[str]): Imported attribute names.
            level (int): Relative import level.

        Returns:
            types.ModuleType: Computed output value from `mock_import`.

        Raises:
            RuntimeError: When raised during `mock_import` (boom).
        """
        if name == "ml_switcheroo_compiler.ops":
            raise RuntimeError("boom")
        return orig_import(name, globals_map, locals_map, fromlist, level)

    with patch("builtins.__import__", side_effect=mock_import):
        assert _np_distributions(DummyBackend(), 1) == "dist_exc"
        assert _np_confusion_matrix(DummyBackend(), 1) == "cm_exc"
        assert _np_descriptive_2(DummyBackend(), 1) == "desc_exc"
        assert _np_rem_2(DummyBackend(), 1) == "rem_exc"


def test_np_confusion_matrix_actual_logic() -> None:
    """Test function test_np_confusion_matrix_actual_logic."""
    import numpy as np

    from ml_switcheroo_compiler.backends.eager.reference_math.math_matrix_utils import (
        _np_confusion_matrix,
        _np_confusion_matrix_cap,
    )

    class DummyBackend(MockTestStub):
        """Test class DummyBackend."""

        pass

    res = _np_confusion_matrix(DummyBackend(), [0, 1], [1, 1], num_classes=2)
    assert np.array_equal(ensure_ndarray(res), [[0, 1], [0, 1]])
    res2 = _np_confusion_matrix(DummyBackend(), [0, 1], [1, 1])
    assert extract_test_shape(res2) == (2, 2)
    assert _np_confusion_matrix_cap(DummyBackend(), [0], [0]) is not None


def test_math_matrix_utils_simple_forwards() -> None:
    """Test function test_math_matrix_utils_simple_forwards."""
    from ml_switcheroo_compiler.backends.eager.reference_math.math_matrix_utils import (
        _np_diag_indices_,
        _np_diag_indices_from_,
        _np_diagflat_,
        _np_diagonal_,
        _np_indices_,
        _np_linearoperatorblockdiag,
        _np_linearoperatordiag,
        _np_mask_indices_,
    )

    class DummyBackend(MockTestStub):
        """Test class DummyBackend."""

        def diag_indices(self) -> BackendArg:
            """Test function diag_indices.

            Returns:
                BackendArg: Computed output value from `diag_indices`.
            """
            return 1

        def diag_indices_from(self) -> BackendArg:
            """Test function diag_indices_from.

            Returns:
                BackendArg: Computed output value from `diag_indices_from`.
            """
            return 2

        def diagflat(self) -> BackendArg:
            """Test function diagflat.

            Returns:
                BackendArg: Computed output value from `diagflat`.
            """
            return 3

        def diagonal(self) -> BackendArg:
            """Test function diagonal.

            Returns:
                BackendArg: Computed output value from `diagonal`.
            """
            return 4

        def indices(self) -> BackendArg:
            """Test function indices.

            Returns:
                BackendArg: Computed output value from `indices`.
            """
            return 5

        def mask_indices(self) -> BackendArg:
            """Test function mask_indices.

            Returns:
                BackendArg: Computed output value from `mask_indices`.
            """
            return 6

        def linearoperatorblockdiag(self) -> BackendArg:
            """Test function linearoperatorblockdiag.

            Returns:
                BackendArg: Computed output value from `linearoperatorblockdiag`.
            """
            return 7

        def linearoperatordiag(self) -> BackendArg:
            """Test function linearoperatordiag.

            Returns:
                BackendArg: Computed output value from `linearoperatordiag`.
            """
            return 8

    db = DummyBackend()
    assert _np_diag_indices_(db) == 1
    assert _np_diag_indices_from_(db) == 2
    assert _np_diagflat_(db) == 3
    assert _np_diagonal_(db) == 4
    assert _np_indices_(db) == 5
    assert _np_mask_indices_(db) == 6
    assert type(_np_linearoperatorblockdiag(db)).__name__ == "LinearOperatorBlockDiag"
    assert type(_np_linearoperatordiag(db)).__name__ == "LinearOperatorDiag"
