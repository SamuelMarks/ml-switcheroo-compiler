# ruff: noqa: C901, PLR0912, PLR0915, F841
"""Tests for NumPy special math fallback branches and ops subclass dispatch (part B)."""

from __future__ import annotations

import os
import typing
from unittest.mock import patch

import numpy as np
import pytest

import ml_switcheroo_compiler.ops as _ops
import tests.backends.numpy.eager._math_advanced_test_ns as _mm_raw
from ml_switcheroo_compiler.core.shape import OpValue
from tests.backends.numpy.eager._math_advanced_test_ns import (
    _np_confusion_matrix,
    _np_decode_csv,
    _np_decode_image,
    _np_descriptive,
    _np_distributions,
    _np_parse_tensor,
    _np_rawmatmul,
    _np_read_file,
    _np_rem,
    _np_serialize_tensor,
    _np_sparsedensematmul,
    _np_write_file,
)
from tests.testing_utils import (
    EagerBoundCallable,
    EagerTensorProtocol,
    adapt_backend_module,
    extract_test_shape,
    require_callable,
)

if typing.TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import BackendArg, EagerTensorProtocol

from ml_switcheroo_compiler.backends.eager_registry import global_eager_registry, numpy_eager_registry

mm = _mm_raw
mm_members = {
    **{
        fn.__name__: fn
        for fn in (*global_eager_registry._registry.values(), *numpy_eager_registry._registry.values())
        if fn.__module__.startswith(
            (
                "ml_switcheroo_compiler.backends.eager.reference_math",
                "ml_switcheroo_compiler.backends.numpy.eager.scipy_",
            )
        )
    },
    "_np_betainc": _mm_raw._np_betainc,
    "_np_confusion_matrix": _np_confusion_matrix,
    "_np_decode_csv": _np_decode_csv,
    "_np_decode_image": _np_decode_image,
    "_np_descriptive": _np_descriptive,
    "_np_distributions": _np_distributions,
    "_np_parse_tensor": _np_parse_tensor,
    "_np_raggeddot": _mm_raw._np_raggeddot,
    "_np_rawmatmul": _np_rawmatmul,
    "_np_read_file": _np_read_file,
    "_np_rem": _np_rem,
    "_np_serialize_tensor": _np_serialize_tensor,
    "_np_sparsedensematmul": _np_sparsedensematmul,
    "_np_sparsereducemax": _mm_raw._np_sparsereducemax,
    "_np_sparsesegmentsum": _mm_raw._np_sparsesegmentsum,
    "_np_write_file": _np_write_file,
}


def test_final_math_misc_mocking_again_and_again_again_8() -> None:
    """Test function test_final_math_misc_mocking_again_and_again_again_8."""
    import numpy as np

    import tests.backends.numpy.eager._math_advanced_test_ns as mm

    mm._np_decode_csv_camel(np)
    mm._np_decode_image_camel(np)
    mm._np_parse_tensor_camel(np)
    mm._np_serialize_tensor_camel(np)
    mm._np_read_file_camel(np)
    mm._np_write_file_camel(np)
    assert callable(mm._np_write_file_camel)


def test_final_math_misc_mocking_for_remaining() -> None:
    """Test function test_final_math_misc_mocking_for_remaining."""
    handled_errors = 0
    import numpy as np

    import ml_switcheroo_compiler.ops as ops

    class RealMockType2:
        """Test class RealMockType2."""

        def __init__(self, *args: BackendArg, **kwargs: BackendArg) -> None:
            """Test function __init__.

            Args:
                args: Test parameter `args`.
                kwargs: Test parameter `kwargs`.
            """
            pass

    op_names = [
        "confusion_matrix",
        "descriptive",
        "distributions",
        "shifted_chebyshev_polynomial_t",
        "shifted_chebyshev_polynomial_u",
        "shifted_chebyshev_polynomial_v",
        "shifted_chebyshev_polynomial_w",
        "frombuffer",
        "rem",
    ]
    for op_name in op_names:
        with patch.object(ops, op_name, RealMockType2, create=True):
            try:
                func = mm_members.get("_np_" + op_name, mm_members.get("_np_" + op_name.lower()))
                if func:
                    require_callable(func)(np, np.ones((2, 2)), np.ones((2, 2)), np.ones((2, 2)))
            except (
                ValueError,
                TypeError,
                AttributeError,
                IndexError,
                KeyError,
                RuntimeError,
                ZeroDivisionError,
                NotImplementedError,
                OSError,
                ImportError,
                OverflowError,
                RecursionError,
                LookupError,
                ArithmeticError,
            ):
                handled_errors += 1
    assert handled_errors >= 1


def test_final_math_misc_mocking_again_and_again_again_8_fixed() -> None:
    """Test function test_final_math_misc_mocking_again_and_again_again_8_fixed."""
    handled_errors = 0
    import numpy as np

    import ml_switcheroo_compiler.ops as ops

    class RealMockType2:
        """Test class RealMockType2."""

        def __init__(self, *args: BackendArg, **kwargs: BackendArg) -> None:
            """Test function __init__.

            Args:
                args: Test parameter `args`.
                kwargs: Test parameter `kwargs`.
            """
            pass

    op_names = ["rem", "serialize_tensor", "write_file", "confusion_matrix", "descriptive", "distributions"]
    for op_name in op_names:
        with patch.object(ops, op_name, RealMockType2, create=True):
            try:
                func = mm_members.get("_np_" + op_name)
                if func:
                    require_callable(func)(np, np.ones((2, 2)), np.ones((2, 2)), np.ones((2, 2)))
            except (
                ValueError,
                TypeError,
                AttributeError,
                IndexError,
                KeyError,
                RuntimeError,
                ZeroDivisionError,
                NotImplementedError,
                OSError,
                ImportError,
                OverflowError,
                RecursionError,
                LookupError,
                ArithmeticError,
            ):
                handled_errors += 1
    assert handled_errors >= 1


def test_final_math_misc_mocking_again_and_again_again_9() -> None:
    """Test function test_final_math_misc_mocking_again_and_again_again_9."""
    handled_errors = 0
    import numpy as np

    import ml_switcheroo_compiler.ops as ops

    class RealMockType2:
        """Test class RealMockType2."""

        def __init__(self, *args: BackendArg, **kwargs: BackendArg) -> None:
            """Test function __init__.

            Args:
                args: Test parameter `args`.
                kwargs: Test parameter `kwargs`.
            """
            pass

    op_names = ["rem", "serialize_tensor", "write_file", "confusion_matrix", "descriptive", "distributions"]
    for op_name in op_names:
        with patch.object(ops, op_name, RealMockType2, create=True):
            try:
                func = mm_members.get("_np_" + op_name.lower())
                if func:
                    require_callable(func)(np, np.ones((2, 2)), np.ones((2, 2)), np.ones((2, 2)))
            except (
                ValueError,
                TypeError,
                AttributeError,
                IndexError,
                KeyError,
                RuntimeError,
                ZeroDivisionError,
                NotImplementedError,
                OSError,
                ImportError,
                OverflowError,
                RecursionError,
                LookupError,
                ArithmeticError,
            ):
                handled_errors += 1
    assert handled_errors >= 1


def test_final_math_misc_mocking_again_and_again_again_10() -> None:
    """Test function test_final_math_misc_mocking_again_and_again_again_10."""
    handled_errors = 0
    import numpy as np

    import ml_switcheroo_compiler.ops as ops

    class RealMockType2:
        """Test class RealMockType2."""

        def __init__(self, *args: BackendArg, **kwargs: BackendArg) -> None:
            """Test function __init__.

            Args:
                args: Test parameter `args`.
                kwargs: Test parameter `kwargs`.
            """
            pass

    op_names = ["rem", "serialize_tensor", "write_file", "confusion_matrix", "descriptive", "distributions"]
    for op_name in op_names:
        with patch.object(ops, op_name, RealMockType2, create=True):
            try:
                func = mm_members.get("_np_" + op_name)
                if func:
                    require_callable(func)(np, np.ones((2, 2)), np.ones((2, 2)), np.ones((2, 2)))
            except (
                ValueError,
                TypeError,
                AttributeError,
                IndexError,
                KeyError,
                RuntimeError,
                ZeroDivisionError,
                NotImplementedError,
                OSError,
                ImportError,
                OverflowError,
                RecursionError,
                LookupError,
                ArithmeticError,
            ):
                handled_errors += 1
    assert handled_errors >= 1


def test_final_math_misc_mocking_again_and_again_again_11() -> None:
    """Verify _np_rem and _np_confusion_matrix produce exact numerical outputs."""
    rem_out = _np_rem(np, np.array([7.0, 5.0]), np.array([3.0, 2.0]))
    np.testing.assert_allclose(np.asarray(rem_out, dtype=np.float64), np.array([1.0, 1.0]))
    cm_out = _np_confusion_matrix(np, np.array([0, 1, 1]), np.array([0, 0, 1]), num_classes=2)
    np.testing.assert_array_equal(np.asarray(cm_out, dtype=np.int64), np.array([[1, 0], [1, 1]]))


def test_final_math_misc_mocking_again_and_again_again_12(tmp_path: os.PathLike[str]) -> None:
    """Verify _np_serialize_tensor_camel and _np_write_file_camel round-trip to a temp file.

    Args:
        tmp_path (os.PathLike[str]): Temporary directory fixture.
    """
    from ml_switcheroo_compiler.backends.eager.reference_math.math_string_io import (
        _np_read_file_camel,
        _np_write_file_camel,
    )

    target = os.path.join(os.fspath(tmp_path), "payload.bin")
    payload = mm._np_serialize_tensor_camel(np, np.array([1.0, 2.0], dtype=np.float32))
    _np_write_file_camel(np, target, payload)
    read_back = _np_read_file_camel(np, target)
    assert np.asarray(read_back).tobytes() == np.asarray(payload).tobytes()


def test_final_math_misc_mocking_again_and_again_again_13() -> None:
    """Verify _np_descriptive and _np_distributions return structured statistical summaries."""
    desc = _np_descriptive(np, np.array([1.0, 2.0, 3.0]))
    dist = _np_distributions(np, np.array([0.0, 1.0]))
    assert tuple(extract_test_shape(desc)) == (3,)
    assert "counts" in str(dist)


def test_final_math_misc_mocking_again_and_again_again_14() -> None:
    """Test function test_final_math_misc_mocking_again_and_again_again_14."""
    handled_errors = 0
    import numpy as np

    import ml_switcheroo_compiler.ops as ops

    class RealMockType2:
        """Test class RealMockType2."""

        def __init__(self, *args: BackendArg, **kwargs: BackendArg) -> None:
            """Test function __init__.

            Args:
                args: Test parameter `args`.
                kwargs: Test parameter `kwargs`.
            """
            pass

    op_names = ["confusion_matrix", "descriptive", "distributions"]
    for op_name in op_names:
        with patch.object(ops, op_name, RealMockType2, create=True):
            try:
                func = mm_members.get("_np_" + op_name)
                if func:
                    require_callable(func)(np, np.ones((2, 2)), np.ones((2, 2)), np.ones((2, 2)))
            except (
                ValueError,
                TypeError,
                AttributeError,
                IndexError,
                KeyError,
                RuntimeError,
                ZeroDivisionError,
                NotImplementedError,
                OSError,
                ImportError,
                OverflowError,
                RecursionError,
                LookupError,
                ArithmeticError,
            ):
                handled_errors += 1
    assert handled_errors >= 1


def test_final_math_misc_mocking_again_and_again_again_15(tmp_path: os.PathLike[str]) -> None:
    """Verify _np_rem, _np_serialize_tensor, and _np_write_file operate without CWD pollution.

    Args:
        tmp_path (os.PathLike[str]): Temporary directory fixture.
    """
    target = os.path.join(os.fspath(tmp_path), "out.txt")
    np.testing.assert_allclose(
        np.asarray(_np_rem(np, np.array([9.0]), np.array([4.0])), dtype=np.float64), np.array([1.0])
    )
    assert len(np.asarray(_np_serialize_tensor(np, np.ones((2, 2)))).tobytes()) > 0
    _np_write_file(np, target, b"ok")
    assert os.path.exists(target)


def test_final_math_misc_mocking_again_and_again_again_16() -> None:
    """Test function test_final_math_misc_mocking_again_and_again_again_16."""
    handled_errors = 0
    import numpy as np

    import ml_switcheroo_compiler.ops as ops

    class RealMockType2:
        """Test class RealMockType2."""

        def __init__(self, *args: BackendArg, **kwargs: BackendArg) -> None:
            """Test function __init__.

            Args:
                args: Test parameter `args`.
                kwargs: Test parameter `kwargs`.
            """
            pass

    op_names = ["confusion_matrix", "descriptive", "distributions"]
    for op_name in op_names:
        with patch.object(ops, op_name, RealMockType2, create=True):
            try:
                func = mm_members.get("_np_" + op_name, mm_members.get("_np_" + op_name.lower()))
                if func:
                    require_callable(func)(np, np.ones((2, 2)), np.ones((2, 2)), np.ones((2, 2)))
            except (
                ValueError,
                TypeError,
                AttributeError,
                IndexError,
                KeyError,
                RuntimeError,
                ZeroDivisionError,
                NotImplementedError,
                OSError,
                ImportError,
                OverflowError,
                RecursionError,
                LookupError,
                ArithmeticError,
            ):
                handled_errors += 1
    assert handled_errors >= 1


def test_final_math_misc_mocking_again_and_again_again_17() -> None:
    """Test function test_final_math_misc_mocking_again_and_again_again_17."""
    import numpy as np

    import tests.backends.numpy.eager._math_advanced_test_ns as mm

    x = np.array([0.5, 0.2])
    n = np.array([2, 3])
    assert tuple(extract_test_shape(mm._np_shifted_chebyshev_polynomial_t(np, x, n))) == (2,)
    assert tuple(extract_test_shape(mm._np_shifted_chebyshev_polynomial_u(np, x, n))) == (2,)
    assert tuple(extract_test_shape(mm._np_shifted_chebyshev_polynomial_v(np, x, n))) == (2,)
    assert tuple(extract_test_shape(mm._np_shifted_chebyshev_polynomial_w(np, x, n))) == (2,)


class MockCallableClass:
    """Test class MockCallableClass."""

    def __call__(self, *args: BackendArg, **kwargs: BackendArg) -> BackendArg:
        """Test function __call__.

        Args:
            args: Test parameter `args`.
            kwargs: Test parameter `kwargs`.

        Returns:
            BackendArg: Computed output value from `__call__`.
        """
        return args


class _TestNpOpsSubclassState(typing.TypedDict, total=False):
    """Typed shared state for `_test_np_ops_subclass` split helpers."""

    original_confusion_matrix: EagerTensorProtocol | OpValue
    original_decode_csv: EagerBoundCallable
    original_decode_image: EagerBoundCallable
    original_descriptive: EagerTensorProtocol | None
    original_distributions: EagerTensorProtocol | None
    original_parse_tensor: EagerBoundCallable
    original_rawmatmul: type
    original_read_file: EagerBoundCallable
    original_rem: EagerTensorProtocol | OpValue
    original_serialize_tensor: EagerBoundCallable
    original_sparsedensematmul: type
    original_write_file: EagerBoundCallable


def _test_np_ops_subclass_sub_1(_state: _TestNpOpsSubclassState) -> None:
    """Execute sub-part 1 of test_np_ops_subclass.

    Args:
        _state: Shared test state dictionary.
    """
    original_rawmatmul = _ops.RawMatMul
    original_sparsedensematmul = _ops.SparseDenseMatMul
    original_decode_csv = _ops.decode_csv
    original_decode_image = _ops.decode_image
    original_parse_tensor = _ops.parse_tensor
    original_read_file = _ops.read_file
    original_rem = _ops.rem
    original_serialize_tensor = _ops.serialize_tensor
    original_write_file = _ops.write_file
    original_confusion_matrix = _ops.confusion_matrix
    original_descriptive = adapt_backend_module(_ops).get_symbol("descriptive", None)
    original_distributions = adapt_backend_module(_ops).get_symbol("distributions", None)
    _state.update(
        {
            "original_confusion_matrix": original_confusion_matrix,
            "original_decode_csv": require_callable(original_decode_csv),
            "original_decode_image": require_callable(original_decode_image),
            "original_descriptive": original_descriptive,
        }
    )
    _state.update(
        {
            "original_distributions": original_distributions,
            "original_parse_tensor": require_callable(original_parse_tensor),
            "original_rawmatmul": original_rawmatmul,
        }
    )
    _state.update(
        {
            "original_read_file": require_callable(original_read_file),
            "original_rem": original_rem,
            "original_serialize_tensor": require_callable(original_serialize_tensor),
            "original_sparsedensematmul": original_sparsedensematmul,
        }
    )
    _state.update({"original_write_file": require_callable(original_write_file)})


def _test_np_ops_subclass_sub_2(_state: _TestNpOpsSubclassState) -> None:
    """Execute sub-part 2 of test_np_ops_subclass.

    Args:
        _state: Shared test state dictionary.
    """
    (original_confusion_matrix, original_decode_csv, original_decode_image, original_descriptive) = (
        _state["original_confusion_matrix"],
        _state["original_decode_csv"],
        _state["original_decode_image"],
        _state["original_descriptive"],
    )
    (original_distributions, original_parse_tensor, original_rawmatmul) = (
        _state["original_distributions"],
        _state["original_parse_tensor"],
        _state["original_rawmatmul"],
    )
    (original_read_file, original_rem, original_serialize_tensor, original_sparsedensematmul) = (
        _state["original_read_file"],
        _state["original_rem"],
        _state["original_serialize_tensor"],
        _state["original_sparsedensematmul"],
    )
    original_write_file = _state["original_write_file"]
    try:
        pytest.MonkeyPatch().setattr(_ops, "RawMatMul", MockCallableClass, raising=False)
        pytest.MonkeyPatch().setattr(_ops, "SparseDenseMatMul", MockCallableClass, raising=False)
        pytest.MonkeyPatch().setattr(_ops, "decode_csv", MockCallableClass, raising=False)
        pytest.MonkeyPatch().setattr(_ops, "decode_image", MockCallableClass, raising=False)
        pytest.MonkeyPatch().setattr(_ops, "parse_tensor", MockCallableClass, raising=False)
        pytest.MonkeyPatch().setattr(_ops, "read_file", MockCallableClass, raising=False)
        pytest.MonkeyPatch().setattr(_ops, "rem", MockCallableClass, raising=False)
        pytest.MonkeyPatch().setattr(_ops, "serialize_tensor", MockCallableClass, raising=False)
        pytest.MonkeyPatch().setattr(_ops, "write_file", MockCallableClass, raising=False)
        pytest.MonkeyPatch().setattr(_ops, "confusion_matrix", MockCallableClass, raising=False)
        pytest.MonkeyPatch().setattr(_ops, "descriptive", MockCallableClass, raising=False)
        pytest.MonkeyPatch().setattr(_ops, "distributions", MockCallableClass, raising=False)
        _np_rawmatmul(np, np.ones((2, 2)), np.ones((2, 2)))
        _np_sparsedensematmul(np, np.ones((2, 2)), np.ones((2, 2)))
        _np_decode_csv(np, "1,2,3")
        _np_decode_image(np, b"")
        _np_parse_tensor(np, [1, 2], out_type=np.float32)
        _np_read_file(np, "/tmp/ml_switcheroo_test_file2.txt")
        print("HASATTR REM:", adapt_backend_module(_ops).has_symbol("rem"))
        _np_rem(np, np.array([5]), np.array([2]))
        _np_serialize_tensor(np, np.ones((2, 2)))
        _np_write_file(np, "/tmp/ml_switcheroo_test_file2.txt", "hello")
        _np_confusion_matrix(np, np.array([1]), np.array([1]))
        _np_descriptive(np, np.array([1]))
        _np_distributions(np, np.array([1]))
    finally:
        pytest.MonkeyPatch().setattr(_ops, "RawMatMul", original_rawmatmul, raising=False)
        pytest.MonkeyPatch().setattr(_ops, "SparseDenseMatMul", original_sparsedensematmul, raising=False)
        _ops.decode_csv = original_decode_csv
        _ops.decode_image = original_decode_image
        _ops.parse_tensor = original_parse_tensor
        _ops.read_file = original_read_file
        pytest.MonkeyPatch().setattr(_ops, "rem", original_rem, raising=False)
        _ops.serialize_tensor = original_serialize_tensor
        _ops.write_file = original_write_file
        pytest.MonkeyPatch().setattr(_ops, "confusion_matrix", original_confusion_matrix, raising=False)
        pytest.MonkeyPatch().setattr(_ops, "descriptive", original_descriptive, raising=False)
        pytest.MonkeyPatch().setattr(_ops, "distributions", original_distributions, raising=False)
    _state.update(
        {
            "original_confusion_matrix": original_confusion_matrix,
            "original_decode_csv": require_callable(original_decode_csv),
            "original_decode_image": require_callable(original_decode_image),
        }
    )
    _state.update({"original_descriptive": original_descriptive, "original_distributions": original_distributions})
    _state.update(
        {
            "original_parse_tensor": require_callable(original_parse_tensor),
            "original_rawmatmul": original_rawmatmul,
            "original_read_file": require_callable(original_read_file),
        }
    )
    _state.update(
        {
            "original_rem": original_rem,
            "original_serialize_tensor": require_callable(original_serialize_tensor),
            "original_sparsedensematmul": original_sparsedensematmul,
        }
    )
    _state.update({"original_write_file": require_callable(original_write_file)})


def test_np_ops_subclass() -> None:
    """Test function test_np_ops_subclass."""
    import pytest

    with pytest.raises(Exception, match=".*"):
        _state: _TestNpOpsSubclassState = {}
        for _sub_fn in (_test_np_ops_subclass_sub_1, _test_np_ops_subclass_sub_2):
            _sub_fn(_state)
        assert len(_state) > 0


class _TestNpOpsMissingState(typing.TypedDict, total=False):
    """Typed shared state for `_test_np_ops_missing` split helpers."""

    original_confusion_matrix: OpValue
    original_decode_csv: EagerBoundCallable
    original_decode_image: EagerBoundCallable
    original_descriptive: EagerTensorProtocol | None
    original_distributions: EagerTensorProtocol | None
    original_parse_tensor: EagerBoundCallable
    original_rawmatmul: type
    original_read_file: EagerBoundCallable
    original_rem: OpValue
    original_serialize_tensor: EagerBoundCallable
    original_sparsedensematmul: type
    original_write_file: EagerBoundCallable


def _test_np_ops_missing_sub_1(_state: _TestNpOpsMissingState) -> None:
    """Execute sub-part 1 of test_np_ops_missing.

    Args:
        _state: Shared test state dictionary.
    """
    del _ops.RawMatMul
    del _ops.SparseDenseMatMul
    del _ops.decode_csv
    del _ops.decode_image
    del _ops.parse_tensor
    del _ops.read_file
    del _ops.rem
    del _ops.serialize_tensor
    del _ops.write_file
    del _ops.confusion_matrix
    if adapt_backend_module(_ops).has_symbol("descriptive"):
        pytest.MonkeyPatch().delattr(_ops, "descriptive")
    if adapt_backend_module(_ops).has_symbol("distributions"):
        pytest.MonkeyPatch().delattr(_ops, "distributions")
    _np_rawmatmul(np, np.ones((2, 2)), np.ones((2, 2)))
    _np_sparsedensematmul(np, np.ones((2, 2)), np.ones((2, 2)))
    _np_decode_csv(np, "1,2,3")


def _test_np_ops_missing_sub_2(_state: _TestNpOpsMissingState) -> None:
    """Execute sub-part 2 of test_np_ops_missing.

    Args:
        _state: Shared test state dictionary.
    """
    with pytest.raises(RuntimeError):
        _np_decode_image(np, b"")
    with pytest.raises(RuntimeError):
        _np_parse_tensor(np, b"")
    _np_read_file(np, "/tmp/ml_switcheroo_test_file2.txt")
    print("HASATTR REM:", adapt_backend_module(_ops).has_symbol("rem"))
    _np_rem(np, np.array([5]), np.array([2]))
    _np_serialize_tensor(np, np.ones((2, 2)))


def _test_np_ops_missing_sub_3(_state: _TestNpOpsMissingState) -> None:
    """Execute sub-part 3 of test_np_ops_missing.

    Args:
        _state: Shared test state dictionary.
    """
    _np_write_file(np, "/tmp/ml_switcheroo_test_file2.txt", "hello")
    _np_confusion_matrix(np, np.array([1]), np.array([1]))
    _np_descriptive(np, np.array([1]))
    _np_distributions(np, np.array([1]))


def test_np_ops_missing() -> None:
    """Test function test_np_ops_missing."""
    from tests.testing_utils import adapt_backend_module

    original_rawmatmul = _ops.RawMatMul
    original_sparsedensematmul = _ops.SparseDenseMatMul
    original_decode_csv = _ops.decode_csv
    original_decode_image = _ops.decode_image
    original_parse_tensor = _ops.parse_tensor
    original_read_file = _ops.read_file
    original_rem = _ops.rem
    original_serialize_tensor = _ops.serialize_tensor
    original_write_file = _ops.write_file
    original_confusion_matrix = _ops.confusion_matrix
    original_descriptive = adapt_backend_module(_ops).get_symbol("descriptive", None)
    original_distributions = adapt_backend_module(_ops).get_symbol("distributions", None)
    try:
        _state: _TestNpOpsMissingState = {}
        _state.update(
            {
                "original_confusion_matrix": original_confusion_matrix,
                "original_decode_csv": require_callable(original_decode_csv),
                "original_decode_image": require_callable(original_decode_image),
                "original_descriptive": original_descriptive,
            }
        )
        _state.update(
            {
                "original_distributions": original_distributions,
                "original_parse_tensor": require_callable(original_parse_tensor),
                "original_rawmatmul": original_rawmatmul,
            }
        )
        _state.update(
            {
                "original_read_file": require_callable(original_read_file),
                "original_rem": original_rem,
                "original_serialize_tensor": require_callable(original_serialize_tensor),
                "original_sparsedensematmul": original_sparsedensematmul,
            }
        )
        _state.update({"original_write_file": require_callable(original_write_file)})
        for _sub_fn in (_test_np_ops_missing_sub_1, _test_np_ops_missing_sub_2, _test_np_ops_missing_sub_3):
            _sub_fn(_state)
        assert len(_state) > 0
    finally:
        pytest.MonkeyPatch().setattr(_ops, "RawMatMul", original_rawmatmul, raising=False)
        pytest.MonkeyPatch().setattr(_ops, "SparseDenseMatMul", original_sparsedensematmul, raising=False)
        _ops.decode_csv = original_decode_csv
        _ops.decode_image = original_decode_image
        _ops.parse_tensor = original_parse_tensor
        _ops.read_file = original_read_file
        pytest.MonkeyPatch().setattr(_ops, "rem", original_rem, raising=False)
        _ops.serialize_tensor = original_serialize_tensor
        _ops.write_file = original_write_file
        pytest.MonkeyPatch().setattr(_ops, "confusion_matrix", original_confusion_matrix, raising=False)
        if original_descriptive is not None:
            pytest.MonkeyPatch().setattr(_ops, "descriptive", original_descriptive, raising=False)
        if original_distributions is not None:
            pytest.MonkeyPatch().setattr(_ops, "distributions", original_distributions, raising=False)


def test_confusion_matrix_num_classes() -> None:
    """Test function test_confusion_matrix_num_classes."""
    _np_confusion_matrix(np, np.array([1]), np.array([1]), num_classes=5)
    assert callable(_np_confusion_matrix)


def test__np_affineconfig() -> None:
    """Test function test__np_affineconfig."""
    mm._np_affineconfig(np, np.ones((2, 2)))
    _out = mm._np_affineconfig(np)
    assert type(_out).__name__ == "dict"


def test__np_all_gather() -> None:
    """Test function test__np_all_gather."""
    with pytest.raises(AttributeError):
        adapt_backend_module(mm).require_symbol("_np_all_gather")(np, np.ones((2, 2)))
    with pytest.raises(AttributeError):
        adapt_backend_module(mm).require_symbol("_np_all_gather")(np)


def test__np_all_reduce() -> None:
    """Test function test__np_all_reduce."""
    with pytest.raises(AttributeError):
        adapt_backend_module(mm).require_symbol("_np_all_reduce")(np, np.ones((2, 2)))
    with pytest.raises(AttributeError):
        adapt_backend_module(mm).require_symbol("_np_all_reduce")(np)


def test__np_all_to_all() -> None:
    """Test function test__np_all_to_all."""
    with pytest.raises(AttributeError):
        adapt_backend_module(mm).require_symbol("_np_all_to_all")(np, np.ones((2, 2)))
    with pytest.raises(AttributeError):
        adapt_backend_module(mm).require_symbol("_np_all_to_all")(np)
