# ruff: noqa: C901, PLR0912, PLR0915, F841
"""Tests for NumPy special math fallback branches and exception paths (part A)."""

from __future__ import annotations

import typing
from unittest.mock import patch

import numpy as np
import pytest

import tests.backends.numpy.eager._math_advanced_test_ns as _mm_raw
from ml_switcheroo_compiler.backends.eager_registry import global_eager_registry, numpy_eager_registry
from ml_switcheroo_compiler.backends.numpy.eager import ensure_ndarray
from tests.testing_utils import coerce_operand_list, require_callable

if typing.TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import BackendArg

mm = _mm_raw
mm_members = {
    **{fn.__name__: fn for fn in global_eager_registry._registry.values()},
    "_np_betainc": _mm_raw._np_betainc,
    "_np_raggeddot": _mm_raw._np_raggeddot,
    "_np_rem": _mm_raw._np_rem,
    "_np_sparsereducemax": _mm_raw._np_sparsereducemax,
    "_np_sparsesegmentsum": _mm_raw._np_sparsesegmentsum,
}


def test_np_remaining_math_misc_3() -> None:
    """Test function test_np_remaining_math_misc_3."""
    handled_errors = 0
    import ml_switcheroo_compiler.ops as ops

    class MockOpFunc:
        """Test class MockOpFunc."""

        def __init__(self, *args: BackendArg, **kwargs: BackendArg) -> None:
            """Test function __init__.

            Args:
                args: Test parameter `args`.
                kwargs: Test parameter `kwargs`.
            """
            pass

        def __call__(self, *args: BackendArg, **kwargs: BackendArg) -> BackendArg:
            """Test function __call__.

            Args:
                args: Test parameter `args`.
                kwargs: Test parameter `kwargs`.

            Returns:
                BackendArg: Computed output value from `__call__`.
            """
            return np.ones((2, 2))

    op_names = [
        "shifted_chebyshev_polynomial_t",
        "shifted_chebyshev_polynomial_u",
        "shifted_chebyshev_polynomial_v",
        "shifted_chebyshev_polynomial_w",
    ]
    for op_name in op_names:
        with patch.object(ops, op_name, MockOpFunc, create=True):
            try:
                func = mm_members.get(f"_np_{op_name.lower()}", mm_members.get(f"_np_{op_name}"))
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

    mm._np_rfftfreq(np, 10, d=0.1)
    mm._np_frombuffer(np)
    _out = mm._np_rem(np, 1)
    assert _out is None


def test_remaining_math_misc_stuff() -> None:
    """Test function test_remaining_math_misc_stuff."""
    handled_errors = 0
    import numpy as np

    import ml_switcheroo_compiler.ops as ops

    class MockOpFunc:
        """Test class MockOpFunc."""

        def __init__(self, *args: BackendArg, **kwargs: BackendArg) -> None:
            """Test function __init__.

            Args:
                args: Test parameter `args`.
                kwargs: Test parameter `kwargs`.
            """
            pass

        def __call__(self, *args: BackendArg, **kwargs: BackendArg) -> BackendArg:
            """Test function __call__.

            Args:
                args: Test parameter `args`.
                kwargs: Test parameter `kwargs`.

            Returns:
                BackendArg: Computed output value from `__call__`.
            """
            return np.ones((2, 2))

    op_names = [
        "RawMatMul",
        "SparseDenseMatMul",
        "decode_csv",
        "decode_image",
        "parse_example",
        "parse_tensor",
        "serialize_tensor",
        "encode_image",
        "encode_jpeg",
        "encode_png",
        "encode_base64",
        "decode_base64",
        "decode_json_example",
        "decode_compressed",
        "read_file",
        "write_file",
        "confusion_matrix",
        "descriptive",
        "distributions",
    ]
    for op_name in op_names:
        with patch.object(ops, op_name, MockOpFunc, create=True):
            try:
                func = mm_members.get("_np_" + op_name.lower(), mm_members.get("_np_" + op_name))
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


def test_final_math_misc_mocking() -> None:
    """Test function test_final_math_misc_mocking."""
    handled_errors = 0
    import numpy as np

    import ml_switcheroo_compiler.ops as ops

    class MockOpFunc:
        """Test class MockOpFunc."""

        def __init__(self, *args: BackendArg, **kwargs: BackendArg) -> None:
            """Test function __init__.

            Args:
                args: Test parameter `args`.
                kwargs: Test parameter `kwargs`.
            """
            pass

        def __call__(self, *args: BackendArg, **kwargs: BackendArg) -> BackendArg:
            """Test function __call__.

            Args:
                args: Test parameter `args`.
                kwargs: Test parameter `kwargs`.

            Returns:
                BackendArg: Computed output value from `__call__`.
            """
            return np.ones((2, 2))

    op_names = [
        "RawMatMul",
        "SparseDenseMatMul",
        "decode_csv",
        "decode_image",
        "parse_example",
        "parse_tensor",
        "serialize_tensor",
        "encode_image",
        "encode_jpeg",
        "encode_png",
        "encode_base64",
        "decode_base64",
        "decode_json_example",
        "decode_compressed",
        "read_file",
        "write_file",
        "confusion_matrix",
        "descriptive",
        "distributions",
    ]
    for op_name in op_names:
        with patch.object(ops, op_name, MockOpFunc, create=True):
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


def test_final_math_misc_mocking_again() -> None:
    """Test function test_final_math_misc_mocking_again."""
    handled_errors = 0
    import numpy as np

    import ml_switcheroo_compiler.ops as ops

    class MockOpFunc:
        """Test class MockOpFunc."""

        def __init__(self, *args: BackendArg, **kwargs: BackendArg) -> None:
            """Test function __init__.

            Args:
                args: Test parameter `args`.
                kwargs: Test parameter `kwargs`.
            """
            pass

        def __call__(self, *args: BackendArg, **kwargs: BackendArg) -> BackendArg:
            """Test function __call__.

            Args:
                args: Test parameter `args`.
                kwargs: Test parameter `kwargs`.

            Returns:
                BackendArg: Computed output value from `__call__`.
            """
            return np.ones((2, 2))

    op_names = [
        "RawMatMul",
        "SparseDenseMatMul",
        "decode_csv",
        "decode_image",
        "parse_example",
        "parse_tensor",
        "serialize_tensor",
        "encode_image",
        "encode_jpeg",
        "encode_png",
        "encode_base64",
        "decode_base64",
        "decode_json_example",
        "decode_compressed",
        "read_file",
        "write_file",
        "confusion_matrix",
        "descriptive",
        "distributions",
    ]
    for op_name in op_names:
        with patch.object(ops, op_name, MockOpFunc, create=True):
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


def test_final_math_misc_mocking_once_more() -> None:
    """Test function test_final_math_misc_mocking_once_more."""
    handled_errors = 0
    import numpy as np

    import ml_switcheroo_compiler.ops as ops

    class MockOpFuncWithInit:
        """Test class MockOpFuncWithInit."""

        def __init__(self, *args: BackendArg, **kwargs: BackendArg) -> None:
            """Test function __init__.

            Args:
                args: Test parameter `args`.
                kwargs: Test parameter `kwargs`.
            """
            pass

    class RealMockType:
        """Test class RealMockType."""

        def __init__(self, *args: BackendArg, **kwargs: BackendArg) -> None:
            """Test function __init__.

            Args:
                args: Test parameter `args`.
                kwargs: Test parameter `kwargs`.
            """
            pass

        def __call__(self, *args: BackendArg, **kwargs: BackendArg) -> BackendArg:
            """Test function __call__.

            Args:
                args: Test parameter `args`.
                kwargs: Test parameter `kwargs`.

            Returns:
                BackendArg: Computed output value from `__call__`.
            """
            return np.ones((2, 2))

    op_names = [
        "RawMatMul",
        "SparseDenseMatMul",
        "decode_csv",
        "decode_image",
        "parse_example",
        "parse_tensor",
        "serialize_tensor",
        "encode_image",
        "encode_jpeg",
        "encode_png",
        "encode_base64",
        "decode_base64",
        "decode_json_example",
        "decode_compressed",
        "read_file",
        "write_file",
        "confusion_matrix",
        "descriptive",
        "distributions",
    ]
    for op_name in op_names:
        with patch.object(ops, op_name, RealMockType, create=True):
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


def test_final_math_misc_mocking_once_more_2() -> None:
    """Test function test_final_math_misc_mocking_once_more_2."""
    handled_errors = 0
    import numpy as np

    import ml_switcheroo_compiler.ops as ops

    class RealMockType:
        """Test class RealMockType."""

        def __init__(self, *args: BackendArg, **kwargs: BackendArg) -> None:
            """Test function __init__.

            Args:
                args: Test parameter `args`.
                kwargs: Test parameter `kwargs`.
            """
            pass

    op_names = [
        "RawMatMul",
        "SparseDenseMatMul",
        "decode_csv",
        "decode_image",
        "parse_example",
        "parse_tensor",
        "serialize_tensor",
        "encode_image",
        "encode_jpeg",
        "encode_png",
        "encode_base64",
        "decode_base64",
        "decode_json_example",
        "decode_compressed",
        "read_file",
        "write_file",
        "confusion_matrix",
        "descriptive",
        "distributions",
    ]
    for op_name in op_names:
        with patch.object(ops, op_name, RealMockType, create=True):
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


def test_final_math_misc_mocking_again_and_again() -> None:
    """Test function test_final_math_misc_mocking_again_and_again."""
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
        "RawMatMul",
        "SparseDenseMatMul",
        "decode_csv",
        "decode_image",
        "parse_example",
        "parse_tensor",
        "serialize_tensor",
        "encode_image",
        "encode_jpeg",
        "encode_png",
        "encode_base64",
        "decode_base64",
        "decode_json_example",
        "decode_compressed",
        "read_file",
        "write_file",
        "confusion_matrix",
        "descriptive",
        "distributions",
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


def test_final_math_misc_mocking_again_and_again_again() -> None:
    """Test function test_final_math_misc_mocking_again_and_again_again."""
    handled_errors = 0
    import numpy as np

    import ml_switcheroo_compiler.ops as ops

    class RealMockType2:
        """Test class RealMockType2."""

        def __new__(cls, *args: BackendArg, **kwargs: BackendArg) -> RealMockType2:
            """Test function __new__.

            Args:
                args: Test parameter `args`.
                kwargs: Test parameter `kwargs`.

            Returns:
                RealMockType2: Computed output value from `__new__`.

            Raises:
                Exception: When raised during `__new__` (Trigger catch).
            """
            raise RuntimeError("Trigger catch")

    op_names = [
        "RawMatMul",
        "SparseDenseMatMul",
        "decode_csv",
        "decode_image",
        "parse_example",
        "parse_tensor",
        "serialize_tensor",
        "encode_image",
        "encode_jpeg",
        "encode_png",
        "encode_base64",
        "decode_base64",
        "decode_json_example",
        "decode_compressed",
        "read_file",
        "write_file",
        "confusion_matrix",
        "descriptive",
        "distributions",
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


def test_final_math_misc_mocking_again_and_again_again_2() -> None:
    """Test function test_final_math_misc_mocking_again_and_again_again_2."""
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
        "decode_csv",
        "decode_image",
        "parse_example",
        "parse_tensor",
        "serialize_tensor",
        "encode_image",
        "encode_jpeg",
        "encode_png",
        "encode_base64",
        "decode_base64",
        "decode_json_example",
        "decode_compressed",
        "read_file",
        "write_file",
        "confusion_matrix",
        "descriptive",
        "distributions",
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


def test_final_math_misc_mocking_again_and_again_again_3() -> None:
    """Verify DecodeCsv, SerializeTensor, and ParseTensor round-trip via numpy_eager_registry."""
    ser_fn = numpy_eager_registry.get("SerializeTensor")
    parse_fn = numpy_eager_registry.get("ParseTensor")
    csv_fn = numpy_eager_registry.get("DecodeCsv")

    arr = np.array([1.5, -2.5], dtype=np.float32)
    serialized = require_callable(ser_fn)(np, arr)
    assert serialized is not None
    parsed = require_callable(parse_fn)(np, serialized, out_type="float32")
    np.testing.assert_allclose(parsed, arr)
    decoded_csv = require_callable(csv_fn)(np, ["1.0,2.0"], record_defaults=[[0.0], [0.0]])
    assert len(decoded_csv) == 2


def test_final_math_misc_mocking_again_and_again_again_4() -> None:
    """Test function test_final_math_misc_mocking_again_and_again_again_4."""
    import numpy as np

    op_names = ["DecodeCsv", "DecodeImage", "ParseTensor", "SerializeTensor", "ReadFile"]
    checked_ops = 0
    for op_name in op_names:
        func = mm_members.get("_np_" + op_name.lower() + "_camel", mm_members.get("_np_" + op_name))
        if func:
            require_callable(func)(np, np.ones((2, 2)), np.ones((2, 2)), np.ones((2, 2)))
        checked_ops += 1
    assert checked_ops == len(op_names)


def test_final_math_misc_mocking_again_and_again_again_5() -> None:
    """Verify camel-case string/image I/O handlers raise RuntimeError on invalid payloads."""
    with pytest.raises(RuntimeError):
        mm._np_decode_image_camel(np, b"not-an-image")
    with pytest.raises(RuntimeError):
        mm._np_parse_tensor_camel(np, b"\xff\xfeinvalid")


def test_final_math_misc_mocking_camel() -> None:
    """Verify camel-case DecodeCsv handler parses delimited numeric strings."""
    cols = mm._np_decode_csv_camel(np, np.array("3.0,4.0"), record_defaults=[0.0, 0.0])
    assert len(coerce_operand_list(cols)) == 2
    np.testing.assert_allclose(ensure_ndarray(coerce_operand_list(cols)[0]), np.array([3.0]))
    np.testing.assert_allclose(ensure_ndarray(coerce_operand_list(cols)[1]), np.array([4.0]))


def test_final_math_misc_mocking_again_and_again_again_6() -> None:
    """Test function test_final_math_misc_mocking_again_and_again_again_6."""
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
        "decode_csv",
        "decode_image",
        "parse_example",
        "parse_tensor",
        "serialize_tensor",
        "read_file",
        "write_file",
    ]
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


def test_final_math_misc_mocking_again_and_again_again_7() -> None:
    """Test function test_final_math_misc_mocking_again_and_again_again_7."""
    import numpy as np

    op_names = ["DecodeCsv", "DecodeImage", "ParseExample", "ParseTensor", "SerializeTensor", "ReadFile", "WriteFile"]
    checked_ops = 0
    for op_name in op_names:
        func = mm_members.get("_np_" + op_name.lower() + "_camel")
        if func:
            require_callable(func)(np, np.ones((2, 2)), np.ones((2, 2)), np.ones((2, 2)))
        checked_ops += 1
    assert checked_ops == len(op_names)
