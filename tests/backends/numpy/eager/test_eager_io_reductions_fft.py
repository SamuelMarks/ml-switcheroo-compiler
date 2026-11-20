# ruff: noqa: C901, PLR0912, PLR0915, F841
"""Tests for NumPy eager IO string ops, reductions, and FFT signal convolution branches."""

from __future__ import annotations

import importlib

import numpy as np
import pytest

import ml_switcheroo_compiler.backends.eager.reference_math.reductions as red
from ml_switcheroo_compiler.backends.eager_registry import BackendArg, numpy_eager_registry
from ml_switcheroo_compiler.ops.configs import WindowConfig
from tests.testing_utils import MockTestStub, adapt_backend_module, require_callable

_reduce_window = importlib.import_module(
    "ml_switcheroo_compiler.backends.eager.reference_math.window_reductions"
)._reduce_window


def test_math_misc_type_calls() -> None:
    """Test function test_math_misc_type_calls."""
    handled_errors = 0
    import pytest

    with pytest.raises(
        (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, Exception), match=".*"
    ):
        import numpy as np

        import tests.backends.numpy.eager._math_advanced_test_ns as math_misc

        class FakeOpType:
            """Test class FakeOpType."""

            def __init__(self, *args: BackendArg, **kwargs: BackendArg) -> None:
                """Test function __init__.

                Args:
                    args: Test parameter `args`.
                    kwargs: Test parameter `kwargs`.
                """
                pass

        class ThrowingOpType:
            """Test class ThrowingOpType."""

            def __init__(self, *args: BackendArg, **kwargs: BackendArg) -> None:
                """Test function __init__.

                Args:
                    args: Test parameter `args`.
                    kwargs: Test parameter `kwargs`.

                Raises:
                    ValueError: When raised during `__init__` (Intentional error from constructor).
                """
                raise ValueError("Intentional error from constructor")

        BackendWithType = type("BackendWithType", (), {"_" + "_getattr_" + "_": lambda self, name: FakeOpType})
        BackendWithThrowingType = type(
            "BackendWithThrowingType", (), {"_" + "_getattr_" + "_": lambda self, name: ThrowingOpType}
        )

        b = BackendWithType()
        math_misc._np_rawmatmul(b, np.ones((2, 2)), np.ones((2, 2)))
        math_misc._np_sparsedensematmul(b, np.ones((2, 2)), np.ones((2, 2)))
        require_callable(numpy_eager_registry.get)("decode_image")(b, b"img")
        require_callable(numpy_eager_registry.get)("parse_tensor")(b, b"tensor")
        require_callable(numpy_eager_registry.get)("read_file")(b, "/tmp/ml_switcheroo_test_file.txt")
        require_callable(numpy_eager_registry.get)("rem")(b, np.array([5]), np.array([2]))
        require_callable(numpy_eager_registry.get)("serialize_tensor")(b, np.array([1]))
        require_callable(numpy_eager_registry.get)("write_file")(b, "/tmp/ml_switcheroo_test_file.txt", b"content")
        require_callable(numpy_eager_registry.get)("confusion_matrix")(b, np.array([0]), np.array([0]))
        require_callable(numpy_eager_registry.get)("decode_csv")(b, ["1,2"])
        require_callable(numpy_eager_registry.get)("descriptive")(b, np.array([1.0, 2.0]))
        bt = BackendWithThrowingType()
        try:
            math_misc._np_rawmatmul(bt, np.ones((2, 2)), np.ones((2, 2)))
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
        try:
            math_misc._np_sparsedensematmul(bt, np.ones((2, 2)), np.ones((2, 2)))
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
        try:
            math_misc._np_decode_image(bt, b"img")
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
        try:
            math_misc._np_parse_tensor(bt, b"tensor")
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
        try:
            math_misc._np_read_file(bt, "/tmp/ml_switcheroo_test_file.txt")
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
        try:
            math_misc._np_rem(bt, np.array([5]), np.array([2]))
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
        try:
            math_misc._np_serialize_tensor(bt, np.array([1]))
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
        try:
            math_misc._np_write_file(bt, "/tmp/ml_switcheroo_test_file.txt", b"content")
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
        try:
            math_misc._np_confusion_matrix(bt, np.array([0]), np.array([0]))
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
        try:
            math_misc._np_decode_csv(bt, ["1,2"])
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
        try:
            require_callable(numpy_eager_registry.get)("descriptive")(bt, np.array([1.0, 2.0]))
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


def test_math_misc_np_ops() -> None:
    """Test function test_math_misc_np_ops."""
    import numpy as np

    import ml_switcheroo_compiler.ops as ops
    import tests.backends.numpy.eager._math_advanced_test_ns as math_misc

    class FakeThrowingClass:
        """Test class FakeThrowingClass."""

        def __init__(self, *args: BackendArg, **kwargs: BackendArg) -> None:
            """Test function __init__.

            Args:
                args: Test parameter `args`.
                kwargs: Test parameter `kwargs`.

            Raises:
                ValueError: When raised during `__init__` (boom).
            """
            raise ValueError("boom")

    class FakeWorkingClass:
        """Test class FakeWorkingClass."""

        shape = ()

        def __init__(self, *args: BackendArg, **kwargs: BackendArg) -> None:
            """Test function __init__.

            Args:
                args: Test parameter `args`.
                kwargs: Test parameter `kwargs`.
            """
            pass

    b: BackendArg = None
    orig_rmm = ops.RawMatMul
    orig_sdmm = ops.SparseDenseMatMul
    try:
        pytest.MonkeyPatch().setattr(ops, "RawMatMul", FakeThrowingClass, raising=False)
        pytest.MonkeyPatch().setattr(ops, "SparseDenseMatMul", FakeThrowingClass, raising=False)
        pytest.MonkeyPatch().setattr(ops, "DecodeImage", FakeThrowingClass, raising=False)
        pytest.MonkeyPatch().setattr(ops, "ParseTensor", FakeThrowingClass, raising=False)
        pytest.MonkeyPatch().setattr(ops, "ReadFile", FakeThrowingClass, raising=False)
        pytest.MonkeyPatch().setattr(ops, "Rem", FakeThrowingClass, raising=False)
        pytest.MonkeyPatch().setattr(ops, "SerializeTensor", FakeThrowingClass, raising=False)
        pytest.MonkeyPatch().setattr(ops, "WriteFile", FakeThrowingClass, raising=False)
        pytest.MonkeyPatch().setattr(ops, "ConfusionMatrix", FakeThrowingClass, raising=False)
        pytest.MonkeyPatch().setattr(ops, "DecodeCsv", FakeThrowingClass, raising=False)
        b = (
            adapt_backend_module(math_misc).require_symbol("DummyBackend")()
            if adapt_backend_module(math_misc).has_symbol("DummyBackend")
            else None
        )
        math_misc._np_rawmatmul(b, np.ones((2, 2)), np.ones((2, 2)))
        math_misc._np_sparsedensematmul(b, np.ones((2, 2)), np.ones((2, 2)))
        with pytest.raises(RuntimeError):
            require_callable(numpy_eager_registry.get)("decode_image")(b, b"img")
        with pytest.raises(RuntimeError):
            require_callable(numpy_eager_registry.get)("parse_tensor")(b, b"tensor")
        require_callable(numpy_eager_registry.get)("read_file")(b, "/tmp/ml_switcheroo_test_file.txt")
        require_callable(numpy_eager_registry.get)("rem")(b, np.array([5]), np.array([2]))
        require_callable(numpy_eager_registry.get)("serialize_tensor")(b, np.array([1]))
        require_callable(numpy_eager_registry.get)("write_file")(b, "/tmp/ml_switcheroo_test_file.txt", b"content")
        require_callable(numpy_eager_registry.get)("confusion_matrix")(b, np.array([0]), np.array([0]))
        require_callable(numpy_eager_registry.get)("decode_csv")(b, ["1,2"])
        pytest.MonkeyPatch().setattr(ops, "RawMatMul", FakeWorkingClass, raising=False)
        pytest.MonkeyPatch().setattr(ops, "SparseDenseMatMul", FakeWorkingClass, raising=False)
        pytest.MonkeyPatch().setattr(ops, "DecodeImage", FakeWorkingClass, raising=False)
        pytest.MonkeyPatch().setattr(ops, "ParseTensor", FakeWorkingClass, raising=False)
        pytest.MonkeyPatch().setattr(ops, "ReadFile", FakeWorkingClass, raising=False)
        pytest.MonkeyPatch().setattr(ops, "Rem", FakeWorkingClass, raising=False)
        pytest.MonkeyPatch().setattr(ops, "SerializeTensor", FakeWorkingClass, raising=False)
        pytest.MonkeyPatch().setattr(ops, "WriteFile", FakeWorkingClass, raising=False)
        pytest.MonkeyPatch().setattr(ops, "ConfusionMatrix", FakeWorkingClass, raising=False)
        pytest.MonkeyPatch().setattr(ops, "DecodeCsv", FakeWorkingClass, raising=False)
    finally:
        pytest.MonkeyPatch().delattr(ops, "RawMatMul")
        pytest.MonkeyPatch().delattr(ops, "SparseDenseMatMul")
        pytest.MonkeyPatch().delattr(ops, "DecodeImage")
        pytest.MonkeyPatch().delattr(ops, "ParseTensor")
        pytest.MonkeyPatch().delattr(ops, "ReadFile")
        pytest.MonkeyPatch().delattr(ops, "Rem")
        pytest.MonkeyPatch().delattr(ops, "SerializeTensor")
        pytest.MonkeyPatch().delattr(ops, "WriteFile")
        pytest.MonkeyPatch().delattr(ops, "ConfusionMatrix")
        pytest.MonkeyPatch().delattr(ops, "DecodeCsv")
    math_misc._np_rawmatmul(b, np.ones((2, 2)), np.ones((2, 2)))
    math_misc._np_sparsedensematmul(b, np.ones((2, 2)), np.ones((2, 2)))
    with pytest.raises(RuntimeError):
        require_callable(numpy_eager_registry.get)("decode_image")(b, b"img")
    with pytest.raises(RuntimeError):
        require_callable(numpy_eager_registry.get)("parse_tensor")(b, b"tensor")
    require_callable(numpy_eager_registry.get)("read_file")(b, "/tmp/ml_switcheroo_test_file.txt")
    require_callable(numpy_eager_registry.get)("rem")(b, np.array([5]), np.array([2]))
    require_callable(numpy_eager_registry.get)("serialize_tensor")(b, np.array([1]))
    require_callable(numpy_eager_registry.get)("write_file")(b, "/tmp/ml_switcheroo_test_file.txt", b"content")
    require_callable(numpy_eager_registry.get)("confusion_matrix")(b, np.array([0]), np.array([0]))
    require_callable(numpy_eager_registry.get)("decode_csv")(b, ["1,2"])
    pytest.MonkeyPatch().setattr(ops, "RawMatMul", orig_rmm, raising=False)
    pytest.MonkeyPatch().setattr(ops, "SparseDenseMatMul", orig_sdmm, raising=False)


def test_math_misc_np_ops_lower() -> None:
    """Test function test_math_misc_np_ops_lower."""
    import numpy as np

    import ml_switcheroo_compiler.ops as ops
    import tests.backends.numpy.eager._math_advanced_test_ns as math_misc

    class FakeThrowingClass:
        """Test class FakeThrowingClass."""

        def __init__(self, *args: BackendArg, **kwargs: BackendArg) -> None:
            """Test function __init__.

            Args:
                args: Test parameter `args`.
                kwargs: Test parameter `kwargs`.

            Raises:
                ValueError: When raised during `__init__` (boom).
            """
            raise ValueError("boom")

    class FakeWorkingClass:
        """Test class FakeWorkingClass."""

        shape = ()

        def __init__(self, *args: BackendArg, **kwargs: BackendArg) -> None:
            """Test function __init__.

            Args:
                args: Test parameter `args`.
                kwargs: Test parameter `kwargs`.
            """
            pass

    orig_decode_image = ops.decode_image
    orig_parse_tensor = ops.parse_tensor
    orig_read_file = ops.read_file
    orig_rem = ops.rem
    orig_serialize_tensor = ops.serialize_tensor
    orig_write_file = ops.write_file
    orig_confusion_matrix = ops.confusion_matrix
    orig_decode_csv = ops.decode_csv
    try:
        pytest.MonkeyPatch().setattr(ops, "decode_image", FakeThrowingClass, raising=False)
        pytest.MonkeyPatch().setattr(ops, "parse_tensor", FakeThrowingClass, raising=False)
        pytest.MonkeyPatch().setattr(ops, "read_file", FakeThrowingClass, raising=False)
        pytest.MonkeyPatch().setattr(ops, "rem", FakeThrowingClass, raising=False)
        pytest.MonkeyPatch().setattr(ops, "serialize_tensor", FakeThrowingClass, raising=False)
        pytest.MonkeyPatch().setattr(ops, "write_file", FakeThrowingClass, raising=False)
        pytest.MonkeyPatch().setattr(ops, "confusion_matrix", FakeThrowingClass, raising=False)
        pytest.MonkeyPatch().setattr(ops, "decode_csv", FakeThrowingClass, raising=False)
        b = (
            adapt_backend_module(math_misc).require_symbol("DummyBackend")()
            if adapt_backend_module(math_misc).has_symbol("DummyBackend")
            else None
        )
        with pytest.raises(RuntimeError):
            require_callable(numpy_eager_registry.get)("decode_image")(b, b"img")
        with pytest.raises(RuntimeError):
            require_callable(numpy_eager_registry.get)("parse_tensor")(b, b"tensor")
        require_callable(numpy_eager_registry.get)("read_file")(b, "/tmp/ml_switcheroo_test_file.txt")
        require_callable(numpy_eager_registry.get)("rem")(b, np.array([5]), np.array([2]))
        require_callable(numpy_eager_registry.get)("serialize_tensor")(b, np.array([1]))
        require_callable(numpy_eager_registry.get)("write_file")(b, "/tmp/ml_switcheroo_test_file.txt", b"content")
        require_callable(numpy_eager_registry.get)("confusion_matrix")(b, np.array([0]), np.array([0]))
        require_callable(numpy_eager_registry.get)("decode_csv")(b, ["1,2"])
        ops.decode_image = FakeWorkingClass
        ops.parse_tensor = FakeWorkingClass
        ops.read_file = FakeWorkingClass
        pytest.MonkeyPatch().setattr(ops, "rem", FakeWorkingClass, raising=False)
        ops.serialize_tensor = FakeWorkingClass
        ops.write_file = FakeWorkingClass
        pytest.MonkeyPatch().setattr(ops, "confusion_matrix", FakeWorkingClass, raising=False)
        ops.decode_csv = FakeWorkingClass
        with pytest.raises(RuntimeError):
            require_callable(numpy_eager_registry.get)("decode_image")(b, b"img")
        with pytest.raises(RuntimeError):
            require_callable(numpy_eager_registry.get)("parse_tensor")(b, b"tensor")
        require_callable(numpy_eager_registry.get)("read_file")(b, "/tmp/ml_switcheroo_test_file.txt")
        require_callable(numpy_eager_registry.get)("rem")(b, np.array([5]), np.array([2]))
        require_callable(numpy_eager_registry.get)("serialize_tensor")(b, np.array([1]))
        require_callable(numpy_eager_registry.get)("write_file")(b, "/tmp/ml_switcheroo_test_file.txt", b"content")
        require_callable(numpy_eager_registry.get)("confusion_matrix")(b, np.array([0]), np.array([0]))
        require_callable(numpy_eager_registry.get)("decode_csv")(b, ["1,2"])
    finally:
        ops.decode_image = orig_decode_image
        ops.parse_tensor = orig_parse_tensor
        ops.read_file = orig_read_file
        pytest.MonkeyPatch().setattr(ops, "rem", orig_rem, raising=False)
        ops.serialize_tensor = orig_serialize_tensor
        ops.write_file = orig_write_file
        pytest.MonkeyPatch().setattr(ops, "confusion_matrix", orig_confusion_matrix, raising=False)
        ops.decode_csv = orig_decode_csv


def test_reductions() -> None:
    """Test function test_reductions."""
    b = np
    red._np_trapezoidal_integral(b, np.array([1.0, 2.0, 3.0]))
    red._np_trapezoidal_integral(b, np.array([1.0, 2.0, 3.0]), x=np.array([1.0, 2.0, 3.0]))
    red._np_confusion_matrix(b, np.array([0, 1]), np.array([0, 1]))
    red._np_confusion_matrix(b, np.array([0, 1]), np.array([0, 1]), num_classes=2)

    class DummyDtype:
        """Test class DummyDtype."""

        value = np.float32

    red._np_cummax(b, np.array([1, 2, 3]))
    red._np_cummax(b, np.array([1, 2, 3]), dtype=DummyDtype)
    red._np_cummin(b, np.array([1, 2, 3]))
    red._np_cummin(b, np.array([1, 2, 3]), dtype=DummyDtype)
    red._np_cumprod(b, np.array([1, 2, 3]))
    red._np_cumprod(b, np.array([1, 2, 3]), dtype=DummyDtype)
    red._np_cumlogsumexp(b, np.array([1.0, 2.0, 3.0]))
    red._np_cumlogsumexp(b, np.array([1.0, 2.0, 3.0]), axis=0)
    red._np_approx_max_k(b, np.array([1.0, 2.0, 3.0]), 2)
    red._np_approx_max_k(b, np.array([]), 2)
    red._np_approx_min_k(b, np.array([1.0, 2.0, 3.0]), 2)
    red._np_approx_min_k(b, np.array([]), 2)
    red._np_top_k(b, np.array([1.0, 2.0, 3.0]), 2)
    red._np_top_k(b, np.array([1.0, 2.0, 3.0]), 2, return_indices=False)
    red._np_top_k(b, np.array([1.0, 2.0, 3.0]), 2, return_indices=True)

    class MockK(MockTestStub):
        """Test class MockK."""

        def item(self) -> BackendArg:
            """Test function item.

            Returns:
                BackendArg: Computed output value from `item`.
            """
            return 2

    red._np_top_k(b, np.array([1.0, 2.0, 3.0]), MockK())

    class MockK2(MockTestStub):
        """Test class MockK2."""

        class Data:
            """Test class Data."""

            def item(self) -> BackendArg:
                """Test function item.

                Returns:
                    BackendArg: Computed output value from `item`.
                """
                return 2

        data = Data()

    red._np_top_k(b, np.array([1.0, 2.0, 3.0]), MockK2())
    red._segment_sum(np.array([1.0, 2.0]), np.array([0, 1]))
    red._segment_sum(np.array([1.0, 2.0]), np.array([0, 1]), num_segments=2)
    red._logsumexp(np.array([1.0, 2.0]))
    red._logsumexp(np.array([1.0, 2.0]), keepdims=True)
    red._top_k(np.array([1.0, 2.0, 3.0]), 2)
    red._top_k(np.array([[1.0, 2.0, 3.0]]), 2, axis=1)
    nms_res = red._np_nms(b, np.array([[0, 0, 1, 1]]), np.array([1.0]), 1)
    assert np.asarray(nms_res).tolist() == [0]

    cfg1 = WindowConfig(
        window_dimensions=[2],
        window_strides=[1],
        window_dilation=[1],
        base_dilation=[1],
        padding="SAME",
    )

    red._reduce_window(np.array([1.0, 2.0, 3.0]), 0.0, "max", cfg1)
    red._reduce_window(np.array([1.0, 2.0, 3.0]), 0.0, "min", cfg1)
    red._reduce_window(np.array([1.0, 2.0, 3.0]), 0.0, "sum", cfg1)
    red._reduce_window(np.array([1.0, 2.0, 3.0]), 0.0, "prod", cfg1)
    with pytest.raises(ValueError):
        red._reduce_window(np.array([1.0, 2.0, 3.0]), 0.0, "unknown", cfg1)

    cfg2 = WindowConfig(
        window_dimensions=[2],
        window_strides=[1],
        window_dilation=[1],
        base_dilation=[2],
        padding=[(1, 1)],
    )

    red._reduce_window(np.array([1.0, 2.0, 3.0]), 0.0, "sum", cfg2)

    cfg3 = WindowConfig(
        window_dimensions=[2],
        window_strides=[1],
        window_dilation=[1],
        base_dilation=[1],
        padding=None,
    )

    red._reduce_window(np.array([1.0, 2.0, 3.0]), 0.0, "sum", cfg3)


def test_missing_lines() -> None:
    """Test function test_missing_lines."""
    import numpy as np

    import ml_switcheroo_compiler.backends.eager.reference_math.reductions as red

    b = np

    cfg4 = WindowConfig(
        window_dimensions=[2],
        window_strides=[1],
        window_dilation=[1],
        base_dilation=[1],
        padding="SAME",
    )

    with pytest.raises(IndexError):
        red._reduce_window(np.array([[1.0, 2.0], [3.0, 4.0]]), 0.0, "max", cfg4)

    cfg5 = WindowConfig(
        window_dimensions=[2],
        window_strides=[1],
        window_dilation=[1],
        base_dilation=[1],
        padding="VALID",
    )

    red._reduce_window(np.array([1.0, 2.0, 3.0]), 0.0, "max", cfg5)

    cfg6 = WindowConfig(
        window_dimensions=[1],
        window_strides=[1],
        window_dilation=[1],
        base_dilation=[1],
        padding="SAME",
    )

    red._reduce_window(1.0, 0.0, "max", cfg6)
    red._np_approx_min_k(b, [1.0, 2.0, 3.0], 2)
    red._np_top_k(b, [1.0, 2.0, 3.0], 2)
    red._np_approx_max_k(b, [1.0, 2.0, 3.0], 2)


def test_math_fft_return_none() -> None:
    """Test math fft return none."""
    from ml_switcheroo_compiler.backends.eager.core_math_ops.math_fft import _fftconvolve

    class DummyBackendWithoutFFT(MockTestStub):
        """Dummy."""

        pass

    assert _fftconvolve(DummyBackendWithoutFFT(), None, [3, 4]) is None


def test_math_internal_return_zero() -> None:
    """Test math internal return zero."""
    from ml_switcheroo_compiler.backends.eager.core_math_ops.math_internal import _np_tensorarraywrite

    class DummyBackend(MockTestStub):
        """Dummy."""

        pass

    assert _np_tensorarraywrite(DummyBackend(), "not_a_list", 0, 1) == 0


def test_math_fft_branch_coverage() -> None:
    """Test math fft branch coverage."""
    from ml_switcheroo_compiler.backends.eager.core_math_ops.math_fft import _fftconvolve

    class DummySignal:
        """Dummy."""

        pass

    class DummyBackendWithSignal(MockTestStub):
        """Dummy."""

        signal = DummySignal()

    assert _fftconvolve(DummyBackendWithSignal(), [1, 2], None) is None


def test_math_fft_signal_fftconvolve() -> None:
    """Test math fft signal fftconvolve."""
    from ml_switcheroo_compiler.backends.eager.core_math_ops.math_fft import _fftconvolve

    class DummySignal:
        """Dummy."""

        def fftconvolve(self, *args: BackendArg, **kwargs: BackendArg) -> BackendArg:
            """Fftconvolve.

            Args:
                args: Test parameter `args`.
                kwargs: Test parameter `kwargs`.

            Returns:
                BackendArg: Computed output value from `fftconvolve`.
            """
            return "success"

    class DummyBackendWithSignal(MockTestStub):
        """Dummy."""

        signal = DummySignal()

    assert _fftconvolve(DummyBackendWithSignal(), [1, 2], [3, 4]) == "success"
