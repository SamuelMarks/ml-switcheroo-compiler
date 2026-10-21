"""Exhaustive unit tests for eager execution primitives and dynamic tensor arrays."""

from __future__ import annotations

import os
import pathlib
import tempfile
import typing

import numpy as np
import pytest

from ml_switcheroo_compiler.backends.eager.core_math_ops.math_fft import (
    _fft,
    _fftconvolve,
    _fftnd,
    _np_hfft,
    _rfft,
)
from ml_switcheroo_compiler.backends.eager.core_math_ops.math_internal import (
    _np_tensorarrayread,
    _np_tensorarraywrite,
)
from ml_switcheroo_compiler.backends.eager.core_math_ops.math_logical import (
    _axis_index,
    _infeed,
    clear_axis_indices,
    clear_infeed_queues,
    register_infeed_queue,
    set_axis_index,
)
from ml_switcheroo_compiler.backends.eager.core_math_ops.math_string_io import _np_writefile
from ml_switcheroo_compiler.backends.eager.core_math_ops.math_testing import _assert
from ml_switcheroo_compiler.backends.eager_registry import EagerValue
from tests.testing_utils import as_callable, as_mod, as_tensor

if typing.TYPE_CHECKING:
    from collections.abc import Iterator

    from ml_switcheroo_compiler.backends.eager_registry import EagerValue


def test_tensorarray_read_and_write() -> None:
    """Test dynamic tensorarray read, write, bounds checking, and expansion."""
    # Empty / None inputs
    assert _np_tensorarrayread(np) == 0
    assert _np_tensorarraywrite(np) == 0

    # 1. List handle read / write
    handle_list = [10, 20, 30]
    assert _np_tensorarrayread(np, handle_list, 1) == 20

    # Out of bounds on list
    with pytest.raises(IndexError, match="out of bounds"):
        _np_tensorarrayread(np, handle_list, 5)
    with pytest.raises(IndexError, match="out of bounds"):
        _np_tensorarrayread(np, handle_list, -1)

    # In-bounds list write
    w1 = _np_tensorarraywrite(np, handle_list, 1, 99)
    assert as_tensor(getattr(w1, "__getitem__")(1)) == 99
    assert handle_list[1] == 20  # non-mutating copy

    # Expanding list write
    w_expand = _np_tensorarraywrite(np, handle_list, 5, 555)
    assert as_callable(len)(w_expand) == 6
    assert as_tensor(getattr(w_expand, "__getitem__")(5)) == 555
    assert as_tensor(getattr(w_expand, "__getitem__")(3)) is None

    # Write on None creates new list
    w_none = _np_tensorarraywrite(np, None, 2, "hello")
    assert as_callable(len)(w_none) == 3
    assert as_tensor(getattr(w_none, "__getitem__")(2)) == "hello"

    # Negative index rejection
    with pytest.raises(IndexError, match="Negative TensorArray index"):
        _np_tensorarraywrite(np, handle_list, -1, 100)

    # 2. NumPy ndarray read / write
    arr_np = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    elem0 = _np_tensorarrayread(np, arr_np, 0)
    assert as_callable(np.allclose)(elem0, [1.0, 2.0])

    with pytest.raises(IndexError, match="out of bounds"):
        _np_tensorarrayread(np, arr_np, 4)

    # NumPy in-bounds write
    arr_w = _np_tensorarraywrite(np, arr_np, 1, np.array([9.0, 9.0], dtype=np.float32))
    assert as_callable(np.allclose)(as_tensor(getattr(arr_w, "__getitem__")(1)), [9.0, 9.0])
    assert np.allclose(arr_np[1], [3.0, 4.0])  # non-mutating

    # NumPy expanding write
    arr_exp = _np_tensorarraywrite(np, arr_np, 3, np.array([8.0, 8.0], dtype=np.float32))
    assert as_tensor(getattr(as_tensor(getattr(arr_exp, "shape")), "__getitem__")(0)) == 4
    assert as_callable(np.allclose)(as_tensor(getattr(arr_exp, "__getitem__")(3)), [8.0, 8.0])

    # 3. Dict handle read / write
    d_handle: dict[int, str] = {0: "zero", 1: "one"}
    assert _np_tensorarrayread(np, as_tensor(d_handle), 1) == "one"

    with pytest.raises(IndexError, match="not found in sparse handle"):
        _np_tensorarrayread(np, as_tensor(d_handle), 99)

    d_w = _np_tensorarraywrite(np, as_tensor(d_handle), 2, "two")
    assert as_tensor(getattr(d_w, "__getitem__")(2)) == "two"

    # 4. Unknown type fallback
    assert _np_tensorarrayread(np, 12345) == 0
    assert _np_tensorarraywrite(np, 12345) == 0


def test_axis_index_and_infeed() -> None:
    """Test SPMD axis index context management and infeed host queue ingestion."""
    clear_axis_indices()
    # Default without registration
    assert _axis_index(np, "batch") == 0
    assert _axis_index(np, axis_name="model") == 0

    # Set and retrieve axis indices
    set_axis_index("batch", 3)
    set_axis_index("data_parallel", 7)
    assert _axis_index(np, "batch") == 3
    assert _axis_index(np, "data_parallel") == 7

    clear_axis_indices()
    assert _axis_index(np, "batch") == 0

    clear_infeed_queues()
    # Infeed on unconfigured queue
    assert _infeed(np, "unconfigured") == 0

    # Infeed from list queue
    batch_data = [np.array([1, 2]), np.array([3, 4])]
    register_infeed_queue("train_queue", list(batch_data))

    item1 = _infeed(np, "train_queue")
    assert as_callable(np.allclose)(item1, [1, 2])
    item2 = _infeed(np, queue_name="train_queue")
    assert as_callable(np.allclose)(item2, [3, 4])

    # Exhausted queue returns 0
    assert _infeed(np, "train_queue") == 0

    # Infeed from generator / iterator
    def dummy_gen() -> Iterator[EagerValue]:
        """Test function dummy_gen."""
        yield 100
        yield 200

    as_callable(register_infeed_queue)("gen_queue", as_callable(dummy_gen)())
    assert _infeed(np, "gen_queue") == 100
    assert _infeed(np, "gen_queue") == 200
    assert _infeed(np, "gen_queue") == 0

    clear_infeed_queues()


def test_assert_operation() -> None:
    """Test _assert condition verification, error formatting, and summaries."""
    # True condition passes silently
    assert _assert(np, True, data=None) is None
    assert _assert(np, np.array([True, True]), data=np.array([1, 2, 3])) is None

    # False condition raises AssertionError with node ID
    with pytest.raises(AssertionError, match="Assertion failed in node 'Node_42'"):
        _assert(np, False, data=np.array([10, 20, 30, 40, 50]), summarize=2, node_id="Node_42")

    # False array condition with data summary
    cond_arr = np.array([True, False, True])
    with pytest.raises(AssertionError, match=r"Data sample: \[10, 20\]"):
        _assert(np, cond_arr, data=np.array([10, 20, 30, 40]), summarize=2, node_id="CheckPos")

    # False condition with data=None
    with pytest.raises(AssertionError, match="Assertion failed in node 'AssertOp'"):
        _assert(np, False, data=None)


def test_writefile_operation() -> None:
    """Test atomic file writing with directory creation across text, bytes, and array data."""
    # Missing args
    assert _np_writefile(np) == 0

    with tempfile.TemporaryDirectory() as tmpdir:
        # 1. Text write to nested directory
        nested_txt = os.path.join(tmpdir, "subdir", "test.txt")
        ret_txt = _np_writefile(np, nested_txt, "Hello, world!")
        assert ret_txt == 1
        with open(nested_txt, encoding="utf-8") as f:
            assert f.read() == "Hello, world!"

        # 2. Bytes write
        bin_path = os.path.join(tmpdir, "test.bin")
        payload = b"\x00\x01\x02\x03\x04"
        ret_bin = _np_writefile(np, bin_path, payload)
        assert ret_bin == 1
        with open(bin_path, "rb") as fb:
            assert as_tensor(fb.read()) == as_tensor(payload)

        # 3. Array tobytes write
        arr_path = os.path.join(tmpdir, "arr.bin")
        arr = np.array([1.5, 2.5, 3.5], dtype=np.float64)
        ret_arr = _np_writefile(np, arr_path, arr)
        assert ret_arr == 1
        with open(arr_path, "rb") as fb:
            read_arr = as_callable(np.frombuffer)(as_callable(fb.read)(), dtype=np.float64)
            assert as_callable(np.allclose)(read_arr, arr)

        # 4. Integer / generic content write
        int_path = os.path.join(tmpdir, "int.txt")
        assert _np_writefile(np, int_path, 98765) == 1
        with open(int_path, encoding="utf-8") as f:
            assert f.read() == "98765"

        # 5. Backend with custom writefile method
        class MockWritefileBackend:
            """Test class MockWritefileBackend."""

            @staticmethod
            def writefile(*a: EagerValue, **k: EagerValue) -> EagerValue:
                """Test function writefile.

                Args:
                    a: Test parameter `a`.
                    k: Test parameter `k`.
                """
                return 42

        assert _np_writefile(as_mod(MockWritefileBackend()), "dummy.txt", "abc") == 42

        # 6. OSError trigger on write failure
        with pytest.raises(OSError):
            _np_writefile(np, "/non_existent_path_dir/forbidden/file.txt", "fail")


def test_fftconvolve_fallback() -> None:
    """Test frequency-domain convolution fallback across modes, dimensions, and dtypes."""
    # None inputs
    assert _fftconvolve(np, None, None) is None

    # 1D real signals
    x = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    h = np.array([0.5, 1.0], dtype=np.float32)

    # Full convolution: length 4 + 2 - 1 = 5
    out_full = _fftconvolve(np, x, h, mode="full")
    assert as_tensor(getattr(out_full, "shape")) == (5,)
    expected_full = np.convolve(x, h, mode="full")
    assert as_callable(np.allclose)(out_full, expected_full, atol=1e-5)

    # Same convolution: length 4
    out_same = _fftconvolve(np, x, h, mode="same")
    assert as_tensor(getattr(out_same, "shape")) == (4,)
    expected_same = np.convolve(x, h, mode="same")
    assert as_callable(np.allclose)(out_same, expected_same, atol=1e-5)

    # Valid convolution: length 4 - 2 + 1 = 3
    out_valid = _fftconvolve(np, x, h, mode="valid")
    assert as_tensor(getattr(out_valid, "shape")) == (3,)
    expected_valid = np.convolve(x, h, mode="valid")
    assert as_callable(np.allclose)(out_valid, expected_valid, atol=1e-5)

    # Valid mode with h larger than x returns empty
    out_empty = _fftconvolve(np, np.array([1.0]), np.array([1.0, 2.0, 3.0]), mode="valid")
    assert as_tensor(getattr(out_empty, "size")) == 0

    # 2D real signals
    img2d = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    k2d = np.array([[1.0, 0.5], [0.5, 0.25]], dtype=np.float32)
    out_2d = _fftconvolve(np, img2d, k2d, mode="full")
    assert as_tensor(getattr(out_2d, "shape")) == (3, 3)

    # Complex signal convolution
    c1 = np.array([1.0 + 2.0j, 2.0 - 1.0j], dtype=np.complex64)
    c2 = np.array([0.5 + 0.5j, 1.0 + 0.0j], dtype=np.complex64)
    out_complex = _fftconvolve(np, c1, c2, mode="full")
    assert as_tensor(getattr(out_complex, "shape")) == (3,)

    # Backend with fft.fftconvolve
    class MockFFTModule:
        """Test class MockFFTModule."""

        @staticmethod
        def fftconvolve(*a: EagerValue, **k: EagerValue) -> EagerValue:
            """Test function fftconvolve.

            Args:
                a: Test parameter `a`.
                k: Test parameter `k`.
            """
            return np.array([77.0])

    class MockFFTBackend:
        """Test class MockFFTBackend."""

        fft = MockFFTModule()

    assert as_callable(np.allclose)(as_callable(_fftconvolve)(as_callable(MockFFTBackend)(), x, h), [77.0])

    # Backend with signal.fftconvolve
    class MockSignalModule:
        """Test class MockSignalModule."""

        @staticmethod
        def fftconvolve(*a: EagerValue, **k: EagerValue) -> EagerValue:
            """Test function fftconvolve.

            Args:
                a: Test parameter `a`.
                k: Test parameter `k`.
            """
            return np.array([88.0])

    class MockSignalBackend:
        """Test class MockSignalBackend."""

        signal = MockSignalModule()

    assert as_callable(np.allclose)(as_callable(_fftconvolve)(as_callable(MockSignalBackend)(), x, h), [88.0])


def test_math_fft_edge_cases() -> None:
    """Verify edge cases and fallback branches for fft operations."""
    x = np.array([1.0, 2.0], dtype=np.float32)
    h = np.array([0.5], dtype=np.float32)

    # 1. Unknown convolution mode falls through to 'return out'
    out_custom = _fftconvolve(np, x, h, mode="custom_unhandled")
    assert as_tensor(getattr(out_custom, "shape")) == (2,)

    # 2. _fft and _rfft with backend missing fft
    class EmptyBackend:
        """Test class EmptyBackend."""

        pass

    assert _fft(as_mod(EmptyBackend()), x) is None
    assert _rfft(as_mod(EmptyBackend()), x) is None

    # 3. _fftnd with fftnd attribute
    class MockFFTNDModule:
        """Test class MockFFTNDModule."""

        @staticmethod
        def fftnd(*args: EagerValue, **kwargs: EagerValue) -> EagerValue:
            """Test function fftnd.

            Args:
                args: Test parameter `args`.
                kwargs: Test parameter `kwargs`.
            """
            return np.array([123.0])

    class MockFFTNDBackend:
        """Test class MockFFTNDBackend."""

        fft = MockFFTNDModule()

    assert as_callable(np.allclose)(as_callable(_fftnd)(as_callable(MockFFTNDBackend)(), x), [123.0])

    # 4. _np_hfft with hfft attribute on backend.fft
    class MockHFFTModule:
        """Test class MockHFFTModule."""

        @staticmethod
        def hfft(*args: EagerValue, **kwargs: EagerValue) -> EagerValue:
            """Test function hfft.

            Args:
                args: Test parameter `args`.
                kwargs: Test parameter `kwargs`.
            """
            return np.array([456.0])

    class MockHFFTBackend:
        """Test class MockHFFTBackend."""

        fft = MockHFFTModule()

    assert as_callable(np.allclose)(as_callable(_np_hfft)(as_callable(MockHFFTBackend)(), x), [456.0])

    # 5. _np_hfft fallback to np.fft.hfft
    res_hfft_fallback = _np_hfft(as_mod(EmptyBackend()), x)
    assert res_hfft_fallback is not None


def test_math_string_io_write_file_error(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify _write_file error handling and cleanup on OSError.

    Args:
        tmp_path: Test parameter `tmp_path`.
        monkeypatch: Test parameter `monkeypatch`.
    """
    from ml_switcheroo_compiler.backends.eager.core_math_ops.math_string_io import _np_writefile

    target_file = tmp_path / "test_out.txt"

    # 1. Normal write success with str, bytes, and tobytes
    assert _np_writefile(as_mod(object()), str(target_file), "hello world") == 1
    assert _np_writefile(as_mod(object()), str(target_file), b"hello bytes") == 1

    class DummyWithToBytes:
        """Test class DummyWithToBytes."""

        def tobytes(self) -> bytes:
            """Test function tobytes."""
            return b"dummy_bytes"

    assert _np_writefile(as_mod(object()), str(target_file), as_tensor(DummyWithToBytes())) == 1

    # 2. OSError where tmp file exists and gets cleaned up
    def mock_replace_error(src: str, dst: str) -> None:
        """Test function mock_replace_error.

        Args:
            src: Test parameter `src`.
            dst: Test parameter `dst`.
        """
        raise OSError("Disk full error")

    monkeypatch.setattr("os.replace", mock_replace_error)
    with pytest.raises(OSError, match="Disk full error"):
        _np_writefile(as_mod(object()), str(target_file), "content")

    # 3. OSError where tmp file does not exist
    def mock_open_error(*args: EagerValue, **kwargs: EagerValue) -> EagerValue:
        """Test function mock_open_error.

        Args:
            args: Test parameter `args`.
            kwargs: Test parameter `kwargs`.
        """
        raise OSError("Cannot open error")

    monkeypatch.setattr("builtins.open", mock_open_error)
    with pytest.raises(OSError, match="Cannot open error"):
        _np_writefile(as_mod(object()), str(target_file), "content")

    # 4. Bare filename (no directory) covering dir_name == "" branch and other object types
    monkeypatch.undo()
    monkeypatch.chdir(tmp_path)
    assert _np_writefile(as_mod(object()), "bare_filename.txt", 12345) == 1
    assert (tmp_path / "bare_filename.txt").read_text(encoding="utf-8") == "12345"
