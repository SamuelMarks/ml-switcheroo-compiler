"""Exhaustive unit tests for eager execution primitives and dynamic tensor arrays."""

from __future__ import annotations

import os
import tempfile

import numpy as np
import pytest

from ml_switcheroo_compiler.backends.eager.core_math_ops.math_fft import _fftconvolve
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
    assert w1[1] == 99
    assert handle_list[1] == 20  # non-mutating copy

    # Expanding list write
    w_expand = _np_tensorarraywrite(np, handle_list, 5, 555)
    assert len(w_expand) == 6
    assert w_expand[5] == 555
    assert w_expand[3] is None

    # Write on None creates new list
    w_none = _np_tensorarraywrite(np, None, 2, "hello")
    assert len(w_none) == 3
    assert w_none[2] == "hello"

    # Negative index rejection
    with pytest.raises(IndexError, match="Negative TensorArray index"):
        _np_tensorarraywrite(np, handle_list, -1, 100)

    # 2. NumPy ndarray read / write
    arr_np = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    elem0 = _np_tensorarrayread(np, arr_np, 0)
    assert np.allclose(elem0, [1.0, 2.0])

    with pytest.raises(IndexError, match="out of bounds"):
        _np_tensorarrayread(np, arr_np, 4)

    # NumPy in-bounds write
    arr_w = _np_tensorarraywrite(np, arr_np, 1, np.array([9.0, 9.0], dtype=np.float32))
    assert np.allclose(arr_w[1], [9.0, 9.0])
    assert np.allclose(arr_np[1], [3.0, 4.0])  # non-mutating

    # NumPy expanding write
    arr_exp = _np_tensorarraywrite(np, arr_np, 3, np.array([8.0, 8.0], dtype=np.float32))
    assert arr_exp.shape[0] == 4
    assert np.allclose(arr_exp[3], [8.0, 8.0])

    # 3. Dict handle read / write
    d_handle: dict[int, str] = {0: "zero", 1: "one"}
    assert _np_tensorarrayread(np, d_handle, 1) == "one"

    with pytest.raises(IndexError, match="not found in sparse handle"):
        _np_tensorarrayread(np, d_handle, 99)

    d_w = _np_tensorarraywrite(np, d_handle, 2, "two")
    assert d_w[2] == "two"

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
    assert np.allclose(item1, [1, 2])
    item2 = _infeed(np, queue_name="train_queue")
    assert np.allclose(item2, [3, 4])

    # Exhausted queue returns 0
    assert _infeed(np, "train_queue") == 0

    # Infeed from generator / iterator
    def dummy_gen():
        yield 100
        yield 200

    register_infeed_queue("gen_queue", dummy_gen())
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
        with open(bin_path, "rb") as f:
            assert f.read() == payload

        # 3. Array tobytes write
        arr_path = os.path.join(tmpdir, "arr.bin")
        arr = np.array([1.5, 2.5, 3.5], dtype=np.float64)
        ret_arr = _np_writefile(np, arr_path, arr)
        assert ret_arr == 1
        with open(arr_path, "rb") as f:
            read_arr = np.frombuffer(f.read(), dtype=np.float64)
            assert np.allclose(read_arr, arr)

        # 4. Integer / generic content write
        int_path = os.path.join(tmpdir, "int.txt")
        assert _np_writefile(np, int_path, 98765) == 1
        with open(int_path, encoding="utf-8") as f:
            assert f.read() == "98765"

        # 5. Backend with custom writefile method
        class MockWritefileBackend:
            @staticmethod
            def writefile(*a, **k):
                return 42

        assert _np_writefile(MockWritefileBackend(), "dummy.txt", "abc") == 42

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
    assert out_full.shape == (5,)
    expected_full = np.convolve(x, h, mode="full")
    assert np.allclose(out_full, expected_full, atol=1e-5)

    # Same convolution: length 4
    out_same = _fftconvolve(np, x, h, mode="same")
    assert out_same.shape == (4,)
    expected_same = np.convolve(x, h, mode="same")
    assert np.allclose(out_same, expected_same, atol=1e-5)

    # Valid convolution: length 4 - 2 + 1 = 3
    out_valid = _fftconvolve(np, x, h, mode="valid")
    assert out_valid.shape == (3,)
    expected_valid = np.convolve(x, h, mode="valid")
    assert np.allclose(out_valid, expected_valid, atol=1e-5)

    # Valid mode with h larger than x returns empty
    out_empty = _fftconvolve(np, np.array([1.0]), np.array([1.0, 2.0, 3.0]), mode="valid")
    assert out_empty.size == 0

    # 2D real signals
    img2d = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    k2d = np.array([[1.0, 0.5], [0.5, 0.25]], dtype=np.float32)
    out_2d = _fftconvolve(np, img2d, k2d, mode="full")
    assert out_2d.shape == (3, 3)

    # Complex signal convolution
    c1 = np.array([1.0 + 2.0j, 2.0 - 1.0j], dtype=np.complex64)
    c2 = np.array([0.5 + 0.5j, 1.0 + 0.0j], dtype=np.complex64)
    out_complex = _fftconvolve(np, c1, c2, mode="full")
    assert out_complex.shape == (3,)

    # Backend with fft.fftconvolve
    class MockFFTModule:
        @staticmethod
        def fftconvolve(*a, **k):
            return np.array([77.0])

    class MockFFTBackend:
        fft = MockFFTModule()

    assert np.allclose(_fftconvolve(MockFFTBackend(), x, h), [77.0])

    # Backend with signal.fftconvolve
    class MockSignalModule:
        @staticmethod
        def fftconvolve(*a, **k):
            return np.array([88.0])

    class MockSignalBackend:
        signal = MockSignalModule()

    assert np.allclose(_fftconvolve(MockSignalBackend(), x, h), [88.0])
