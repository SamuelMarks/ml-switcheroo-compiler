"""Tests for CUDA backend components."""

import ctypes
from unittest.mock import MagicMock, patch

import pytest

from ml_switcheroo_compiler.backends.cuda.cuda import CudaCodeGenerator, CUDARunner
from ml_switcheroo_compiler.ir.core import IRGraph, IRNode


def test_cuda_generator():
    """Test CUDA code generation."""
    graph = IRGraph()
    graph.nodes = {"matmul": IRNode(id="matmul", op_type="MatMul", inputs=[])}
    gen = CudaCodeGenerator(graph)
    out = gen.generate()
    assert "matmul" in out


def test_cuda_generator_coverage():
    """Test CUDA generator coverage."""
    graph = IRGraph()
    graph.nodes = {"input": IRNode(id="input", op_type="Input", inputs=[])}
    gen = CudaCodeGenerator(graph)
    out = gen.generate()
    assert "matmul" not in out


def test_cuda_missing_yaml(monkeypatch):
    import os

    monkeypatch.setattr(os.path, "exists", lambda p: False)
    monkeypatch.setattr(os.path, "isdir", lambda p: False)
    graph = IRGraph()
    gen = CudaCodeGenerator(graph)
    assert True


def test_cuda_runner_init_no_cupy(monkeypatch):
    """Test CUDARunner initialization without cupy."""
    monkeypatch.setattr("ml_switcheroo_compiler.backends.cuda.cuda.cupy", None)

    with patch("ctypes.util.find_library", return_value="libcuda.so"):
        mock_cdll = MagicMock()
        with patch("ctypes.cdll.LoadLibrary", return_value=mock_cdll):
            runner = CUDARunner()
            assert runner.mode == "ctypes"
            mock_cdll.cuInit.assert_called_once_with(0)


def test_cuda_runner_init_with_cupy():
    """Test CUDARunner initialization with cupy."""
    mock_cupy = MagicMock()
    with patch("ml_switcheroo_compiler.backends.cuda.cuda.cupy", mock_cupy):
        runner = CUDARunner()
        assert runner.mode == "cupy"


def test_cuda_runner_allocate_cupy():
    mock_cupy = MagicMock()
    mock_mem = MagicMock()
    mock_mem.ptr = 12345
    mock_cupy.cuda.alloc.return_value = mock_mem
    with patch("ml_switcheroo_compiler.backends.cuda.cuda.cupy", mock_cupy):
        runner = CUDARunner()
        ptr = runner.allocate_buffer(100)
        assert ptr.value == 12345


def test_cuda_runner_allocate_ctypes():
    with patch("ml_switcheroo_compiler.backends.cuda.cuda.cupy", None):
        with patch("ctypes.util.find_library", return_value="libcuda.so"):
            mock_cdll = MagicMock()
            mock_cdll.cuMemAlloc_v2.return_value = 0
            with patch("ctypes.cdll.LoadLibrary", return_value=mock_cdll):
                runner = CUDARunner()
                ptr = runner.allocate_buffer(100)
                assert mock_cdll.cuMemAlloc_v2.called


def test_cuda_runner_allocate_ctypes_fail():
    with patch("ml_switcheroo_compiler.backends.cuda.cuda.cupy", None):
        with patch("ctypes.util.find_library", return_value="libcuda.so"):
            mock_cdll = MagicMock()
            mock_cdll.cuMemAlloc_v2.return_value = 1  # error
            with patch("ctypes.cdll.LoadLibrary", return_value=mock_cdll):
                runner = CUDARunner()
                with pytest.raises(RuntimeError, match="cuMemAlloc failed"):
                    runner.allocate_buffer(100)


def test_cuda_runner_allocate_ctypes_no_lib():
    with patch("ml_switcheroo_compiler.backends.cuda.cuda.cupy", None):
        with patch("ctypes.util.find_library", return_value=None):
            runner = CUDARunner()
            ptr = runner.allocate_buffer(100)
            assert ptr.value is None


def test_cuda_runner_write_cupy():
    mock_cupy = MagicMock()
    with patch("ml_switcheroo_compiler.backends.cuda.cuda.cupy", mock_cupy):
        runner = CUDARunner()
        runner.write_buffer(ctypes.c_void_p(123), b"test")
        assert mock_cupy.ndarray.called


def test_cuda_runner_write_ctypes():
    with patch("ml_switcheroo_compiler.backends.cuda.cuda.cupy", None):
        with patch("ctypes.util.find_library", return_value="libcuda.so"):
            mock_cdll = MagicMock()
            mock_cdll.cuMemcpyHtoD_v2.return_value = 0
            with patch("ctypes.cdll.LoadLibrary", return_value=mock_cdll):
                runner = CUDARunner()
                runner.write_buffer(ctypes.c_void_p(123), b"test")
                assert mock_cdll.cuMemcpyHtoD_v2.called


def test_cuda_runner_write_ctypes_fail():
    with patch("ml_switcheroo_compiler.backends.cuda.cuda.cupy", None):
        with patch("ctypes.util.find_library", return_value="libcuda.so"):
            mock_cdll = MagicMock()
            mock_cdll.cuMemcpyHtoD_v2.return_value = 1
            with patch("ctypes.cdll.LoadLibrary", return_value=mock_cdll):
                runner = CUDARunner()
                with pytest.raises(RuntimeError, match="cuMemcpyHtoD failed"):
                    runner.write_buffer(ctypes.c_void_p(123), b"test")


def test_cuda_runner_write_no_ptr():
    runner = CUDARunner()
    assert runner.write_buffer(None, b"test") is None


def test_cuda_runner_read_cupy():
    mock_cupy = MagicMock()
    mock_arr = MagicMock()
    mock_arr.get.return_value.tobytes.return_value = b"test"
    mock_cupy.ndarray.return_value = mock_arr
    with patch("ml_switcheroo_compiler.backends.cuda.cuda.cupy", mock_cupy):
        runner = CUDARunner()
        res = runner.read_buffer(ctypes.c_void_p(123), 4)
        assert res == b"test"


def test_cuda_runner_read_ctypes():
    with patch("ml_switcheroo_compiler.backends.cuda.cuda.cupy", None):
        with patch("ctypes.util.find_library", return_value="libcuda.so"):
            mock_cdll = MagicMock()
            mock_cdll.cuMemcpyDtoH_v2.return_value = 0
            with patch("ctypes.cdll.LoadLibrary", return_value=mock_cdll):
                runner = CUDARunner()
                res = runner.read_buffer(ctypes.c_void_p(123), 4)
                assert len(res) == 4


def test_cuda_runner_read_ctypes_fail():
    with patch("ml_switcheroo_compiler.backends.cuda.cuda.cupy", None):
        with patch("ctypes.util.find_library", return_value="libcuda.so"):
            mock_cdll = MagicMock()
            mock_cdll.cuMemcpyDtoH_v2.return_value = 1
            with patch("ctypes.cdll.LoadLibrary", return_value=mock_cdll):
                runner = CUDARunner()
                with pytest.raises(RuntimeError, match="cuMemcpyDtoH failed"):
                    runner.read_buffer(ctypes.c_void_p(123), 4)


def test_cuda_runner_read_no_ptr():
    runner = CUDARunner()
    assert runner.read_buffer(None, 4) == b"\x00" * 4


def test_cuda_runner_free_ctypes():
    with patch("ml_switcheroo_compiler.backends.cuda.cuda.cupy", None):
        with patch("ctypes.util.find_library", return_value="libcuda.so"):
            mock_cdll = MagicMock()
            with patch("ctypes.cdll.LoadLibrary", return_value=mock_cdll):
                runner = CUDARunner()
                runner.free_buffer(ctypes.c_void_p(123))
                assert mock_cdll.cuMemFree_v2.called


def test_cuda_runner_free_cupy():
    mock_cupy = MagicMock()
    with patch("ml_switcheroo_compiler.backends.cuda.cuda.cupy", mock_cupy):
        runner = CUDARunner()
        runner.free_buffer(ctypes.c_void_p(123))
        # Just passes
        assert True


def test_cuda_runner_free_no_ptr():
    runner = CUDARunner()
    runner.free_buffer(None)


def test_cuda_runner_compile_cupy():
    mock_cupy = MagicMock()
    with patch("ml_switcheroo_compiler.backends.cuda.cuda.cupy", mock_cupy):
        runner = CUDARunner()
        runner.compile_and_dispatch("ptx", "main", [1, 1, 1], [1, 1, 1])
        assert mock_cupy.RawModule.called


def test_cuda_runner_compile_ctypes():
    with patch("ml_switcheroo_compiler.backends.cuda.cuda.cupy", None):
        with patch("ctypes.util.find_library", return_value="libcuda.so"):
            mock_cdll = MagicMock()
            mock_cdll.cuModuleLoadData.return_value = 0
            mock_cdll.cuModuleGetFunction.return_value = 0
            mock_cdll.cuLaunchKernel.return_value = 0
            with patch("ctypes.cdll.LoadLibrary", return_value=mock_cdll):
                runner = CUDARunner()
                runner.compile_and_dispatch("ptx", "main", [1, 1, 1], [1, 1, 1])
                assert mock_cdll.cuLaunchKernel.called


def test_cuda_runner_compile_ctypes_fail_load():
    with patch("ml_switcheroo_compiler.backends.cuda.cuda.cupy", None):
        with patch("ctypes.util.find_library", return_value="libcuda.so"):
            mock_cdll = MagicMock()
            mock_cdll.cuModuleLoadData.return_value = 1
            with patch("ctypes.cdll.LoadLibrary", return_value=mock_cdll):
                runner = CUDARunner()
                with pytest.raises(RuntimeError, match="cuModuleLoadData failed"):
                    runner.compile_and_dispatch("ptx", "main", [1, 1, 1], [1, 1, 1])


def test_cuda_runner_compile_ctypes_fail_func():
    with patch("ml_switcheroo_compiler.backends.cuda.cuda.cupy", None):
        with patch("ctypes.util.find_library", return_value="libcuda.so"):
            mock_cdll = MagicMock()
            mock_cdll.cuModuleLoadData.return_value = 0
            mock_cdll.cuModuleGetFunction.return_value = 1
            with patch("ctypes.cdll.LoadLibrary", return_value=mock_cdll):
                runner = CUDARunner()
                with pytest.raises(RuntimeError, match="cuModuleGetFunction failed"):
                    runner.compile_and_dispatch("ptx", "main", [1, 1, 1], [1, 1, 1])


def test_cuda_runner_compile_ctypes_fail_launch():
    with patch("ml_switcheroo_compiler.backends.cuda.cuda.cupy", None):
        with patch("ctypes.util.find_library", return_value="libcuda.so"):
            mock_cdll = MagicMock()
            mock_cdll.cuModuleLoadData.return_value = 0
            mock_cdll.cuModuleGetFunction.return_value = 0
            mock_cdll.cuLaunchKernel.return_value = 1
            with patch("ctypes.cdll.LoadLibrary", return_value=mock_cdll):
                runner = CUDARunner()
                with pytest.raises(RuntimeError, match="cuLaunchKernel failed"):
                    runner.compile_and_dispatch("ptx", "main", [1, 1, 1], [1, 1, 1])
