def test_rocm_load_yaml_dict_string(monkeypatch):
    import os
    from unittest.mock import MagicMock, patch

    from ml_switcheroo_compiler.backends.rocm.rocm import RocmCodeGenerator
    from ml_switcheroo_compiler.ir.core import IRGraph

    def mock_isdir(path):
        return True

    def mock_listdir(path):
        return ["test.yaml"]

    mock_data = {"templates": {"op1": "string body", "op2": ["not", "string", "or", "dict_with_body"]}}

    with patch("builtins.open", new_callable=MagicMock) as mock_open_func:
        mock_open_func.return_value.__enter__.return_value = "mock_file"
        with patch("yaml.safe_load", return_value=mock_data):
            monkeypatch.setattr(os.path, "isdir", mock_isdir)
            monkeypatch.setattr(os, "listdir", mock_listdir)

            graph = IRGraph()
            gen = RocmCodeGenerator(graph)
            assert gen.config.templates["op1"] == {"body": "string body"}
            assert gen.config.templates["op2"] == ["not", "string", "or", "dict_with_body"]


def test_rocm_load_fallback_yaml(monkeypatch):
    import os
    from unittest.mock import MagicMock, patch

    from ml_switcheroo_compiler.backends.rocm.rocm import RocmCodeGenerator
    from ml_switcheroo_compiler.ir.core import IRGraph

    def mock_isdir(path):
        return False

    def mock_exists(path):
        return True

    mock_data = {"templates": {"op3": {"body": "test"}}}

    with patch("builtins.open", new_callable=MagicMock) as mock_open_func:
        mock_open_func.return_value.__enter__.return_value = "mock_file"
        with patch("yaml.safe_load", return_value=mock_data):
            monkeypatch.setattr(os.path, "isdir", mock_isdir)
            monkeypatch.setattr(os.path, "exists", mock_exists)

            graph = IRGraph()
            gen = RocmCodeGenerator(graph)
            assert gen.config.templates["op3"] == {"body": "test"}


def test_rocm_runner_init_no_cupy(monkeypatch):
    from unittest.mock import MagicMock, patch

    from ml_switcheroo_compiler.backends.rocm.rocm import ROCmRunner

    monkeypatch.setattr("ml_switcheroo_compiler.backends.rocm.rocm.cupy", None)

    with patch("ctypes.util.find_library", return_value="libamdhip64.so"):
        mock_cdll = MagicMock()
        with patch("ctypes.cdll.LoadLibrary", return_value=mock_cdll):
            runner = ROCmRunner()
            assert runner.mode == "ctypes"
            mock_cdll.hipInit.assert_called_once_with(0)


def test_rocm_runner_init_ctypes_fail(monkeypatch):
    from unittest.mock import patch

    from ml_switcheroo_compiler.backends.rocm.rocm import ROCmRunner

    monkeypatch.setattr("ml_switcheroo_compiler.backends.rocm.rocm.cupy", None)
    with patch("ctypes.util.find_library", side_effect=Exception("mock fail")):
        runner = ROCmRunner()
        assert runner.rocm_lib is None


def test_rocm_runner_init_with_cupy():
    from unittest.mock import MagicMock, patch

    from ml_switcheroo_compiler.backends.rocm.rocm import ROCmRunner

    mock_cupy = MagicMock()
    with patch("ml_switcheroo_compiler.backends.rocm.rocm.cupy", mock_cupy):
        runner = ROCmRunner()
        assert runner.mode == "cupy"


def test_rocm_runner_allocate_cupy():
    from unittest.mock import MagicMock, patch

    from ml_switcheroo_compiler.backends.rocm.rocm import ROCmRunner

    mock_cupy = MagicMock()
    mock_mem = MagicMock()
    mock_mem.ptr = 12345
    mock_cupy.cuda.alloc.return_value = mock_mem
    with patch("ml_switcheroo_compiler.backends.rocm.rocm.cupy", mock_cupy):
        runner = ROCmRunner()
        ptr = runner.allocate_buffer(100)
        assert ptr.value == 12345


def test_rocm_runner_allocate_ctypes():
    from unittest.mock import MagicMock, patch

    from ml_switcheroo_compiler.backends.rocm.rocm import ROCmRunner

    with patch("ml_switcheroo_compiler.backends.rocm.rocm.cupy", None):
        with patch("ctypes.util.find_library", return_value="libamdhip64.so"):
            mock_cdll = MagicMock()
            mock_cdll.hipMalloc.return_value = 0
            with patch("ctypes.cdll.LoadLibrary", return_value=mock_cdll):
                runner = ROCmRunner()
                ptr = runner.allocate_buffer(100)
                assert mock_cdll.hipMalloc.called


def test_rocm_runner_allocate_ctypes_fail():
    from unittest.mock import MagicMock, patch

    import pytest

    from ml_switcheroo_compiler.backends.rocm.rocm import ROCmRunner

    with patch("ml_switcheroo_compiler.backends.rocm.rocm.cupy", None):
        with patch("ctypes.util.find_library", return_value="libamdhip64.so"):
            mock_cdll = MagicMock()
            mock_cdll.hipMalloc.return_value = 1  # error
            with patch("ctypes.cdll.LoadLibrary", return_value=mock_cdll):
                runner = ROCmRunner()
                with pytest.raises(RuntimeError, match="hipMalloc failed"):
                    runner.allocate_buffer(100)


def test_rocm_runner_allocate_ctypes_no_lib():
    from unittest.mock import patch

    from ml_switcheroo_compiler.backends.rocm.rocm import ROCmRunner

    with patch("ml_switcheroo_compiler.backends.rocm.rocm.cupy", None):
        with patch("ctypes.util.find_library", return_value=None):
            runner = ROCmRunner()
            ptr = runner.allocate_buffer(100)
            assert ptr.value is None


def test_rocm_runner_write_cupy():
    import ctypes
    from unittest.mock import MagicMock, patch

    from ml_switcheroo_compiler.backends.rocm.rocm import ROCmRunner

    mock_cupy = MagicMock()
    with patch("ml_switcheroo_compiler.backends.rocm.rocm.cupy", mock_cupy):
        runner = ROCmRunner()
        runner.write_buffer(ctypes.c_void_p(123), b"test")
        assert mock_cupy.ndarray.called


def test_rocm_runner_write_ctypes():
    import ctypes
    from unittest.mock import MagicMock, patch

    from ml_switcheroo_compiler.backends.rocm.rocm import ROCmRunner

    with patch("ml_switcheroo_compiler.backends.rocm.rocm.cupy", None):
        with patch("ctypes.util.find_library", return_value="libamdhip64.so"):
            mock_cdll = MagicMock()
            mock_cdll.hipMemcpyHtoD.return_value = 0
            with patch("ctypes.cdll.LoadLibrary", return_value=mock_cdll):
                runner = ROCmRunner()
                runner.write_buffer(ctypes.c_void_p(123), b"test")
                assert mock_cdll.hipMemcpyHtoD.called


def test_rocm_runner_write_ctypes_fail():
    import ctypes
    from unittest.mock import MagicMock, patch

    import pytest

    from ml_switcheroo_compiler.backends.rocm.rocm import ROCmRunner

    with patch("ml_switcheroo_compiler.backends.rocm.rocm.cupy", None):
        with patch("ctypes.util.find_library", return_value="libamdhip64.so"):
            mock_cdll = MagicMock()
            mock_cdll.hipMemcpyHtoD.return_value = 1
            with patch("ctypes.cdll.LoadLibrary", return_value=mock_cdll):
                runner = ROCmRunner()
                with pytest.raises(RuntimeError, match="hipMemcpyHtoD failed"):
                    runner.write_buffer(ctypes.c_void_p(123), b"test")


def test_rocm_runner_write_no_ptr():
    from ml_switcheroo_compiler.backends.rocm.rocm import ROCmRunner

    runner = ROCmRunner()
    assert runner.write_buffer(None, b"test") is None


def test_rocm_runner_read_cupy():
    import ctypes
    from unittest.mock import MagicMock, patch

    from ml_switcheroo_compiler.backends.rocm.rocm import ROCmRunner

    mock_cupy = MagicMock()
    mock_arr = MagicMock()
    mock_arr.get.return_value.tobytes.return_value = b"test"
    mock_cupy.ndarray.return_value = mock_arr
    with patch("ml_switcheroo_compiler.backends.rocm.rocm.cupy", mock_cupy):
        runner = ROCmRunner()
        res = runner.read_buffer(ctypes.c_void_p(123), 4)
        assert res == b"test"


def test_rocm_runner_read_ctypes():
    import ctypes
    from unittest.mock import MagicMock, patch

    from ml_switcheroo_compiler.backends.rocm.rocm import ROCmRunner

    with patch("ml_switcheroo_compiler.backends.rocm.rocm.cupy", None):
        with patch("ctypes.util.find_library", return_value="libamdhip64.so"):
            mock_cdll = MagicMock()
            mock_cdll.hipMemcpyDtoH.return_value = 0
            with patch("ctypes.cdll.LoadLibrary", return_value=mock_cdll):
                runner = ROCmRunner()
                res = runner.read_buffer(ctypes.c_void_p(123), 4)
                assert len(res) == 4


def test_rocm_runner_read_ctypes_fail():
    import ctypes
    from unittest.mock import MagicMock, patch

    import pytest

    from ml_switcheroo_compiler.backends.rocm.rocm import ROCmRunner

    with patch("ml_switcheroo_compiler.backends.rocm.rocm.cupy", None):
        with patch("ctypes.util.find_library", return_value="libamdhip64.so"):
            mock_cdll = MagicMock()
            mock_cdll.hipMemcpyDtoH.return_value = 1
            with patch("ctypes.cdll.LoadLibrary", return_value=mock_cdll):
                runner = ROCmRunner()
                with pytest.raises(RuntimeError, match="hipMemcpyDtoH failed"):
                    runner.read_buffer(ctypes.c_void_p(123), 4)


def test_rocm_runner_read_no_ptr():
    from ml_switcheroo_compiler.backends.rocm.rocm import ROCmRunner

    runner = ROCmRunner()
    assert runner.read_buffer(None, 4) == b"\x00" * 4


def test_rocm_runner_read_no_lib():
    import ctypes
    from unittest.mock import patch

    from ml_switcheroo_compiler.backends.rocm.rocm import ROCmRunner

    with patch("ml_switcheroo_compiler.backends.rocm.rocm.cupy", None):
        with patch("ctypes.util.find_library", return_value=None):
            runner = ROCmRunner()
            assert runner.read_buffer(ctypes.c_void_p(123), 4) == b"\x00" * 4


def test_rocm_runner_free_ctypes():
    import ctypes
    from unittest.mock import MagicMock, patch

    from ml_switcheroo_compiler.backends.rocm.rocm import ROCmRunner

    with patch("ml_switcheroo_compiler.backends.rocm.rocm.cupy", None):
        with patch("ctypes.util.find_library", return_value="libamdhip64.so"):
            mock_cdll = MagicMock()
            with patch("ctypes.cdll.LoadLibrary", return_value=mock_cdll):
                runner = ROCmRunner()
                runner.free_buffer(ctypes.c_void_p(123))
                assert mock_cdll.hipFree.called


def test_rocm_runner_free_cupy():
    import ctypes
    from unittest.mock import MagicMock, patch

    from ml_switcheroo_compiler.backends.rocm.rocm import ROCmRunner

    mock_cupy = MagicMock()
    with patch("ml_switcheroo_compiler.backends.rocm.rocm.cupy", mock_cupy):
        runner = ROCmRunner()
        runner.free_buffer(ctypes.c_void_p(123))
        assert True


def test_rocm_runner_free_no_ptr():
    from ml_switcheroo_compiler.backends.rocm.rocm import ROCmRunner

    runner = ROCmRunner()
    runner.free_buffer(None)


def test_rocm_runner_compile_cupy():
    from unittest.mock import MagicMock, mock_open, patch

    from ml_switcheroo_compiler.backends.rocm.rocm import ROCmRunner

    mock_cupy = MagicMock()
    with patch("ml_switcheroo_compiler.backends.rocm.rocm.cupy", mock_cupy):
        with patch("builtins.open", mock_open(read_data=b"code")):
            runner = ROCmRunner()
            runner.load_and_dispatch("ptx", "main", [1, 1, 1], [1, 1, 1])
            assert mock_cupy.RawModule.called


def test_rocm_runner_compile_ctypes():
    from unittest.mock import MagicMock, patch

    from ml_switcheroo_compiler.backends.rocm.rocm import ROCmRunner

    with patch("ml_switcheroo_compiler.backends.rocm.rocm.cupy", None):
        with patch("ctypes.util.find_library", return_value="libamdhip64.so"):
            mock_cdll = MagicMock()
            mock_cdll.hipModuleLoad.return_value = 0
            mock_cdll.hipModuleGetFunction.return_value = 0
            mock_cdll.hipModuleLaunchKernel.return_value = 0
            with patch("ctypes.cdll.LoadLibrary", return_value=mock_cdll):
                runner = ROCmRunner()
                runner.load_and_dispatch("ptx", "main", [1, 1, 1], [1, 1, 1])
                assert mock_cdll.hipModuleLaunchKernel.called


def test_rocm_runner_compile_ctypes_fail_load():
    from unittest.mock import MagicMock, patch

    import pytest

    from ml_switcheroo_compiler.backends.rocm.rocm import ROCmRunner

    with patch("ml_switcheroo_compiler.backends.rocm.rocm.cupy", None):
        with patch("ctypes.util.find_library", return_value="libamdhip64.so"):
            mock_cdll = MagicMock()
            mock_cdll.hipModuleLoad.return_value = 1
            with patch("ctypes.cdll.LoadLibrary", return_value=mock_cdll):
                runner = ROCmRunner()
                with pytest.raises(RuntimeError, match="hipModuleLoad failed"):
                    runner.load_and_dispatch("ptx", "main", [1, 1, 1], [1, 1, 1])


def test_rocm_runner_compile_ctypes_fail_func():
    from unittest.mock import MagicMock, patch

    import pytest

    from ml_switcheroo_compiler.backends.rocm.rocm import ROCmRunner

    with patch("ml_switcheroo_compiler.backends.rocm.rocm.cupy", None):
        with patch("ctypes.util.find_library", return_value="libamdhip64.so"):
            mock_cdll = MagicMock()
            mock_cdll.hipModuleLoad.return_value = 0
            mock_cdll.hipModuleGetFunction.return_value = 1
            with patch("ctypes.cdll.LoadLibrary", return_value=mock_cdll):
                runner = ROCmRunner()
                with pytest.raises(RuntimeError, match="hipModuleGetFunction failed"):
                    runner.load_and_dispatch("ptx", "main", [1, 1, 1], [1, 1, 1])


def test_rocm_runner_compile_ctypes_fail_launch():
    from unittest.mock import MagicMock, patch

    import pytest

    from ml_switcheroo_compiler.backends.rocm.rocm import ROCmRunner

    with patch("ml_switcheroo_compiler.backends.rocm.rocm.cupy", None):
        with patch("ctypes.util.find_library", return_value="libamdhip64.so"):
            mock_cdll = MagicMock()
            mock_cdll.hipModuleLoad.return_value = 0
            mock_cdll.hipModuleGetFunction.return_value = 0
            mock_cdll.hipModuleLaunchKernel.return_value = 1
            with patch("ctypes.cdll.LoadLibrary", return_value=mock_cdll):
                runner = ROCmRunner()
                with pytest.raises(RuntimeError, match="hipModuleLaunchKernel failed"):
                    runner.load_and_dispatch("ptx", "main", [1, 1, 1], [1, 1, 1])


def test_rocm_runner_compile_ctypes_no_lib():
    from unittest.mock import patch

    from ml_switcheroo_compiler.backends.rocm.rocm import ROCmRunner

    with patch("ml_switcheroo_compiler.backends.rocm.rocm.cupy", None):
        with patch("ctypes.util.find_library", return_value=None):
            runner = ROCmRunner()
            runner.load_and_dispatch("ptx", "main", [1, 1, 1], [1, 1, 1])
            assert True
