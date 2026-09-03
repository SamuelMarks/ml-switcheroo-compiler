from unittest.mock import MagicMock, patch


def test_cuda_load_yaml_dict_string(monkeypatch):
    import os

    from ml_switcheroo_compiler.backends.cuda.cuda import CudaCodeGenerator
    from ml_switcheroo_compiler.ir.core import IRGraph

    def mock_isdir(path):
        return True

    def mock_listdir(path):
        return ["test.yaml"]

    mock_data = {"templates": {"op1": "string body", "op2": ["not", "string", "or", "dict_with_body"]}}

    # We patch open to return our mock yaml
    with patch("builtins.open", new_callable=MagicMock) as mock_open_func:
        mock_open_func.return_value.__enter__.return_value = "mock_file"
        with patch("yaml.safe_load", return_value=mock_data):
            monkeypatch.setattr(os.path, "isdir", mock_isdir)
            monkeypatch.setattr(os, "listdir", mock_listdir)

            graph = IRGraph()
            gen = CudaCodeGenerator(graph)
            assert gen.config.templates["op1"] == {"body": "string body"}
            assert gen.config.templates["op2"] == ["not", "string", "or", "dict_with_body"]


def test_cuda_load_fallback_yaml(monkeypatch):
    import os

    from ml_switcheroo_compiler.backends.cuda.cuda import CudaCodeGenerator
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
            gen = CudaCodeGenerator(graph)
            assert gen.config.templates["op3"] == {"body": "test"}


def test_cuda_runner_init_ctypes_except(monkeypatch):
    from ml_switcheroo_compiler.backends.cuda.cuda import CUDARunner

    monkeypatch.setattr("ml_switcheroo_compiler.backends.cuda.cuda.cupy", None)
    with patch("ctypes.util.find_library", side_effect=Exception("mocked exception")):
        runner = CUDARunner()
        assert runner.mode == "ctypes"


def test_cuda_runner_read_ctypes_return_zeros():
    import ctypes

    from ml_switcheroo_compiler.backends.cuda.cuda import CUDARunner

    with patch("ml_switcheroo_compiler.backends.cuda.cuda.cupy", None):
        with patch("ctypes.util.find_library", return_value="libcuda.so"):
            mock_cdll = MagicMock()
            # If we delete cuMemcpyDtoH_v2 it will fall through
            with patch("ctypes.cdll.LoadLibrary", return_value=mock_cdll):
                runner = CUDARunner()
                # Clear cuda_lib to hit the end
                runner.cuda_lib = None
                res = runner.read_buffer(ctypes.c_void_p(123), 4)
                assert res == b"\x00" * 4
