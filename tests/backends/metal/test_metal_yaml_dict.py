def test_metal_load_yaml_dict_string(monkeypatch):
    import os
    from unittest.mock import MagicMock, patch

    from ml_switcheroo_compiler.backends.metal.metal import MetalCodeGenerator
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
            gen = MetalCodeGenerator(graph)
            assert gen.config.templates["op1"] == {"body": "string body"}
            assert gen.config.templates["op2"] == ["not", "string", "or", "dict_with_body"]


def test_metal_load_fallback_yaml(monkeypatch):
    import os
    from unittest.mock import MagicMock, patch

    from ml_switcheroo_compiler.backends.metal.metal import MetalCodeGenerator
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
            gen = MetalCodeGenerator(graph)
            assert gen.config.templates["op3"] == {"body": "test"}


def test_metal_runner_init_coverage(monkeypatch):
    from unittest.mock import MagicMock, patch

    from ml_switcheroo_compiler.backends.metal.metal import MetalRunner

    with patch("ctypes.util.find_library", side_effect=["objc_lib", None]):
        with patch("ctypes.cdll.LoadLibrary", return_value=MagicMock()):
            runner = MetalRunner()
            assert runner.objc is not None
            assert runner.metal is None

    with patch("ctypes.util.find_library", side_effect=[None, "metal_lib"]):
        with patch("ctypes.cdll.LoadLibrary", return_value=MagicMock()):
            runner = MetalRunner()
            assert runner.objc is None
            assert runner.metal is not None

    with patch("ctypes.util.find_library", side_effect=Exception("mock")):
        runner = MetalRunner()
        assert runner.objc is None
        assert runner.metal is None


def test_metal_runner_compile_dispatch():
    from unittest.mock import MagicMock, patch

    from ml_switcheroo_compiler.backends.metal.metal import MetalRunner

    with patch("ctypes.util.find_library", side_effect=[None, "metal_lib"]):
        with patch("ctypes.cdll.LoadLibrary", return_value=MagicMock()):
            runner = MetalRunner()
            runner.compile_and_dispatch("code", "entry", [1, 1, 1])
            assert runner.allocate_buffer(100).value is None
            runner.write_buffer(None, b"test")


def test_metal_runner_methods():

    from ml_switcheroo_compiler.backends.metal.metal import MetalRunner

    runner = MetalRunner()
    # Mock self.metal is None
    runner.metal = None
    assert runner.allocate_buffer(100) is None
    assert runner.read_buffer(None, 4) == b""
