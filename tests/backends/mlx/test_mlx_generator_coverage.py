"""Test MLX generator edge cases coverage."""

from ml_switcheroo_compiler.backends.mlx.generator import MLXCodeGenerator


def test_mlx_generator_save_load(monkeypatch):
    """Test save and load methods of MLX generator."""

    # We mock mlx.core methods to verify they get called
    class MockMX:
        @staticmethod
        def load(*args, **kwargs):
            return "mock_load"

        @staticmethod
        def save(*args, **kwargs):
            return "mock_save"

        @staticmethod
        def save_safetensors(*args, **kwargs):
            return "mock_save_safetensors"

    # Mock the actual module import within MLXCodeGenerator.load/save
    import sys
    import types

    # Create a mock module for mlx.core
    mock_mlx_core = types.ModuleType("mlx.core")
    mock_mlx_core.load = MockMX.load
    mock_mlx_core.save = MockMX.save
    mock_mlx_core.save_safetensors = MockMX.save_safetensors

    mock_mlx = types.ModuleType("mlx")
    mock_mlx.core = mock_mlx_core

    # Use monkeypatch.setitem on sys.modules
    monkeypatch.setitem(sys.modules, "mlx.core", mock_mlx_core)
    monkeypatch.setitem(sys.modules, "mlx", mock_mlx)

    pass

    # Patch just in case real mlx exists
    if "mlx.core" in sys.modules:
        monkeypatch.setattr(sys.modules["mlx.core"], "load", MockMX.load, raising=False)
        monkeypatch.setattr(sys.modules["mlx.core"], "save", MockMX.save, raising=False)
        monkeypatch.setattr(sys.modules["mlx.core"], "save_safetensors", MockMX.save_safetensors, raising=False)

    assert MLXCodeGenerator.load("dummy.npy") == "mock_load"
    assert MLXCodeGenerator.save("dummy.npy", None) is None
    assert MLXCodeGenerator.savez("dummy.npz", None) is None
    assert MLXCodeGenerator.savez_compressed("dummy.npz", None) is None
