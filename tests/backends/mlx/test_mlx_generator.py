"""Test MLX generator edge cases coverage."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from ml_switcheroo_compiler.backends.mlx.generator import MLXCodeGenerator

if TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import EagerTensorProtocol, EagerValue


def test_mlx_generator_save_load(monkeypatch: EagerTensorProtocol) -> None:
    """Test save and load methods of MLX generator."""

    # We mock mlx.core methods to verify they get called
    class MockMX:
        """Test class MockMX."""

        @staticmethod
        def load(*args: EagerValue, **kwargs: EagerValue) -> EagerValue:
            """Test function load."""
            return "mock_load"

        @staticmethod
        def save(*args: EagerValue, **kwargs: EagerValue) -> EagerValue:
            """Test function save."""
            return "mock_save"

        @staticmethod
        def save_safetensors(*args: EagerValue, **kwargs: EagerValue) -> EagerValue:
            """Test function save_safetensors."""
            return "mock_save_safetensors"

    # Mock the actual module import within MLXCodeGenerator.load/save
    import sys
    import types

    # Create a mock module for mlx.core
    mock_mlx_core = types.ModuleType("mlx.core")
    setattr(mock_mlx_core, "load", cast("EagerTensorProtocol", getattr(MockMX, "load")))
    setattr(mock_mlx_core, "save", cast("EagerTensorProtocol", getattr(MockMX, "save")))
    setattr(mock_mlx_core, "save_safetensors", cast("EagerTensorProtocol", getattr(MockMX, "save_safetensors")))

    mock_mlx = types.ModuleType("mlx")
    setattr(mock_mlx, "core", mock_mlx_core)

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
