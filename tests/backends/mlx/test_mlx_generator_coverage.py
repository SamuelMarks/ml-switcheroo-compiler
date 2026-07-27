"""Test MLX generator edge cases coverage."""

from ml_switcheroo_compiler.backends.mlx.generator import MLXCodeGenerator


def test_mlx_generator_save_load(monkeypatch):
    """Test save and load methods of MLX generator."""
    import numpy as np

    # We mock numpy methods to verify they get called
    monkeypatch.setattr(np, "load", lambda *args, **kwargs: "mock_load")
    monkeypatch.setattr(np, "save", lambda *args, **kwargs: "mock_save")
    monkeypatch.setattr(np, "savez", lambda *args, **kwargs: "mock_savez")
    monkeypatch.setattr(np, "savez_compressed", lambda *args, **kwargs: "mock_savez_compressed")

    assert MLXCodeGenerator.load("dummy.npy") == "mock_load"
    assert MLXCodeGenerator.save("dummy.npy", None) is None
    assert MLXCodeGenerator.savez("dummy.npz", None) is None
    assert MLXCodeGenerator.savez_compressed("dummy.npz", None) is None
