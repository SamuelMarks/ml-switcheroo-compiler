"""Test MLX generator edge cases coverage."""

from __future__ import annotations

import typing

from ml_switcheroo_compiler.backends.mlx.generator import MLXCodeGenerator
from tests.testing_utils import ensure_eager_tensor, ensure_mutable

if typing.TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import EagerTensorProtocol, EagerValue


def test_mlx_generator_save_load(monkeypatch: EagerTensorProtocol) -> None:
    """Test save and load methods of MLX generator.

    Args:
        monkeypatch: Test parameter `monkeypatch`.
    """

    # We mock mlx.core methods to verify they get called
    class MockMX:
        """Test class MockMX."""

        @staticmethod
        def load(*args: EagerValue, **kwargs: EagerValue) -> EagerValue:
            """Test function load.

            Args:
                args: Test parameter `args`.
                kwargs: Test parameter `kwargs`.

            Returns:
                EagerValue: Computed output value from `load`.
            """
            return "mock_load"

        @staticmethod
        def save(*args: EagerValue, **kwargs: EagerValue) -> EagerValue:
            """Test function save.

            Args:
                args: Test parameter `args`.
                kwargs: Test parameter `kwargs`.

            Returns:
                EagerValue: Computed output value from `save`.
            """
            return "mock_save"

        @staticmethod
        def save_safetensors(*args: EagerValue, **kwargs: EagerValue) -> EagerValue:
            """Test function save_safetensors.

            Args:
                args: Test parameter `args`.
                kwargs: Test parameter `kwargs`.

            Returns:
                EagerValue: Computed output value from `save_safetensors`.
            """
            return "mock_save_safetensors"

    # Mock the actual module import within MLXCodeGenerator.load/save
    import sys
    import types

    # Create a mock module for mlx.core
    mock_mlx_core = types.ModuleType("mlx.core")
    ensure_mutable(mock_mlx_core).load = ensure_eager_tensor(ensure_eager_tensor(MockMX).load)
    ensure_mutable(mock_mlx_core).save = ensure_eager_tensor(ensure_eager_tensor(MockMX).save)
    ensure_mutable(mock_mlx_core).save_safetensors = ensure_eager_tensor(ensure_eager_tensor(MockMX).save_safetensors)

    mock_mlx = types.ModuleType("mlx")
    ensure_mutable(mock_mlx).core = ensure_eager_tensor(mock_mlx_core)

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


def test_mlx_generator_zeros_and_item(monkeypatch: EagerTensorProtocol) -> None:
    """Test MLXCodeGenerator.zeros, array, asarray, and item classmethods.

    Args:
        monkeypatch: Test parameter `monkeypatch`.
    """
    import types

    import numpy as np

    import ml_switcheroo_compiler.backends.mlx.types as mlx_types

    if ensure_eager_tensor(mlx_types.mx) is None:
        mock_mx = types.ModuleType("mlx.core")
        ensure_mutable(mock_mx).zeros = ensure_eager_tensor(np.zeros)
        ensure_mutable(mock_mx).array = ensure_eager_tensor(np.array)
        monkeypatch.setattr(mlx_types, "mx", mock_mx)

    z_arr = MLXCodeGenerator.zeros((1,))
    assert MLXCodeGenerator.item(z_arr) == 0.0
    a_arr = MLXCodeGenerator.array([1.0])
    assert MLXCodeGenerator.item(a_arr) == 1.0
    as_arr = MLXCodeGenerator.asarray([2.0])
    assert MLXCodeGenerator.item(as_arr) == 2.0
