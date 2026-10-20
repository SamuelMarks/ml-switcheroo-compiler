"""Test export_function across all backends."""

from __future__ import annotations

import typing

from ml_switcheroo_compiler.backends.dask.generator import DaskGenerator
from ml_switcheroo_compiler.backends.edge.webgpu import WebGPUCodeGenerator
from ml_switcheroo_compiler.backends.jax.generator import JAXCodeGenerator
from ml_switcheroo_compiler.backends.pytorch.generator import PyTorchCodeGenerator
from ml_switcheroo_compiler.export.aot import compile_function

if typing.TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import EagerValue


def dummy_export(*args: EagerValue, **kwargs: EagerValue) -> None:
    """Test function dummy_export.

    Args:
        args: Test parameter `args`.
        kwargs: Test parameter `kwargs`.
    """
    pass


def test_export_function_contract() -> None:
    """Test that all backends implement export_function without error."""
    backends = [PyTorchCodeGenerator, JAXCodeGenerator, DaskGenerator, WebGPUCodeGenerator]
    for backend_cls in backends:
        assert hasattr(backend_cls, "export_function")
        backend_cls.export_function(format="onnx")


def test_aot_compile_function_integration() -> None:
    """Test export_function integration via aot.compile_function."""

    def dummy_func(x: EagerValue) -> EagerValue:
        """Test function dummy_func.

        Args:
            x: Test parameter `x`.
        """
        return x

    compiled = compile_function(dummy_func, backend="numpy")
    assert callable(compiled)
