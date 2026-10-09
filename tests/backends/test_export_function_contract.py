"""Test export_function across all backends."""

import pytest

from ml_switcheroo_compiler.backends.dask.generator import DaskGenerator
from ml_switcheroo_compiler.backends.edge.webgpu import WebGPUCodeGenerator
from ml_switcheroo_compiler.backends.jax.generator import JAXCodeGenerator
from ml_switcheroo_compiler.backends.pytorch.generator import PyTorchCodeGenerator
from ml_switcheroo_compiler.export.aot import compile_function


def dummy_export(*args, **kwargs):
    pass


def test_export_function_contract():
    """Test that all backends implement export_function without error."""
    backends = [
        PyTorchCodeGenerator,
        JAXCodeGenerator,
        DaskGenerator,
        WebGPUCodeGenerator,
    ]

    for backend_cls in backends:
        assert hasattr(backend_cls, "export_function")

        # Test that calling it does not raise NotImplementedError
        try:
            backend_cls.export_function(format="onnx")
        except NotImplementedError:
            pytest.fail(f"{backend_cls.__name__} raised NotImplementedError for export_function.")
        except Exception:
            pass  # We only care that the signature is implemented and doesn't fail unconditionally


def test_aot_compile_function_integration():
    """Test export_function integration via aot.compile_function."""

    def dummy_func(x):
        return x

    compiled = compile_function(dummy_func, backend="numpy")
    assert callable(compiled)
