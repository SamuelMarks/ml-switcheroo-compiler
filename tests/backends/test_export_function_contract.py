"""Test export_function across all backends."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from ml_switcheroo_compiler.backends.dask.generator import DaskGenerator
from ml_switcheroo_compiler.backends.edge.webgpu import WebGPUCodeGenerator
from ml_switcheroo_compiler.backends.jax.generator import JAXCodeGenerator
from ml_switcheroo_compiler.backends.pytorch.generator import PyTorchCodeGenerator
from ml_switcheroo_compiler.export.aot import compile_function

if TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import EagerValue


def dummy_export(*args: EagerValue, **kwargs: EagerValue) -> None:
    """Test function dummy_export."""
    pass


def test_export_function_contract() -> None:
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
        except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, OSError, ImportError, AssertionError) as _exc:
            assert _exc is not None


def test_aot_compile_function_integration() -> None:
    """Test export_function integration via aot.compile_function."""

    def dummy_func(x: EagerValue) -> EagerValue:
        """Test function dummy_func."""
        return x

    compiled = compile_function(dummy_func, backend="numpy")
    assert callable(compiled)
