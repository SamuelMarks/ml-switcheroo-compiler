"""Backend code generators for ML Switcheroo Compiler."""

from ml_switcheroo_compiler.backends.base_generator import BaseGenerator
from ml_switcheroo_compiler.backends.jax import JAXCodeGenerator
from ml_switcheroo_compiler.backends.keras import KerasCodeGenerator
from ml_switcheroo_compiler.backends.mlx import MLXCodeGenerator
from ml_switcheroo_compiler.backends.pytorch import PyTorchCodeGenerator
from ml_switcheroo_compiler.backends.registry import BackendRegistry, register_backend
from ml_switcheroo_compiler.backends.tensorflow import TensorFlowCodeGenerator

# Import optional or new backends to register them
from ml_switcheroo_compiler.backends import numpy

try:
    from ml_switcheroo_compiler.backends import cupy
except ImportError:
    pass

try:
    from ml_switcheroo_compiler.backends import dask
except ImportError:
    pass

__all__ = [
    "numpy",
    "cupy",
    "dask",
    "BackendRegistry",
    "BaseGenerator",
    "JAXCodeGenerator",
    "KerasCodeGenerator",
    "MLXCodeGenerator",
    "PyTorchCodeGenerator",
    "TensorFlowCodeGenerator",
    "register_backend",
]
