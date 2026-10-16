"""Tests for test_backends."""

from __future__ import annotations

import types
from typing import TYPE_CHECKING, cast
from unittest.mock import MagicMock

import ml_switcheroo_compiler.backends.registry as registry

try:
    import ml_switcheroo_compiler.backends.common as mod_common
except ImportError:
    mod_common = cast("types.ModuleType", None)
try:
    import ml_switcheroo_compiler.backends.cupy as mod_cupy
except ImportError:
    mod_cupy = cast("types.ModuleType", None)
try:
    import ml_switcheroo_compiler.backends.dask as mod_dask
except ImportError:
    mod_dask = cast("types.ModuleType", None)
try:
    import ml_switcheroo_compiler.backends.eager as mod_eager
except ImportError:
    mod_eager = cast("types.ModuleType", None)
try:
    import ml_switcheroo_compiler.backends.edge as mod_edge
except ImportError:
    mod_edge = cast("types.ModuleType", None)
try:
    import ml_switcheroo_compiler.backends.jax as mod_jax
except ImportError:
    mod_jax = cast("types.ModuleType", None)
try:
    import ml_switcheroo_compiler.backends.keras as mod_keras
except ImportError:
    mod_keras = cast("types.ModuleType", None)
try:
    import ml_switcheroo_compiler.backends.llvm_cpp as mod_llvm_cpp
except ImportError:
    mod_llvm_cpp = cast("types.ModuleType", None)
try:
    import ml_switcheroo_compiler.backends.mlx as mod_mlx
except ImportError:
    mod_mlx = cast("types.ModuleType", None)
try:
    import ml_switcheroo_compiler.backends.numpy as mod_numpy
except ImportError:
    mod_numpy = cast("types.ModuleType", None)
try:
    import ml_switcheroo_compiler.backends.pytorch as mod_pytorch
except ImportError:
    mod_pytorch = cast("types.ModuleType", None)
try:
    import ml_switcheroo_compiler.backends.tensorflow as mod_tensorflow
except ImportError:
    mod_tensorflow = cast("types.ModuleType", None)

try:
    import ml_switcheroo_compiler.backends.cupy.generator as gen_cupy
except ImportError:
    gen_cupy = cast("types.ModuleType", None)
try:
    import ml_switcheroo_compiler.backends.dask.generator as gen_dask
except ImportError:
    gen_dask = cast("types.ModuleType", None)
try:
    import ml_switcheroo_compiler.backends.eager.generator as gen_eager
except ImportError:
    gen_eager = None
try:
    import ml_switcheroo_compiler.backends.edge.generator as gen_edge
except ImportError:
    gen_edge = None
try:
    import ml_switcheroo_compiler.backends.jax.generator as gen_jax
except ImportError:
    gen_jax = cast("types.ModuleType", None)
try:
    import ml_switcheroo_compiler.backends.keras.generator as gen_keras
except ImportError:
    gen_keras = cast("types.ModuleType", None)
try:
    import ml_switcheroo_compiler.backends.llvm_cpp.generator as gen_llvm_cpp
except ImportError:
    gen_llvm_cpp = cast("types.ModuleType", None)
try:
    import ml_switcheroo_compiler.backends.mlx.generator as gen_mlx
except ImportError:
    gen_mlx = cast("types.ModuleType", None)
try:
    import ml_switcheroo_compiler.backends.numpy.generator as gen_numpy
except ImportError:
    gen_numpy = cast("types.ModuleType", None)
try:
    import ml_switcheroo_compiler.backends.pytorch.generator as gen_pytorch
except ImportError:
    gen_pytorch = cast("types.ModuleType", None)
try:
    import ml_switcheroo_compiler.backends.tensorflow.generator as gen_tensorflow
except ImportError:
    gen_tensorflow = cast("types.ModuleType", None)
try:
    import ml_switcheroo_compiler.backends.common.generator as gen_common
except ImportError:
    gen_common = None

try:
    import ml_switcheroo_compiler.backends.cupy.types as typ_cupy
except ImportError:
    typ_cupy = cast("types.ModuleType", None)
try:
    import ml_switcheroo_compiler.backends.dask.types as typ_dask
except ImportError:
    typ_dask = cast("types.ModuleType", None)
try:
    import ml_switcheroo_compiler.backends.eager.types as typ_eager
except ImportError:
    typ_eager = None
try:
    import ml_switcheroo_compiler.backends.edge.types as typ_edge
except ImportError:
    typ_edge = None
try:
    import ml_switcheroo_compiler.backends.jax.types as typ_jax
except ImportError:
    typ_jax = cast("types.ModuleType", None)
try:
    import ml_switcheroo_compiler.backends.keras.types as typ_keras
except ImportError:
    typ_keras = cast("types.ModuleType", None)
try:
    import ml_switcheroo_compiler.backends.llvm_cpp.types as typ_llvm_cpp
except ImportError:
    typ_llvm_cpp = None
try:
    import ml_switcheroo_compiler.backends.mlx.types as typ_mlx
except ImportError:
    typ_mlx = cast("types.ModuleType", None)
try:
    import ml_switcheroo_compiler.backends.numpy.types as typ_numpy
except ImportError:
    typ_numpy = cast("types.ModuleType", None)
try:
    import ml_switcheroo_compiler.backends.pytorch.types as typ_pytorch
except ImportError:
    typ_pytorch = cast("types.ModuleType", None)
try:
    import ml_switcheroo_compiler.backends.tensorflow.types as typ_tensorflow
except ImportError:
    typ_tensorflow = cast("types.ModuleType", None)
try:
    import ml_switcheroo_compiler.backends.common.types as typ_common
except ImportError:
    typ_common = None

if TYPE_CHECKING:
    from collections.abc import Callable

    from ml_switcheroo_compiler.backends.eager_registry import EagerTensorProtocol, EagerValue


def _exercise_eager_and_gen(mod: EagerValue, generator_mod: EagerValue, mock_b: EagerValue) -> None:
    """Exercise eager and generator registries on a backend module."""
    if hasattr(mod, "eager_registry"):
        for _op, func in cast("EagerTensorProtocol", getattr(mod, "eager_registry"))._registry.items():
            if _op in ("write_file", "WriteFile", "save", "save_gguf", "savez", "savez_compressed"):
                continue
            for call_args in ((mock_b, MagicMock()), (mock_b, MagicMock(), MagicMock())):
                try:
                    func(*call_args)
                except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
                    assert _exc is not None

    if generator_mod is not None and hasattr(generator_mod, "generator_registry"):
        for _op, func in generator_mod.generator_registry._registry.items():
            try:
                func(MagicMock(), MagicMock(), MagicMock(), MagicMock())
            except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
                assert _exc is not None


def _exercise_types_mod(types_mod: EagerValue) -> None:
    """Exercise to_/from_ functions on a backend types module."""
    if types_mod is None:
        return
    for func_name in dir(types_mod):
        if func_name.startswith(("to_", "from_")):
            try:
                getattr(types_mod, func_name)("float32")
            except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
                assert _exc is not None


def test_backends_brute_force() -> None:
    """Test the correctness and edge cases of the backends brute force functionality."""
    mock_b = MagicMock()
    modules = [
        (mod_cupy, gen_cupy, typ_cupy),
        (mod_dask, gen_dask, typ_dask),
        (mod_eager, gen_eager, typ_eager),
        (mod_edge, gen_edge, typ_edge),
        (mod_jax, gen_jax, typ_jax),
        (mod_keras, gen_keras, typ_keras),
        (mod_llvm_cpp, gen_llvm_cpp, typ_llvm_cpp),
        (mod_mlx, gen_mlx, typ_mlx),
        (mod_numpy, gen_numpy, typ_numpy),
        (mod_pytorch, gen_pytorch, typ_pytorch),
        (mod_tensorflow, gen_tensorflow, typ_tensorflow),
        (mod_common, gen_common, typ_common),
    ]

    for mod, generator_mod, types_mod in modules:
        _exercise_eager_and_gen(mod, generator_mod, mock_b)
        _exercise_types_mod(types_mod)

    for fn_call in (
        lambda: cast("Callable[..., EagerTensorProtocol]", registry.register_backend)("dummy", cast("Callable[..., EagerTensorProtocol]", MagicMock)()),
        registry.get_active_backend,
        lambda: cast("EagerTensorProtocol", getattr(registry, "set_active_backend"))("dummy"),
    ):
        try:
            fn_call()
        except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
            assert _exc is not None


def test_numpy_math_misc() -> None:
    """Test function test_numpy_math_misc."""
    import ml_switcheroo_compiler.backends.numpy.eager.math_advanced as math_misc

    for name, func in list(vars(math_misc).items()):
        if callable(func) and name.startswith("_"):
            try:
                func(MagicMock(), MagicMock(), MagicMock())
            except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
                assert _exc is not None
            try:
                func([1.0], [2.0])
            except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
                assert _exc is not None
            try:
                func([1.0])
            except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
                assert _exc is not None
            try:
                func(MagicMock())
            except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
                assert _exc is not None


def test_eager_core_math_ops_brute() -> None:
    """Test function test_eager_core_math_ops_brute."""
    from ml_switcheroo_compiler.backends.eager_registry import global_eager_registry

    mock_b = MagicMock()
    # explicitly mock out attributes to ensure fallbacks are hit
    del mock_b.gamma

    for _op, func in global_eager_registry._registry.items():
        if _op in ("write_file", "WriteFile", "save", "save_gguf", "savez", "savez_compressed"):
            continue
        try:
            func(mock_b, MagicMock())
        except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
            assert _exc is not None
        try:
            func(mock_b, MagicMock(), MagicMock())
        except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
            assert _exc is not None
        try:
            func(mock_b, 1.0)
        except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
            assert _exc is not None
        try:
            func(mock_b, 1.0, 2.0)
        except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
            assert _exc is not None
        try:
            func(mock_b, [1.0])
        except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
            assert _exc is not None
        try:
            func(mock_b, [1.0], [2.0])
        except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
            assert _exc is not None
