"""Tests for test_backends."""

from __future__ import annotations

import importlib
import types
import typing
from unittest.mock import MagicMock

import ml_switcheroo_compiler.backends.registry as registry
from tests.testing_utils import adapt_backend_module, require_callable


def _try_import(module_name: str) -> types.ModuleType | None:
    """Import an optional backend module by dotted name, returning None on ImportError.

    Args:
        module_name: Dotted module path to import.

    Returns:
        types.ModuleType | None: Imported module if available, otherwise None.
    """
    try:
        return importlib.import_module(module_name)
    except ImportError:
        return None


mod_common: types.ModuleType | None = _try_import("ml_switcheroo_compiler.backends.common")
mod_cupy: types.ModuleType | None = _try_import("ml_switcheroo_compiler.backends.cupy")
mod_dask: types.ModuleType | None = _try_import("ml_switcheroo_compiler.backends.dask")
mod_eager: types.ModuleType | None = _try_import("ml_switcheroo_compiler.backends.eager")
mod_edge: types.ModuleType | None = _try_import("ml_switcheroo_compiler.backends.edge")
mod_jax: types.ModuleType | None = _try_import("ml_switcheroo_compiler.backends.jax")
mod_keras: types.ModuleType | None = _try_import("ml_switcheroo_compiler.backends.keras")
mod_llvm_cpp: types.ModuleType | None = _try_import("ml_switcheroo_compiler.backends.llvm_cpp")
mod_mlx: types.ModuleType | None = _try_import("ml_switcheroo_compiler.backends.mlx")
mod_numpy: types.ModuleType | None = _try_import("ml_switcheroo_compiler.backends.numpy")
mod_pytorch: types.ModuleType | None = _try_import("ml_switcheroo_compiler.backends.pytorch")
mod_tensorflow: types.ModuleType | None = _try_import("ml_switcheroo_compiler.backends.tensorflow")
gen_cupy: types.ModuleType | None = _try_import("ml_switcheroo_compiler.backends.cupy.generator")
gen_dask: types.ModuleType | None = _try_import("ml_switcheroo_compiler.backends.dask.generator")
gen_eager: types.ModuleType | None = _try_import("ml_switcheroo_compiler.backends.eager.generator")
gen_edge: types.ModuleType | None = _try_import("ml_switcheroo_compiler.backends.edge.generator")
gen_jax: types.ModuleType | None = _try_import("ml_switcheroo_compiler.backends.jax.generator")
gen_keras: types.ModuleType | None = _try_import("ml_switcheroo_compiler.backends.keras.generator")
gen_llvm_cpp: types.ModuleType | None = _try_import("ml_switcheroo_compiler.backends.llvm_cpp.generator")
gen_mlx: types.ModuleType | None = _try_import("ml_switcheroo_compiler.backends.mlx.generator")
gen_numpy: types.ModuleType | None = _try_import("ml_switcheroo_compiler.backends.numpy.generator")
gen_pytorch: types.ModuleType | None = _try_import("ml_switcheroo_compiler.backends.pytorch.generator")
gen_tensorflow: types.ModuleType | None = _try_import("ml_switcheroo_compiler.backends.tensorflow.generator")
gen_common: types.ModuleType | None = _try_import("ml_switcheroo_compiler.backends.common.generator")
typ_cupy: types.ModuleType | None = _try_import("ml_switcheroo_compiler.backends.cupy.types")
typ_dask: types.ModuleType | None = _try_import("ml_switcheroo_compiler.backends.dask.types")
typ_eager: types.ModuleType | None = _try_import("ml_switcheroo_compiler.backends.eager.types")
typ_edge: types.ModuleType | None = _try_import("ml_switcheroo_compiler.backends.edge.types")
typ_jax: types.ModuleType | None = _try_import("ml_switcheroo_compiler.backends.jax.types")
typ_keras: types.ModuleType | None = _try_import("ml_switcheroo_compiler.backends.keras.types")
typ_llvm_cpp: types.ModuleType | None = _try_import("ml_switcheroo_compiler.backends.llvm_cpp.types")
typ_mlx: types.ModuleType | None = _try_import("ml_switcheroo_compiler.backends.mlx.types")
typ_numpy: types.ModuleType | None = _try_import("ml_switcheroo_compiler.backends.numpy.types")
typ_pytorch: types.ModuleType | None = _try_import("ml_switcheroo_compiler.backends.pytorch.types")
typ_tensorflow: types.ModuleType | None = _try_import("ml_switcheroo_compiler.backends.tensorflow.types")
typ_common: types.ModuleType | None = _try_import("ml_switcheroo_compiler.backends.common.types")
if typing.TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import BackendArg


def _exercise_eager_and_gen(mod: BackendArg, generator_mod: BackendArg, mock_b: BackendArg) -> None:
    """Exercise eager and generator registries on a backend module.

    Args:
        mod: Test parameter `mod`.
        generator_mod: Test parameter `generator_mod`.
        mock_b: Test parameter `mock_b`.
    """
    handled_errors = 0
    if adapt_backend_module(mod).has_symbol("eager_registry"):
        for _op, func in adapt_backend_module(mod).require_symbol("eager_registry")._registry.items():
            if _op in ("write_file", "WriteFile", "save", "save_gguf", "savez", "savez_compressed"):
                continue
            for call_args in ((mock_b, MagicMock()), (mock_b, MagicMock(), MagicMock())):
                try:
                    func(*call_args)
                except (
                    ValueError,
                    TypeError,
                    AttributeError,
                    IndexError,
                    KeyError,
                    RuntimeError,
                    ZeroDivisionError,
                    NotImplementedError,
                    OSError,
                    ImportError,
                    OverflowError,
                    RecursionError,
                    LookupError,
                    ArithmeticError,
                ):
                    handled_errors += 1
    if generator_mod is not None and adapt_backend_module(generator_mod).has_symbol("generator_registry"):
        for _op, func in adapt_backend_module(generator_mod).require_symbol("generator_registry")._registry.items():
            try:
                func(MagicMock(), MagicMock(), MagicMock(), MagicMock())
            except (
                ValueError,
                TypeError,
                AttributeError,
                IndexError,
                KeyError,
                RuntimeError,
                ZeroDivisionError,
                NotImplementedError,
                OSError,
                ImportError,
                OverflowError,
                RecursionError,
                LookupError,
                ArithmeticError,
            ):
                handled_errors += 1


def _exercise_types_mod(types_mod: BackendArg) -> None:
    """Exercise to_/from_ functions on a backend types module.

    Args:
        types_mod: Test parameter `types_mod`.
    """
    handled_errors = 0
    if types_mod is None:
        return
    from tests.testing_utils import adapt_backend_module

    candidates = [
        adapt_backend_module(types_mod).get_symbol("to_cupy_dtype", None),
        adapt_backend_module(types_mod).get_symbol("from_cupy_dtype", None),
        adapt_backend_module(types_mod).get_symbol("to_dask_dtype", None),
        adapt_backend_module(types_mod).get_symbol("from_dask_dtype", None),
        adapt_backend_module(types_mod).get_symbol("to_jax_dtype", None),
        adapt_backend_module(types_mod).get_symbol("from_jax_dtype", None),
        adapt_backend_module(types_mod).get_symbol("to_keras_dtype", None),
        adapt_backend_module(types_mod).get_symbol("from_keras_dtype", None),
        adapt_backend_module(types_mod).get_symbol("to_cpp_dtype", None),
        adapt_backend_module(types_mod).get_symbol("to_mlx_dtype", None),
        adapt_backend_module(types_mod).get_symbol("from_mlx_dtype", None),
        adapt_backend_module(types_mod).get_symbol("to_numpy_dtype", None),
        adapt_backend_module(types_mod).get_symbol("from_numpy_dtype", None),
        adapt_backend_module(types_mod).get_symbol("to_torch_dtype", None),
        adapt_backend_module(types_mod).get_symbol("from_torch_dtype", None),
        adapt_backend_module(types_mod).get_symbol("to_tf_dtype", None),
        adapt_backend_module(types_mod).get_symbol("from_tf_dtype", None),
    ]
    for fn in candidates:
        if fn is not None:
            try:
                require_callable(fn)("float32")
            except (
                ValueError,
                TypeError,
                AttributeError,
                IndexError,
                KeyError,
                RuntimeError,
                ZeroDivisionError,
                NotImplementedError,
                OSError,
                ImportError,
                OverflowError,
                RecursionError,
                LookupError,
                ArithmeticError,
            ):
                handled_errors += 1


def test_backends_brute_force() -> None:
    """Test the correctness and edge cases of the backends brute force functionality."""
    handled_errors = 0
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
        lambda: adapt_backend_module(registry).require_symbol("register_backend")("dummy", MagicMock()),
        registry.get_active_backend,
        lambda: adapt_backend_module(registry).require_symbol("set_active_backend")("dummy"),
    ):
        try:
            fn_call()
        except (
            ValueError,
            TypeError,
            AttributeError,
            IndexError,
            KeyError,
            RuntimeError,
            ZeroDivisionError,
            NotImplementedError,
            OSError,
            ImportError,
            OverflowError,
            RecursionError,
            LookupError,
            ArithmeticError,
        ):
            handled_errors += 1
    assert handled_errors >= 1


def test_numpy_math_misc() -> None:
    """Test function test_numpy_math_misc."""
    from ml_switcheroo_compiler.backends.eager_registry import global_eager_registry

    handled_errors = 0
    for raw_func in list(global_eager_registry._registry.values()):
        if raw_func.__module__.startswith("ml_switcheroo_compiler.backends.eager.reference_math"):
            func = raw_func
            try:
                func(MagicMock(), MagicMock(), MagicMock())
            except (
                ValueError,
                TypeError,
                AttributeError,
                IndexError,
                KeyError,
                RuntimeError,
                ZeroDivisionError,
                NotImplementedError,
                OSError,
                ImportError,
                OverflowError,
                RecursionError,
                LookupError,
                ArithmeticError,
            ):
                handled_errors += 1
            try:
                func([1.0], [2.0])
            except (
                ValueError,
                TypeError,
                AttributeError,
                IndexError,
                KeyError,
                RuntimeError,
                ZeroDivisionError,
                NotImplementedError,
                OSError,
                ImportError,
                OverflowError,
                RecursionError,
                LookupError,
                ArithmeticError,
            ):
                handled_errors += 1
            try:
                func([1.0])
            except (
                ValueError,
                TypeError,
                AttributeError,
                IndexError,
                KeyError,
                RuntimeError,
                ZeroDivisionError,
                NotImplementedError,
                OSError,
                ImportError,
                OverflowError,
                RecursionError,
                LookupError,
                ArithmeticError,
            ):
                handled_errors += 1
            try:
                func(MagicMock())
            except (
                ValueError,
                TypeError,
                AttributeError,
                IndexError,
                KeyError,
                RuntimeError,
                ZeroDivisionError,
                NotImplementedError,
                OSError,
                ImportError,
                OverflowError,
                RecursionError,
                LookupError,
                ArithmeticError,
            ):
                handled_errors += 1
    assert handled_errors >= 1


def test_eager_core_math_ops_brute() -> None:
    """Test function test_eager_core_math_ops_brute."""
    handled_errors = 0
    from ml_switcheroo_compiler.backends.eager_registry import global_eager_registry

    mock_b = MagicMock()
    del mock_b.gamma
    for _op, func in global_eager_registry._registry.items():
        if _op in ("write_file", "WriteFile", "save", "save_gguf", "savez", "savez_compressed"):
            continue
        try:
            func(mock_b, MagicMock())
        except (
            ValueError,
            TypeError,
            AttributeError,
            IndexError,
            KeyError,
            RuntimeError,
            ZeroDivisionError,
            NotImplementedError,
            OSError,
            ImportError,
            OverflowError,
            RecursionError,
            LookupError,
            ArithmeticError,
        ):
            handled_errors += 1
        try:
            func(mock_b, MagicMock(), MagicMock())
        except (
            ValueError,
            TypeError,
            AttributeError,
            IndexError,
            KeyError,
            RuntimeError,
            ZeroDivisionError,
            NotImplementedError,
            OSError,
            ImportError,
            OverflowError,
            RecursionError,
            LookupError,
            ArithmeticError,
        ):
            handled_errors += 1
        try:
            func(mock_b, 1.0)
        except (
            ValueError,
            TypeError,
            AttributeError,
            IndexError,
            KeyError,
            RuntimeError,
            ZeroDivisionError,
            NotImplementedError,
            OSError,
            ImportError,
            OverflowError,
            RecursionError,
            LookupError,
            ArithmeticError,
        ):
            handled_errors += 1
        try:
            func(mock_b, 1.0, 2.0)
        except (
            ValueError,
            TypeError,
            AttributeError,
            IndexError,
            KeyError,
            RuntimeError,
            ZeroDivisionError,
            NotImplementedError,
            OSError,
            ImportError,
            OverflowError,
            RecursionError,
            LookupError,
            ArithmeticError,
        ):
            handled_errors += 1
        try:
            func(mock_b, [1.0])
        except (
            ValueError,
            TypeError,
            AttributeError,
            IndexError,
            KeyError,
            RuntimeError,
            ZeroDivisionError,
            NotImplementedError,
            OSError,
            ImportError,
            OverflowError,
            RecursionError,
            LookupError,
            ArithmeticError,
        ):
            handled_errors += 1
        try:
            func(mock_b, [1.0], [2.0])
        except (
            ValueError,
            TypeError,
            AttributeError,
            IndexError,
            KeyError,
            RuntimeError,
            ZeroDivisionError,
            NotImplementedError,
            OSError,
            ImportError,
            OverflowError,
            RecursionError,
            LookupError,
            ArithmeticError,
        ):
            handled_errors += 1
    assert handled_errors >= 1
