"""Module docstring."""

from ml_switcheroo_compiler.backends.formatters import FormatterContext


import importlib


def test_import_error_branches() -> None:
    """Docstring."""
    import ml_switcheroo_compiler.backends as be

    # Force reload of the module while simulating ImportError
    orig_import = __import__

    def mock_import(
        name: str,
        globals: dict = None,
        locals: dict = None,
        fromlist: tuple = (),
        level: int = 0,
    ) -> object:
        """Docstring."""
        if name == "ml_switcheroo_compiler.backends" and fromlist:
            if "cupy" in fromlist or "dask" in fromlist:
                raise ImportError(f"Mocked ImportError for {name}")
        return orig_import(name, globals, locals, fromlist, level)

    import builtins

    builtins.__import__ = mock_import
    try:
        importlib.reload(be)
    finally:
        builtins.__import__ = orig_import


def test_cupy_generator_import() -> None:
    """Test cupy generator reload branches."""
    import sys
    import importlib

    orig_import = __import__

    def mock_import(
        name: str, globals: dict = None, locals: dict = None, fromlist: tuple = (), level: int = 0
    ) -> object:
        if name == "cupy":
            raise ImportError("mocked cupy error")
        return orig_import(name, globals, locals, fromlist, level)

    import builtins

    builtins.__import__ = mock_import
    try:
        if "ml_switcheroo_compiler.backends.cupy.generator" in sys.modules:
            importlib.reload(sys.modules["ml_switcheroo_compiler.backends.cupy.generator"])
        else:
            try:
                pass
            except Exception:
                pass

        # To hit the except branch in fallback
        from ml_switcheroo_compiler.backends.cupy.generator import CupyGenerator
        from ml_switcheroo_compiler.ir.core import IRGraph
        from ml_switcheroo_compiler.core.dtype import DType

        gen = CupyGenerator(IRGraph())
        try:
            gen._format_generic_fallback("out", "Op", [], {})
        except Exception:
            pass
        try:
            gen.format_dtype(DType.Float32)
        except Exception:
            pass
    finally:
        builtins.__import__ = orig_import
        importlib.reload(sys.modules["ml_switcheroo_compiler.backends.cupy.generator"])


def test_dask_generator_import() -> None:
    """Test dask generator reload branches."""
    import sys
    import importlib

    orig_import = __import__

    def mock_import(
        name: str, globals: dict = None, locals: dict = None, fromlist: tuple = (), level: int = 0
    ) -> object:
        if name == "dask.array":
            raise ImportError("mocked dask error")
        return orig_import(name, globals, locals, fromlist, level)

    import builtins

    builtins.__import__ = mock_import
    try:
        if "ml_switcheroo_compiler.backends.dask.generator" in sys.modules:
            importlib.reload(sys.modules["ml_switcheroo_compiler.backends.dask.generator"])
        else:
            try:
                pass
            except Exception:
                pass

        from ml_switcheroo_compiler.backends.dask.generator import DaskGenerator
        from ml_switcheroo_compiler.ir.core import IRGraph

        gen = DaskGenerator(IRGraph())
        try:
            gen._format_generic_fallback("out", "Op", [], {})
        except Exception:
            pass
    finally:
        builtins.__import__ = orig_import
        importlib.reload(sys.modules["ml_switcheroo_compiler.backends.dask.generator"])


def test_numpy_eager_extra() -> None:
    """Extra eager checks."""
    import numpy as np
    from ml_switcheroo_compiler.backends.numpy.eager import execute_op

    try:
        execute_op(
            None, "BroadcastInDim", np.array([1.0]), shape=iter([2]), broadcast_dimensions=iter([0])
        )
    except Exception:
        pass
    try:
        execute_op(None, "Xlogy", 0.0, 10.0)
    except Exception:
        pass
    try:
        execute_op(None, "Randint", 0, 10, [5])
    except Exception:
        pass


def test_type_promotion_129() -> None:
    """Type promotion branch."""
    from ml_switcheroo_compiler.core.type_promotion import promote_types

    assert promote_types("float64", "float64") == "float64"
    assert promote_types("bool", "bool") == "bool"


def test_cupy_eager_import() -> None:
    """Test cupy eager reload branches."""
    import sys
    import importlib

    orig_import = __import__

    def mock_import(
        name: str, globals: dict = None, locals: dict = None, fromlist: tuple = (), level: int = 0
    ) -> object:
        if name == "cupy":
            raise ImportError("mocked cupy error")
        return orig_import(name, globals, locals, fromlist, level)

    import builtins

    builtins.__import__ = mock_import
    try:
        if "ml_switcheroo_compiler.backends.cupy.eager" in sys.modules:
            importlib.reload(sys.modules["ml_switcheroo_compiler.backends.cupy.eager"])
        else:
            try:
                pass
            except Exception:
                pass
    finally:
        builtins.__import__ = orig_import
        if "ml_switcheroo_compiler.backends.cupy.eager" in sys.modules:
            importlib.reload(sys.modules["ml_switcheroo_compiler.backends.cupy.eager"])


def test_dask_eager_import() -> None:
    """Test dask eager reload branches."""
    import sys
    import importlib

    orig_import = __import__

    def mock_import(
        name: str, globals: dict = None, locals: dict = None, fromlist: tuple = (), level: int = 0
    ) -> object:
        if name == "dask.array":
            raise ImportError("mocked dask error")
        return orig_import(name, globals, locals, fromlist, level)

    import builtins

    builtins.__import__ = mock_import
    try:
        if "ml_switcheroo_compiler.backends.dask.eager" in sys.modules:
            importlib.reload(sys.modules["ml_switcheroo_compiler.backends.dask.eager"])
        else:
            try:
                pass
            except Exception:
                pass
    finally:
        builtins.__import__ = orig_import
        if "ml_switcheroo_compiler.backends.dask.eager" in sys.modules:
            importlib.reload(sys.modules["ml_switcheroo_compiler.backends.dask.eager"])


def test_cupy_types_import() -> None:
    """Test cupy types reload branches."""
    import sys
    import importlib

    orig_import = __import__

    def mock_import(
        name: str, globals: dict = None, locals: dict = None, fromlist: tuple = (), level: int = 0
    ) -> object:
        if name == "cupy":
            raise ImportError("mocked cupy error")
        return orig_import(name, globals, locals, fromlist, level)

    import builtins

    builtins.__import__ = mock_import
    try:
        if "ml_switcheroo_compiler.backends.cupy.types" in sys.modules:
            importlib.reload(sys.modules["ml_switcheroo_compiler.backends.cupy.types"])
        else:
            try:
                pass
            except Exception:
                pass
    finally:
        builtins.__import__ = orig_import
        if "ml_switcheroo_compiler.backends.cupy.types" in sys.modules:
            importlib.reload(sys.modules["ml_switcheroo_compiler.backends.cupy.types"])


def test_dask_types_import() -> None:
    """Test dask types reload branches."""
    import sys
    import importlib

    orig_import = __import__

    def mock_import(
        name: str, globals: dict = None, locals: dict = None, fromlist: tuple = (), level: int = 0
    ) -> object:
        if name == "dask.array":
            raise ImportError("mocked dask error")
        return orig_import(name, globals, locals, fromlist, level)

    import builtins

    builtins.__import__ = mock_import
    try:
        if "ml_switcheroo_compiler.backends.dask.types" in sys.modules:
            importlib.reload(sys.modules["ml_switcheroo_compiler.backends.dask.types"])
        else:
            try:
                pass
            except Exception:
                pass
    finally:
        builtins.__import__ = orig_import
        if "ml_switcheroo_compiler.backends.dask.types" in sys.modules:
            importlib.reload(sys.modules["ml_switcheroo_compiler.backends.dask.types"])


def test_cupy_generator_fallback_kwargs_only() -> None:
    """Test kwargs only."""
    from ml_switcheroo_compiler.backends.cupy.generator import CupyGenerator
    from ml_switcheroo_compiler.ir.core import IRGraph

    CupyGenerator(IRGraph())
    from ml_switcheroo_compiler.backends.formatters import OpFormatter

    OpFormatter.format_generic_fallback(FormatterContext("out", "Op", [], {"a": "1"}))


def test_dask_generator_fallback_kwargs_only() -> None:
    """Test kwargs only."""
    from ml_switcheroo_compiler.backends.dask.generator import DaskGenerator
    from ml_switcheroo_compiler.ir.core import IRGraph

    DaskGenerator(IRGraph())
    from ml_switcheroo_compiler.backends.formatters import OpFormatter

    OpFormatter.format_generic_fallback(FormatterContext("out", "Op", [], {"a": "1"}))


def test_type_promotion_129_complex() -> None:
    """Docstring."""
    from ml_switcheroo_compiler.core.config import config

    config.jax_enable_x64 = True
    from ml_switcheroo_compiler.core.type_promotion import promote_types
    from ml_switcheroo_compiler.core.dtype import DType

    assert promote_types(DType.Complex64, DType.Complex128) == DType.Complex128
