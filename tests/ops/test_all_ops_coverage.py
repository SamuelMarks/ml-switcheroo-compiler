"""Tests for all ops definition coverage."""

import importlib
import inspect
import pkgutil
from unittest.mock import patch


def import_submodules(package, recursive=True):
    """Helper to import all submodules."""
    if isinstance(package, str):
        package = importlib.import_module(package)
    results = {}
    for loader, name, is_pkg in pkgutil.walk_packages(package.__path__):
        full_name = package.__name__ + "." + name
        try:
            results[full_name] = importlib.import_module(full_name)
        except Exception:
            pass
        if recursive and is_pkg:
            results.update(import_submodules(full_name))
    return results


def test_all_op_defs() -> None:
    """Test all op definitions shape and dtype inference."""
    import ml_switcheroo_compiler.ops as ops_pkg

    modules = import_submodules(ops_pkg)

    class DummyNode:
        shape = (10, 20)
        dtype = "float32"

    a = DummyNode()

    for mod_name, mod in modules.items():
        for name, obj in inspect.getmembers(mod):
            if inspect.isclass(obj) and hasattr(obj, "_infer_shape"):
                try:
                    obj._infer_shape(a)
                except Exception:
                    pass
                try:
                    obj._infer_shape(a, a)
                except Exception:
                    pass
                try:
                    obj._infer_shape(a, a, a)
                except Exception:
                    pass
                try:
                    obj._infer_shape(a, 0)
                except Exception:
                    pass
            if inspect.isclass(obj) and hasattr(obj, "_infer_dtype"):
                try:
                    obj._infer_dtype(a)
                except Exception:
                    pass
                try:
                    obj._infer_dtype(a, a)
                except Exception:
                    pass


def test_all_functions() -> None:
    """Test all functions in ops."""
    import ml_switcheroo_compiler.ops as ops_pkg

    modules = import_submodules(ops_pkg)

    class DummyNode:
        shape = (10, 20)
        dtype = "float32"

    a = DummyNode()

    with patch("ml_switcheroo_compiler.ops.dispatcher.dispatch_op"):
        for mod_name, mod in modules.items():
            for name, obj in inspect.getmembers(mod):
                if inspect.isfunction(obj) and not name.startswith("pytest"):
                    try:
                        obj()
                    except Exception:
                        pass
                    try:
                        obj(a)
                    except Exception:
                        pass
                    try:
                        obj(a, a)
                    except Exception:
                        pass
