import importlib
import sys
from unittest.mock import patch

import pytest

backends_to_test = {
    "jax": "requires the 'jax' and 'jaxlib'",
    "keras": "requires the 'keras'",
    "pytorch": "requires the 'torch'",
    "tensorflow": "requires the 'tensorflow'",
    "cupy": "requires the 'cupy'",
    "dask": "requires the 'dask'",
}


@pytest.mark.parametrize("backend,error_msg", backends_to_test.items())
def test_backend_import_error(backend, error_msg):
    mod_name = f"ml_switcheroo_compiler.backends.{backend}"
    mod = importlib.import_module(mod_name)

    with patch("importlib.util.find_spec", return_value=None):
        with patch.dict(sys.modules):
            if "pytest" in sys.modules:
                del sys.modules["pytest"]
            if "sphinx" in sys.modules:
                del sys.modules["sphinx"]

            with pytest.raises(ImportError, match=error_msg):
                importlib.reload(mod)


@pytest.mark.parametrize("backend", backends_to_test.keys())
def test_backend_value_error(backend):
    mod_name = f"ml_switcheroo_compiler.backends.{backend}"
    mod = importlib.import_module(mod_name)

    with patch("importlib.util.find_spec", side_effect=ValueError("mock error")):
        with patch.dict(sys.modules):
            if "pytest" in sys.modules:
                del sys.modules["pytest"]
            if "sphinx" in sys.modules:
                del sys.modules["sphinx"]

            # Should not raise ImportError because ValueError means it's loaded
            importlib.reload(mod)
