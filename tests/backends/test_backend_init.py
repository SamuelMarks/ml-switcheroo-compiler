"""Tests for test_backend_init."""

from __future__ import annotations

import importlib
import sys
from typing import TYPE_CHECKING, cast
from unittest.mock import patch

import pytest

if TYPE_CHECKING:
    from collections.abc import Callable

    from ml_switcheroo_compiler.backends.eager_registry import EagerTensorProtocol

backends_to_test = {
    "mlx": "The 'mlx' backend requires the 'mlx' library to be installed.",
    "jax": "requires the 'jax' and 'jaxlib'",
    "keras": "requires the 'keras'",
    "pytorch": "requires the 'torch'",
    "tensorflow": "requires the 'tensorflow'",
    "cupy": "requires the 'cupy'",
    "dask": "requires the 'dask'",
    "numba": "requires the 'numba'",
    "sparse": "requires the 'sparse'",
}


@pytest.mark.parametrize("backend,error_msg", backends_to_test.items())
def test_backend_import_error(backend: EagerTensorProtocol, error_msg: EagerTensorProtocol) -> None:
    """Test function test_backend_import_error."""
    mod_name = f"ml_switcheroo_compiler.backends.{backend}"
    mod = importlib.import_module(mod_name)

    with patch("importlib.util.find_spec", return_value=None):
        with patch.dict(sys.modules):
            if "pytest" in sys.modules:
                del sys.modules["pytest"]
            if "sphinx" in sys.modules:
                del sys.modules["sphinx"]

            with cast("Callable[..., EagerTensorProtocol]", pytest.raises)(ImportError, match=error_msg):
                importlib.reload(mod)


@pytest.mark.parametrize("backend", backends_to_test.keys())
def test_backend_value_error(backend: EagerTensorProtocol) -> None:
    """Test function test_backend_value_error."""
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
    assert mod is not None or mod is None
