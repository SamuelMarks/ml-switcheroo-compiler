"""Test JAX backend edge cases coverage."""

import os
import sys
from unittest.mock import MagicMock, patch

from ml_switcheroo_ir import LogicalGraph

from ml_switcheroo_compiler.backends.jax.generator import JAXCodeGenerator


def test_jaxget_fallback_prefix():
    gen = JAXCodeGenerator(LogicalGraph())
    assert gen.get_fallback_prefix() == "jnp"


def test_jax_format_zeros_like():
    gen = JAXCodeGenerator(LogicalGraph())
    assert gen._format_zeros_like("zeros", {}) == "jnp.zeros({shape})"
    assert gen._format_zeros_like("zeros", {"dtype": "float32"}) == "jnp.zeros({shape}), dtype='float32'"


def test_jax_format_full():
    gen = JAXCodeGenerator(LogicalGraph())
    assert gen._format_full({}) == "jnp.full({shape}, {fill_value})"
    assert gen._format_full({"dtype": "float32"}) == "jnp.full({shape}, {fill_value}), dtype='float32'"


def test_jax_get_fallback_prefix():
    gen = JAXCodeGenerator(LogicalGraph())
    assert gen.get_fallback_prefix() == "jnp"


def test_jax_get_ops_map():
    gen = JAXCodeGenerator(LogicalGraph())
    ops = gen.get_ops_map({"dtype": "float32"})
    assert "Zeros" in ops
    assert "Ones" in ops
    assert "Full" in ops


def test_jax_emit_constant_assignment():
    gen = JAXCodeGenerator(LogicalGraph())
    gen._emit_constant_assignment("var", "val")
    assert len(gen.code) == 1
    assert gen.code[0] == "var = jnp.array(val)"


def test_jax_generate_file_header():
    gen = JAXCodeGenerator(LogicalGraph())
    header = gen._generate_file_header()
    assert isinstance(header, list)


def test_jax_resolve_imports():
    gen = JAXCodeGenerator(LogicalGraph())
    # Should resolve templates
    imports = gen._resolve_imports()
    assert "import jax" in imports
    assert "import jax.numpy as jnp" in imports


def test_jax_generate_function_signature():
    gen = JAXCodeGenerator(LogicalGraph())
    gen._generate_function_signature()
    assert len(gen.code) == 1
    assert gen.code[0] == "def apply_model(params, *args, **kwargs) -> object:"
    assert gen.indent_level == 1


def test_jax_save_load(tmp_path):
    mock_jax = MagicMock()
    mock_jnp = MagicMock()
    mock_jax.numpy = mock_jnp

    with patch.dict(sys.modules, {"jax": mock_jax, "jax.numpy": mock_jnp}):
        filepath = os.path.join(tmp_path, "test.npy")
        arr = [1, 2, 3]

        JAXCodeGenerator.save(filepath, arr)
        mock_jnp.save.assert_called_once_with(filepath, arr, allow_pickle=True, fix_imports=True)

        JAXCodeGenerator.load(filepath)
        mock_jnp.load.assert_called_once_with(filepath, allow_pickle=False, fix_imports=True, encoding="ASCII")


def test_jax_savez(tmp_path):
    mock_jax = MagicMock()
    mock_jnp = MagicMock()
    mock_jax.numpy = mock_jnp

    with patch.dict(sys.modules, {"jax": mock_jax, "jax.numpy": mock_jnp}):
        filepath = os.path.join(tmp_path, "test.npz")
        arr = [1, 2, 3]

        JAXCodeGenerator.savez(filepath, a=arr)
        mock_jnp.savez.assert_called_once_with(filepath, a=arr)


def test_jax_savez_compressed(tmp_path):
    mock_jax = MagicMock()
    mock_jnp = MagicMock()
    mock_jax.numpy = mock_jnp

    with patch.dict(sys.modules, {"jax": mock_jax, "jax.numpy": mock_jnp}):
        filepath = os.path.join(tmp_path, "test_comp.npz")
        arr = [1, 2, 3]

        JAXCodeGenerator.savez_compressed(filepath, a=arr)
        mock_jnp.savez_compressed.assert_called_once_with(filepath, a=arr)
