"""Test AST parsing of generated code."""

import pytest
import ast
from ml_switcheroo_compiler.ir.core import IRGraph
from ml_switcheroo_compiler.backends.pytorch.generator import PyTorchCodeGenerator
from ml_switcheroo_compiler.backends.tensorflow.generator import TensorFlowCodeGenerator
from ml_switcheroo_compiler.backends.jax.generator import JAXCodeGenerator
from ml_switcheroo_compiler.backends.numpy.generator import NumpyGenerator
from ml_switcheroo_compiler.backends.keras.generator import KerasCodeGenerator
from ml_switcheroo_compiler.backends.mlx.generator import MLXCodeGenerator


@pytest.mark.parametrize(
    "generator_cls",
    [
        PyTorchCodeGenerator,
        TensorFlowCodeGenerator,
        JAXCodeGenerator,
        NumpyGenerator,
        KerasCodeGenerator,
        MLXCodeGenerator,
    ],
)
def test_generated_code_ast_parses(generator_cls: type) -> None:
    """Test that generated code from an empty graph parses successfully."""
    graph = IRGraph()
    gen = generator_cls(graph)
    code = gen.generate()

    try:
        ast.parse(code)
    except SyntaxError as e:
        pytest.fail(f"Generated code produced a SyntaxError: {e}\n\nCode:\n{code}")
