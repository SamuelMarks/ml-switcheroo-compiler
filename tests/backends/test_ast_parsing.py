# ruff: noqa: E501
"""Test AST parsing of generated code."""

import ast

import pytest

from ml_switcheroo_compiler.backends.ast_to_ir import parse_ast_to_ir
from ml_switcheroo_compiler.backends.jax.generator import JAXCodeGenerator
from ml_switcheroo_compiler.backends.keras.generator import KerasCodeGenerator
from ml_switcheroo_compiler.backends.mlx.generator import MLXCodeGenerator
from ml_switcheroo_compiler.backends.numpy.generator import NumpyGenerator
from ml_switcheroo_compiler.backends.pytorch.generator import PyTorchCodeGenerator
from ml_switcheroo_compiler.backends.tensorflow.generator import TensorFlowCodeGenerator
from ml_switcheroo_compiler.ir.core import IRGraph


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
    """Verify that generated code from an empty graph parses as valid Python AST.

    Args:
        generator_cls (type): Backend code generator class under test.
    """
    graph = IRGraph()
    gen = generator_cls(graph)
    code = gen.generate()
    tree = ast.parse(code)
    assert isinstance(tree, ast.Module)
    assert len(tree.body) > 0


def test_ast_to_ir_visitor_nodes_and_attributes() -> None:
    """Verify exact IRGraph nodes, inputs, and attributes produced by ASTToIRVisitor."""
    src = """
def forward(x, y):
    z = x + y
    return z
"""
    graph = parse_ast_to_ir(src)
    assert len(graph.outputs) == 1
    out_node = graph.nodes[graph.outputs[0]]
    assert out_node.op_type == "Add"
    assert out_node.inputs == ["x", "y"]
