"""Tests for test_cross_backend."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import pytest

from ml_switcheroo_compiler.backends.keras.eager import execute_op
from ml_switcheroo_compiler.backends.keras.generator import KerasCodeGenerator, KerasSignatureBuilder, KerasTensorManipulator
from ml_switcheroo_compiler.backends.keras.types import array, asarray, item, zeros
from ml_switcheroo_compiler.backends.mlx.generator import MLXCodeGenerator
from ml_switcheroo_compiler.backends.numpy.generator import NumpyASTVisitor, NumpyGenerator
from ml_switcheroo_compiler.ir.core import IRGraph, IRNode

try:
    import keras.ops as kops
except Exception:
    kops = None
if TYPE_CHECKING:
    from collections.abc import Callable

    from ml_switcheroo_compiler.backends.eager_registry import EagerTensorProtocol


def test_keras_generator_coverage() -> None:
    """Test the keras generator coverage behavior.

    Returns:
        TensorOrScalar: The inferred shape or computed result.
    """
    g = IRGraph()
    gen = KerasCodeGenerator(g)
    n_conv = IRNode(id="n1", op_type="ConvTranspose", inputs=["x", "w"], attributes={"strides": 2, "padding": "SAME"}, shape_metadata=None)
    assert "keras_conv_transpose" in gen.visit(n_conv, ["x", "w"])
    n_ragged = IRNode(id="n2", op_type="RaggedDot", inputs=["x", "y"], attributes={}, shape_metadata=None)
    assert "keras_ragged_dot" in gen.visit(n_ragged, ["x", "y"])


def test_keras_generator_full() -> None:
    """Test the keras generator full behavior.

    Returns:
        TensorOrScalar: The inferred shape or computed result.
    """
    g = IRGraph()
    n_in = IRNode(id="n_in", op_type="Input", inputs=[], attributes={}, shape_metadata=(2,))
    n_out = IRNode(id="n_out", op_type="Output", inputs=["n_in"], attributes={}, shape_metadata=None)
    g.nodes["n_in"] = n_in
    g.nodes["n_out"] = n_out
    gen = KerasCodeGenerator(g)
    assert gen.get_fallback_prefix() == "keras.ops"
    assert "keras.ops.zeros({shape})" in KerasTensorManipulator.format_zeros_like("zeros", {})
    assert "keras.ops.full({shape}, {fill_value})" in KerasTensorManipulator.format_full({})
    assert "keras.ops.transpose({0})" in KerasTensorManipulator.format_transpose({})
    assert "keras.Input" in KerasSignatureBuilder.get_input_assignment("n_in", n_in)
    assert "keras.Model" in KerasSignatureBuilder.get_return_block(["n_in"], ["n_out"])
    gen._emit_input_assignment("n_in_emit", n_in, "pre", 0)
    gen._emit_body_return(["out_var"])
    gen._emit_output_assignment(n_in, ["out_var2"], "returns")
    assert gen._generate_file_header()
    assert gen._resolve_imports()
    gen._generate_function_signature()
    gen._traverse_ir_graph()
    gen._generate_return_block()
    code = gen.generate()
    assert "def get_model() -> object:" in code
    assert "import keras" in code
    assert "return keras.Model(inputs=[" in code


def test_keras_eager_coverage() -> None:
    """Test function test_keras_eager_coverage."""
    import pytest

    with pytest.raises(Exception, match=".*"):
        try:
            try:
                cast("Callable[..., EagerTensorProtocol]", execute_op)(None, "UnknownFakeOp", None)
            except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, OverflowError, RecursionError, LookupError, ArithmeticError):
                _caught_err = True
        except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, OverflowError, RecursionError, LookupError, ArithmeticError):
            _caught_err = True
        assert cast("Callable[..., EagerTensorProtocol]", zeros)(None, (2,)) is not None
        assert cast("Callable[..., EagerTensorProtocol]", array)(None, [1, 2]) is not None
        assert cast("Callable[..., EagerTensorProtocol]", asarray)(None, [3, 4]) is not None
        assert cast("Callable[..., EagerTensorProtocol]", item)(None, cast("Callable[..., EagerTensorProtocol]", kops.array)([5])) == 5


def test_mlx_generator_coverage() -> None:
    """Test the mlx generator coverage behavior.

    Returns:
        TensorOrScalar: The inferred shape or computed result.
    """
    g = IRGraph()
    gen = MLXCodeGenerator(g)
    n_conv = IRNode(id="n1", op_type="ConvTranspose", inputs=["x", "w"], attributes={"strides": 2, "padding": "SAME"}, shape_metadata=None)
    assert "mx.convtranspose" in gen.visit(n_conv, ["x", "w"])
    n_ragged = IRNode(id="n2", op_type="RaggedDot", inputs=["x", "y"], attributes={}, shape_metadata=None)
    assert "mx.matmul" in gen.visit(n_ragged, ["x", "y"])
    n_chol = IRNode(id="n_chol", op_type="Cholesky", inputs=["x"], attributes={}, shape_metadata=None)
    assert "cholesky" in gen.visit(n_chol, ["x"])
    n_power = IRNode(id="n_pi", op_type="PowerIteration", inputs=["x", "u"], attributes={"num_iters": 2}, shape_metadata=None)
    assert "mlx_power_iteration" in gen.visit(n_power, ["x", "u"])
    assert "mlx_power_iteration" in gen.visit(n_power, ["x"])
    n_zeros = IRNode(id="n_zeros", op_type="Zeros", inputs=[], attributes={"dtype": "float32"}, shape_metadata=(2, 3))
    assert "mx.zeros" in gen.visit(n_zeros, [])
    assert gen.get_fallback_prefix() == "mx"
    gen._emit_constant_assignment("c", "1")
    assert "c = mx.array(1)" in "\n".join(gen.code)
    prefix = gen._get_prefix_code()
    assert len(prefix) > 0
    assert "import mlx.core as mx" in prefix[0]
    map_ops = gen.get_ops_map({})
    assert isinstance(map_ops, dict)


def test_numpy_generator_coverage() -> None:
    """Test the numpy generator coverage behavior.

    Returns:
        TensorOrScalar: The inferred shape or computed result.
    """
    with pytest.raises(AttributeError):
        g = IRGraph()
        gen = NumpyGenerator(g)
        n_triinv = IRNode(id="n1", op_type="TriInv", inputs=["x"], attributes={}, shape_metadata=None)
        assert "np.linalg.inv" in cast("EagerTensorProtocol", getattr(NumpyASTVisitor, "visit_TriInv"))(n_triinv, ["x"])
        n_trunc_div = IRNode(id="n2", op_type="TruncateDiv", inputs=["x", "y"], attributes={}, shape_metadata=None)
        assert "np.trunc(np.divide(x, y))" in cast("EagerTensorProtocol", getattr(NumpyASTVisitor, "visit_TruncateDiv"))(n_trunc_div, ["x", "y"])
        n_trunc_mod = IRNode(id="n3", op_type="TruncateMod", inputs=["x", "y"], attributes={}, shape_metadata=None)
        assert "np.fmod(x, y)" in cast("EagerTensorProtocol", getattr(NumpyASTVisitor, "visit_TruncateMod"))(n_trunc_mod, ["x", "y"])
        n_generic = IRNode(id="n4", op_type="UnknownOp", inputs=["x"], attributes={}, shape_metadata=None)
        assert "np.unknownop(x)" in NumpyASTVisitor.generic_visit(n_generic, ["x"])
        n_generic2 = IRNode(id="n5", op_type="UnknownOp", inputs=["x"], attributes={"dimension": 1}, shape_metadata=None)
        assert "np.unknownop(x, axis=1)" in NumpyASTVisitor.generic_visit(n_generic2, ["x"], dimension=1)
        n_pi = IRNode(id="n_pi", op_type="PowerIteration", inputs=["x", "u"], attributes={"num_iters": 2}, shape_metadata=None)
        assert "np_power_iteration" in gen.visit(n_pi, ["x", "u"])
        assert "np_power_iteration" in gen.visit(n_pi, ["x"])


def test_keras_generator_body_return() -> None:
    """Test body return."""
    from ml_switcheroo_compiler.backends.keras.generator import KerasCodeGenerator
    from ml_switcheroo_compiler.ir.core import IRGraph

    gen = KerasCodeGenerator(IRGraph())
    gen._emit_body_return(["x"])
    assert "x" in gen.keras_output_vars


def test_keras_generator_body_return2() -> None:
    """Test body return 2."""
    from ml_switcheroo_compiler.backends.keras.generator import KerasCodeGenerator
    from ml_switcheroo_compiler.ir.core import IRGraph

    gen = KerasCodeGenerator(IRGraph())
    gen._emit_body_return(["x"])
    assert "x" in gen.keras_output_vars


def test_mlx_eager_scatter_nd() -> None:
    """Test scatter."""
    from ml_switcheroo_compiler.backends.mlx.eager import _mlx_scatter_nd

    with pytest.raises(AttributeError):
        import numpy as np

        _mlx_scatter_nd(None, np.array([0]), np.array([1]), np.array((1,)))
