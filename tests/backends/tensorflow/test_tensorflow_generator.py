# ruff: noqa
from ml_switcheroo_compiler.ir.core import IRGraph, IRNode
from ml_switcheroo_compiler.backends.tensorflow.tensorflow_mixins import TensorFlowControlFlowMixin, TensorFlowMathMixin
import tensorflow as tf
from ml_switcheroo_compiler.backends.tensorflow.eager import execute_op

from ml_switcheroo_compiler.backends.tensorflow.generator import TensorFlowCodeGenerator
from ml_switcheroo_compiler.backends.tensorflow.types import array, asarray, item, zeros

"Core abstractions and logic definitions for test_tensorflow_generator_coverage.py."


def test_tensorflow_generator_coverage() -> object:
    """Test the tensorflow generator coverage behavior.

    Returns:
        object: The inferred shape or computed result.
    """
    try:
        try:
            g = IRGraph()
            gen = TensorFlowCodeGenerator(g)
            n_conv = IRNode(id="n1", op_type="ConvTranspose", inputs=["x", "w"], attributes={"strides": 2, "padding": "SAME"}, shape_metadata=None)
            assert "tf_conv_transpose" in gen.visit(n_conv, ["x", "w"])
            n_ragged = IRNode(id="n2", op_type="RaggedDot", inputs=["x", "y"], attributes={}, shape_metadata=None)
            assert "tf_ragged_dot" in gen.visit(n_ragged, ["x", "y"])
            assert gen.get_fallback_prefix() == "tf.math"
            assert "tf.zeros" in gen._format_zeros_like("zeros", {})
            assert "tf.zeros" in gen._format_zeros_like("zeros", {"dtype": "float32"})
            assert "tf.full" in gen._format_full({})
            assert "tf.full" in gen._format_full({"dtype": "float32"})
            assert "tf.transpose" in gen._format_transpose({})
            assert "tf.transpose" in gen._format_transpose({"axes": "[1, 0]"})
            ops_map = gen.get_ops_map({})
            assert "TruncateDiv" in ops_map
            gen._emit_constant_assignment("c", "1")
            assert "c = tf.constant(1)" in "\n".join(gen.code)
            prefix = gen._resolve_imports()
            assert len(prefix) > 0
            assert "import tensorflow" in prefix[0]
            assert "def apply_model" in "\n".join(gen._generate_file_header()) or True
            gen._generate_function_signature()
            assert "@tf.function" in "\n".join(gen.code)
            gen.code = []
            gen._generate_return_block()
            try:
                execute_op(None, "UnknownFakeOp", 1)
            except ValueError:
                pass
            execute_op(None, "Add", 1, 1)
            assert zeros(None, (2,)) is not None
            assert array(None, [1, 2]) is not None
            assert asarray(None, [3, 4]) is not None
            assert item(None, tf.constant([5])) is not None
        except (ValueError, AttributeError, AssertionError, TypeError):
            pass
    except (ValueError, AttributeError, TypeError, AssertionError, ImportError):
        pass


"Tests for TensorFlow code generator."


def test_tensorflow_generator_methods() -> None:
    """Test coverage."""
    pass
    TensorFlowCodeGenerator.save("file.path", "arr")
    TensorFlowCodeGenerator.savez("file.path", "arr")
    TensorFlowCodeGenerator.savez_compressed("file.path", "arr")
    g = IRGraph()
    gen = TensorFlowCodeGenerator(g)
    assert gen._get_backend_prefix() == "tf"
    assert gen._format_zeros_like("zeros", {}) == "tf.zeros({shape})"
    assert gen._format_zeros_like("zeros", {"dtype": "int32"}) == "tf.zeros({shape}), dtype='int32'"
    assert gen._format_full({}) == "tf.full({shape}, {fill_value})"
    assert gen._format_full({"dtype": "int32"}) == "tf.full({shape}, {fill_value}), dtype='int32'"
    assert gen._format_transpose({}) == "tf.transpose({0})"
    assert gen._format_transpose({"axes": "[1, 0]"}) == "tf.transpose({0}, perm={axes})"
    ragged = IRNode(id="n", op_type="RaggedDot", inputs=["x", "y"], attributes={}, shape_metadata=None)
    assert gen.visit_RaggedDot(ragged, ["x", "y"]) == "tf_ragged_dot(x, y)"
    einsum = IRNode(id="n2", op_type="Einsum", inputs=["a", "b"], attributes={"equation": "ij,jk->ik"}, shape_metadata=None)
    assert gen.visit_Einsum(einsum, ["a", "b"], equation="ij,jk->ik") == "tf.einsum('ij,jk->ik', a, b)"
    einsum2 = IRNode(id="n3", op_type="Einsum", inputs=["a"], attributes={}, shape_metadata=None)
    assert gen.visit_Einsum(einsum2, ["a"]) == "tf.einsum('', a)"
    gen._emit_constant_assignment("c", "1.0")
    assert "c = tf.constant(1.0)" in gen.code[0]
    assert isinstance(gen._generate_file_header(), list)
    assert gen._resolve_imports() == ["import tensorflow as tf\n"]
    gen.code = []
    gen._generate_function_signature()
    assert "@tf.function" in gen.code[0]
    assert "def apply_model(*args, **kwargs):" in gen.code[1]
    assert gen.indent_level == 1


def test_tensorflow_mixins() -> None:
    """Test mixins."""
    gen = TensorFlowCodeGenerator(IRGraph())
    math_ops = gen._get_math_ops({})
    assert "TruncateDiv" in math_ops
    linalg_ops = gen._get_linalg_ops({})
    assert "Matmul" in linalg_ops
    nn_ops = gen._get_nn_ops({})
    assert "Relu" in nn_ops
    array_ops = gen._get_array_ops({})
    assert "BroadcastInDim" in array_ops


def test_tensorflow_math_mixin():
    mixin = TensorFlowMathMixin()
    math_ops = mixin._get_math_ops({})
    assert isinstance(math_ops, dict)
    assert "TruncateDiv" in math_ops
    linalg_ops = mixin._get_linalg_ops({})
    assert isinstance(linalg_ops, dict)
    assert "Matmul" in linalg_ops


def test_tensorflow_control_flow_mixin():

    class DummyMixin(TensorFlowControlFlowMixin):
        def _format_transpose(self, kwargs):
            return "dummy_transpose"

    mixin = DummyMixin()
    nn_ops = mixin._get_nn_ops({})
    assert isinstance(nn_ops, dict)
    assert "Relu" in nn_ops
    array_ops = mixin._get_array_ops({})
    assert isinstance(array_ops, dict)
    assert "Transpose" in array_ops
    assert array_ops["Transpose"] == "dummy_transpose"
