"""Tests for test_tensorflow_generator."""

from __future__ import annotations

import typing

from ml_switcheroo_ir import NodeDict

from ml_switcheroo_compiler.backends.tensorflow.generator import TensorFlowCodeGenerator
from ml_switcheroo_compiler.ir.core import IRGraph
from tests.testing_utils import as_callable, as_tensor

if typing.TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import EagerTensorProtocol, EagerValue


class DummyNode:
    """Test class DummyNode."""

    def __init__(self, attrs: EagerValue = None, op_type: EagerValue = "Unknown") -> None:
        """Test function __init__.

        Args:
            attrs: Test parameter `attrs`.
            op_type: Test parameter `op_type`.
        """
        self.attributes = attrs or {}
        self.op_type = op_type
        self.id = "n1"


def test_tf_generator_basics(tmp_path: EagerTensorProtocol) -> None:
    """Test function test_tf_generator_basics.

    Args:
        tmp_path: Test parameter `tmp_path`.
    """
    g = IRGraph()
    gen = TensorFlowCodeGenerator(g)

    import numpy as np

    arr = np.array([1, 2, 3])
    file_npy = tmp_path / "test.npy"
    TensorFlowCodeGenerator.save(str(file_npy), arr)
    res_npy = TensorFlowCodeGenerator.load(str(file_npy))
    as_callable(np.testing.assert_array_equal)(res_npy, arr)

    file_npz = tmp_path / "test.npz"
    TensorFlowCodeGenerator.savez(str(file_npz), a=arr)
    res_npz = TensorFlowCodeGenerator.load(str(file_npz))
    as_callable(np.testing.assert_array_equal)(as_tensor(getattr(res_npz, "__getitem__")("a")), arr)

    file_comp = tmp_path / "test_comp.npz"
    TensorFlowCodeGenerator.savez_compressed(str(file_comp), a=arr)
    import gzip
    import pickle

    with gzip.open(str(file_comp), "rb") as f:
        res_comp = pickle.load(f)
    np.testing.assert_array_equal(res_comp["a"], arr)

    assert gen._format_zeros_like("zeros", {}) == "tf.zeros({shape})"
    assert gen._format_zeros_like("zeros", {"dtype": "float32"}) == "tf.zeros({shape}), dtype='float32'"

    assert gen._format_full({}) == "tf.full({shape}, {fill_value})"
    assert gen._format_full({"dtype": "float32"}) == "tf.full({shape}, {fill_value}), dtype='float32'"

    assert gen._format_transpose({}) == "tf.transpose({0})"
    assert gen._format_transpose({"axes": [0, 1]}) == "tf.transpose({0}, perm={axes})"

    assert as_callable(gen.visit_RaggedDot)(as_callable(DummyNode)(), ["x", "y"]) == "tf_ragged_dot(x, y)"
    assert as_callable(gen.visit_Einsum)(as_callable(DummyNode)(), ["x", "y"], equation="i,j->ij") == "tf.einsum('i,j->ij', x, y)"

    assert gen.get_fallback_prefix() == "tf.math"
    assert "Zeros" in gen._get_creation_ops({})

    gen._emit_constant_assignment("c", "5")
    assert "c = tf.constant(5)" in gen.code[-1]

    assert gen._generate_file_header() == [gen.header.strip()]
    assert gen._resolve_imports() == ["import tensorflow as tf\n"]

    gen.code = []
    gen._generate_function_signature()
    assert "def apply_model(*args, **kwargs) -> object:" in gen.code[-1]


def test_tf_generator_pure_tf_and_aot() -> None:
    """Verify generate_pure_tf and compile_aot execution with tf.function."""
    import importlib.util
    import sys
    from unittest.mock import MagicMock, patch

    import numpy as np

    from ml_switcheroo_compiler.backends.tensorflow.generator import TensorFlowCodeGenerator
    from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
    from ml_switcheroo_compiler.ir.core import IRGraph, IRNode

    g = IRGraph()
    n_in = IRNode(id="x", op_type="Input", shape_metadata=[2, 2], attributes={"dtype": "float32"})
    n_relu = IRNode(id="y", op_type="Relu", inputs=["x"], shape_metadata=[2, 2], attributes={"dtype": "float32"})
    g.nodes = NodeDict({"x": n_in, "y": n_relu})
    g.inputs = ["x"]
    g.outputs = ["y"]

    gen = TensorFlowCodeGenerator(g)
    code = gen.generate()
    assert "def apply_model" in code
    assert "import tensorflow as tf" in code

    g_aot = IRGraph()
    n_in_aot = IRNode(id="x", op_type="Input", shape_metadata=[2, 2], attributes={"dtype": "float32"})
    n_relu_aot = IRNode(id="y", op_type="Relu", inputs=["x"], shape_metadata=[2, 2], attributes={"dtype": "float32"})
    g_aot.nodes = NodeDict({"x": n_in_aot, "y": n_relu_aot})
    g_aot.inputs = ["x"]
    g_aot.outputs = ["y"]

    mock_tf = MagicMock()
    mock_tf.nn.relu.side_effect = lambda x: np.maximum(x, 0.0)
    mock_tf.convert_to_tensor.side_effect = lambda x: np.asarray(x)
    ctx = patch.dict(sys.modules, {"tensorflow": mock_tf}) if importlib.util.find_spec("tensorflow") is None else MagicMock()

    with ctx:
        with patch("tensorflow.function", side_effect=lambda fn, **kw: fn):
            runner = gen.compile_aot(g_aot, jit_compile=True)
            x_val = np.array([[-1.0, 2.0], [3.0, -4.0]], dtype=np.float32)
            cfg_2x2 = TensorConfig(shape=[2, 2], dtype="float32", device="cpu")
            out = runner(Tensor(x_val, cfg_2x2))
            as_callable(np.testing.assert_allclose)(out, as_callable(np.maximum)(x_val, 0.0))


def test_tf_generator_aot_fallback() -> None:
    """Verify compile_aot exception fallback to interpreter evaluation."""
    import importlib.util
    import sys
    from unittest.mock import MagicMock, patch

    import numpy as np

    from ml_switcheroo_compiler.backends.tensorflow.generator import TensorFlowCodeGenerator
    from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
    from ml_switcheroo_compiler.ir.core import IRGraph, IRNode

    mock_tf = MagicMock()
    ctx = patch.dict(sys.modules, {"tensorflow": mock_tf}) if importlib.util.find_spec("tensorflow") is None else MagicMock()

    g_multi = IRGraph()
    n_a = IRNode(id="a", op_type="Input", shape_metadata=[2], attributes={"dtype": "float32"})
    n_b = IRNode(id="b_extra", op_type="Input", shape_metadata=[2], attributes={"dtype": "float32"})
    n_add = IRNode(id="add", op_type="Add", inputs=["a", "a"], shape_metadata=[2], attributes={"dtype": "float32"})
    n_sub = IRNode(id="sub", op_type="Sub", inputs=["a", "a"], shape_metadata=[2], attributes={"dtype": "float32"})
    g_multi.nodes = NodeDict({"a": n_a, "b_extra": n_b, "add": n_add, "sub": n_sub})
    g_multi.inputs = ["a", "b_extra"]
    g_multi.outputs = ["add", "sub"]
    gen_multi = TensorFlowCodeGenerator(g_multi)

    with ctx:
        with patch("tensorflow.function", side_effect=RuntimeError("TF jit error")):
            with patch(
                "ml_switcheroo_compiler.interpreter.evaluator.evaluate_graph",
                return_value={"add": np.array([2.0, 4.0]), "sub": np.array([0.0, 0.0])},
            ):
                fb_runner = gen_multi.compile_aot(g_multi)

                a_val = np.array([1.0, 2.0], dtype=np.float32)
                res_multi = fb_runner(Tensor(a_val, TensorConfig(shape=[2], dtype="float32", device="cpu")))
                assert isinstance(res_multi, tuple)
                assert len(res_multi) == 2

                g_multi.outputs = ["add"]
                res_single = fb_runner(a_val)
                as_callable(np.testing.assert_allclose)(res_single, [2.0, 4.0])

                g_multi.outputs = []
                res_dict = fb_runner(a_val)
                assert isinstance(res_dict, dict)
