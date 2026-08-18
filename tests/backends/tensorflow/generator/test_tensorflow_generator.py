from ml_switcheroo_compiler.backends.tensorflow.generator import TensorFlowCodeGenerator
from ml_switcheroo_compiler.ir.core import IRGraph


class DummyNode:
    def __init__(self, attrs=None, op_type="Unknown"):
        self.attributes = attrs or {}
        self.op_type = op_type
        self.id = "n1"


def test_tf_generator_basics(tmp_path):
    g = IRGraph()
    gen = TensorFlowCodeGenerator(g)

    import numpy as np

    arr = np.array([1, 2, 3])
    file_npy = tmp_path / "test.npy"
    TensorFlowCodeGenerator.save(str(file_npy), arr)
    res_npy = TensorFlowCodeGenerator.load(str(file_npy))
    np.testing.assert_array_equal(res_npy, arr)

    file_npz = tmp_path / "test.npz"
    TensorFlowCodeGenerator.savez(str(file_npz), a=arr)
    res_npz = TensorFlowCodeGenerator.load(str(file_npz))
    np.testing.assert_array_equal(res_npz["a"], arr)

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

    assert gen.visit_RaggedDot(DummyNode(), ["x", "y"]) == "tf_ragged_dot(x, y)"
    assert gen.visit_Einsum(DummyNode(), ["x", "y"], equation="i,j->ij") == "tf.einsum('i,j->ij', x, y)"

    assert gen.get_fallback_prefix() == "tf.math"
    assert "Zeros" in gen._get_creation_ops({})

    gen._emit_constant_assignment("c", "5")
    assert "c = tf.constant(5)" in gen.code[-1]

    assert gen._generate_file_header() == [gen.header.strip()]
    assert gen._resolve_imports() == ["import tensorflow as tf\n"]

    gen.code = []
    gen._generate_function_signature()
    assert "def apply_model(*args, **kwargs):" in gen.code[-1]
