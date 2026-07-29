from ml_switcheroo_ir import LogicalGraph

from ml_switcheroo_compiler.backends.tensorflow.generator import TensorFlowCodeGenerator


def test_tf_generator_methods():
    gen = TensorFlowCodeGenerator(LogicalGraph())
    assert gen._get_backend_prefix() == "tf"
    assert gen.get_fallback_prefix() == "tf.math"

    # generate file header
    header = gen._generate_file_header()
    assert isinstance(header, list)

    # check zero format
    assert "tf.zeros" in gen._format_zeros_like("zeros", {})
    assert "dtype='float32'" in gen._format_zeros_like("zeros", {"dtype": "float32"})

    # check format full
    assert "tf.full" in gen._format_full({"fill_value": 5})
    assert "dtype='float32'" in gen._format_full({"dtype": "float32"})

    # ops map
    ops = gen.get_ops_map({})
    assert "Zeros" in ops

    # const assignment
    gen.code = []
    gen._emit_constant_assignment("var", "val")
    assert len(gen.code) == 1
    assert "tf.constant" in gen.code[0]

    # imports
    imports = gen._resolve_imports()
    assert "import tensorflow as tf\n" in imports

    # func signature
    gen.code = []
    gen.indent_level = 0
    gen._generate_function_signature()
    assert "def apply_model" in gen.code[1]


def test_tf_generator_static_methods(tmp_path):
    # Testing pickle dumping/loading since tf generator uses pickle directly
    filepath = tmp_path / "test.pkl"
    TensorFlowCodeGenerator.save(str(filepath), [1, 2, 3])
    loaded = TensorFlowCodeGenerator.load(str(filepath))
    assert loaded == [1, 2, 3]

    filepath_z = tmp_path / "test_z.pkl"
    TensorFlowCodeGenerator.savez(str(filepath_z), a=[1, 2, 3])
    loaded_z = TensorFlowCodeGenerator.load(str(filepath_z))
    assert loaded_z["a"] == [1, 2, 3]

    import gzip
    import pickle

    filepath_comp = tmp_path / "test_comp.pkl"
    TensorFlowCodeGenerator.savez_compressed(str(filepath_comp), a=[1, 2, 3])
    # load a gzip file explicitly
    with gzip.open(str(filepath_comp), "rb") as f:
        loaded_comp = pickle.load(f)
    assert loaded_comp["a"] == [1, 2, 3]
