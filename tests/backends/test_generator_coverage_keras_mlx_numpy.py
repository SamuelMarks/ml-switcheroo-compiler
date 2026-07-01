from ml_switcheroo_compiler.backends.keras.generator import KerasCodeGenerator
from ml_switcheroo_compiler.backends.mlx.generator import MLXCodeGenerator
from ml_switcheroo_compiler.backends.numpy.generator import NumpyGenerator
from ml_switcheroo_compiler.ir.core import IRGraph, IRNode


def test_keras_generator_coverage():
    g = IRGraph()
    gen = KerasCodeGenerator(g)

    n_conv = IRNode(
        id="n1",
        op_type="ConvTranspose",
        inputs=["x", "w"],
        attributes={"strides": 2, "padding": "SAME"},
        shape_metadata=None,
    )
    assert "keras_conv_transpose" in gen.visit(n_conv, ["x", "w"])

    n_ragged = IRNode(
        id="n2", op_type="RaggedDot", inputs=["x", "y"], attributes={}, shape_metadata=None
    )
    assert "keras_ragged_dot" in gen.visit(n_ragged, ["x", "y"])


def test_keras_generator_full():
    g = IRGraph()
    n_in = IRNode(id="n_in", op_type="Parameter", inputs=[], attributes={}, shape_metadata=(2,))
    n_out = IRNode(
        id="n_out", op_type="Return", inputs=["n_in"], attributes={}, shape_metadata=None
    )
    g.nodes["n_in"] = n_in
    g.nodes["n_out"] = n_out

    gen = KerasCodeGenerator(g)
    assert gen.get_fallback_prefix() == "keras.ops"

    from ml_switcheroo_compiler.backends.keras.generator import (
        KerasTensorManipulator,
        KerasSignatureBuilder,
    )

    assert "keras.ops.zeros({shape})" in KerasTensorManipulator.format_zeros_like("zeros", {})
    assert "keras.ops.full({shape}, {fill_value})" in KerasTensorManipulator.format_full({})
    assert "keras.ops.transpose({0})" in KerasTensorManipulator.format_transpose({})

    assert "keras.Input" in KerasSignatureBuilder.get_input_assignment("n_in", n_in)
    assert "keras.Model" in KerasSignatureBuilder.get_return_block(["n_in"], ["n_out"])

    # Call emit functions that lack coverage
    gen._emit_input_assignment("n_in_emit", n_in, "pre", 0)
    gen._emit_body_return(["out_var"])
    gen._emit_output_assignment(n_in, ["out_var2"], "returns")

    assert gen._generate_file_header()
    assert gen._resolve_imports()

    gen._generate_function_signature()
    gen._traverse_ir_graph()
    gen._generate_return_block()

    code = gen.generate()
    assert "def get_model():" in code
    assert "import keras" in code
    assert "return keras.Model(inputs=[" in code


def test_keras_eager_coverage():
    from ml_switcheroo_compiler.backends.keras.eager import execute_op

    try:
        execute_op(None, "UnknownFakeOp", None)
    except NotImplementedError:
        pass

    from ml_switcheroo_compiler.backends.keras.types import zeros, array, asarray, item
    import keras.ops as kops

    assert zeros(None, (2,)) is not None
    assert array(None, [1, 2]) is not None
    assert asarray(None, [3, 4]) is not None
    assert item(None, kops.array([5])) == 5


def test_mlx_generator_coverage():
    g = IRGraph()
    gen = MLXCodeGenerator(g)

    n_conv = IRNode(
        id="n1",
        op_type="ConvTranspose",
        inputs=["x", "w"],
        attributes={"strides": 2, "padding": "SAME"},
        shape_metadata=None,
    )
    assert "mx.conv_transpose" in gen.visit(n_conv, ["x", "w"])

    n_ragged = IRNode(
        id="n2", op_type="RaggedDot", inputs=["x", "y"], attributes={}, shape_metadata=None
    )
    assert "mlx_ragged_dot" in gen.visit(n_ragged, ["x", "y"])

    # MLX eager execution ops via generator mapping. Some are not natively mapped in mlx generator
    n_chol = IRNode(
        id="n_chol", op_type="Cholesky", inputs=["x"], attributes={}, shape_metadata=None
    )
    assert "cholesky" in gen.visit(n_chol, ["x"])

    n_power = IRNode(
        id="n_pi",
        op_type="PowerIteration",
        inputs=["x", "u"],
        attributes={"num_iters": 2},
        shape_metadata=None,
    )
    assert "mlx_power_iteration" in gen.visit(n_power, ["x", "u"])
    assert "mlx_power_iteration" in gen.visit(n_power, ["x"])

    n_zeros = IRNode(
        id="n_zeros",
        op_type="Zeros",
        inputs=[],
        attributes={"dtype": "float32"},
        shape_metadata=(2, 3),
    )
    assert "mx.zeros" in gen.visit(n_zeros, [])

    assert gen.get_fallback_prefix() == "mx"
    assert "mx.zeros" in gen._format_zeros_ones("Zeros", {})
    assert "mx.zeros" in gen._format_zeros_ones("Zeros", {"dtype": "float32"})

    gen._emit_constant_assignment("c", "1")
    assert "c = mx.array(1)" in "\n".join(gen.code)

    prefix = gen._get_prefix_code()
    assert len(prefix) > 0
    assert "import mlx.core as mx" in prefix[0]

    map_ops = gen.get_ops_map({})
    assert map_ops["Beta"] == "mx.random.uniform(shape={shape})"


def test_numpy_generator_coverage():
    g = IRGraph()
    gen = NumpyGenerator(g)

    from ml_switcheroo_compiler.backends.numpy.generator import NumpyASTVisitor

    n_triinv = IRNode(id="n1", op_type="TriInv", inputs=["x"], attributes={}, shape_metadata=None)
    assert "np.linalg.inv" in NumpyASTVisitor.visit_TriInv(n_triinv, ["x"])

    n_trunc_div = IRNode(
        id="n2", op_type="TruncateDiv", inputs=["x", "y"], attributes={}, shape_metadata=None
    )
    assert "np.trunc(np.divide(x, y))" in NumpyASTVisitor.visit_TruncateDiv(n_trunc_div, ["x", "y"])

    n_trunc_mod = IRNode(
        id="n3", op_type="TruncateMod", inputs=["x", "y"], attributes={}, shape_metadata=None
    )
    assert "np.fmod(x, y)" in NumpyASTVisitor.visit_TruncateMod(n_trunc_mod, ["x", "y"])

    n_generic = IRNode(
        id="n4", op_type="UnknownOp", inputs=["x"], attributes={}, shape_metadata=None
    )
    assert "np.unknownop(x)" in NumpyASTVisitor.generic_visit(n_generic, ["x"])

    n_generic2 = IRNode(
        id="n5", op_type="UnknownOp", inputs=["x"], attributes={"dimension": 1}, shape_metadata=None
    )
    assert "np.unknownop(x, axis=1)" in NumpyASTVisitor.generic_visit(
        n_generic2, ["x"], dimension=1
    )

    n_pi = IRNode(
        id="n_pi",
        op_type="PowerIteration",
        inputs=["x", "u"],
        attributes={"num_iters": 2},
        shape_metadata=None,
    )
    assert "np_power_iteration" in gen.visit(n_pi, ["x", "u"])
    assert "np_power_iteration" in gen.visit(n_pi, ["x"])
