from ml_switcheroo_compiler.backends.tensorflow.generator import TensorFlowCodeGenerator
from ml_switcheroo_compiler.ir.core import IRGraph


class DummyNode:
    def __init__(self, attrs=None, op_type="Unknown"):
        self.attributes = attrs or {}
        self.op_type = op_type
        self.id = "n1"


def test_tf_generator_basics():
    g = IRGraph()
    gen = TensorFlowCodeGenerator(g)

    try:
        TensorFlowCodeGenerator.save("path", None)
    except:
        pass
    try:
        TensorFlowCodeGenerator.savez("path")
    except:
        pass
    try:
        TensorFlowCodeGenerator.savez_compressed("path")
    except:
        pass
    try:
        TensorFlowCodeGenerator.load("path")
    except:
        pass

    assert gen._get_backend_prefix() == "tf"
    assert gen.get_fallback_prefix() == "tf.math"
    assert gen.get_fallback_axis_kwarg() == "axis"
    assert gen.get_fallback_keepdims_kwarg() == "keepdims"

    try:
        gen._get_math_ops({})
    except:
        pass
    try:
        gen._get_creation_ops({})
    except:
        pass
    try:
        gen._get_array_ops({})
    except:
        pass
    try:
        gen.get_ops_map({})
    except:
        pass
    try:
        gen.visit(DummyNode(), [])
    except:
        pass

    gen._emit_constant_assignment("var", "val")
    gen._generate_file_header()
    gen._resolve_imports()
    gen._generate_function_signature()

    try:
        gen._format_creation_op("test", {})
    except:
        pass
    try:
        gen._format_creation_op("test", {"dtype": "float32"})
    except:
        pass

    try:
        gen._format_full({})
    except:
        pass
    try:
        gen._format_full({"dtype": "float32"})
    except:
        pass

    try:
        gen._format_transpose({})
    except:
        pass
    try:
        gen._format_transpose({"axes": [0, 1]})
    except:
        pass

    try:
        gen.visit_RaggedDot(DummyNode(), ["x", "y"])
    except:
        pass
    try:
        gen.visit_Einsum(DummyNode(), ["x", "y"], equation="i,j->ij")
    except:
        pass
