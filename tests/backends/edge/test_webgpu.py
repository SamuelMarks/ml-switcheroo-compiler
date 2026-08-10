from ml_switcheroo_compiler.backends.edge.webgpu import WebGPUCodeGenerator
from ml_switcheroo_compiler.ir.core import IRGraph, LogicalNode


def test_webgpu_generator_matmul_dynamic_dispatch():
    graph = IRGraph()
    n1 = LogicalNode(id="in1", op_type="Input")
    n1.shape_metadata = (32, 64)
    n2 = LogicalNode(id="in2", op_type="Input")
    n2.shape_metadata = (64, 128)
    n3 = LogicalNode(id="out", op_type="MatMul", inputs=["in1", "in2"])
    n3.shape_metadata = (32, 128)
    graph.nodes = {"in1": n1, "in2": n2, "out": n3}

    gen = WebGPUCodeGenerator(graph)
    code = gen.generate()

    assert "@workgroup_size(16, 16)" in code
    assert "Math.ceil(128 / 16)" in code
    assert "Math.ceil(32 / 16)" in code


def test_webgpu_generator_elementwise_dynamic_dispatch():
    graph = IRGraph()
    n1 = LogicalNode(id="in1", op_type="Input")
    n1.shape_metadata = (1000,)
    n2 = LogicalNode(id="out", op_type="Add", inputs=["in1", "in1"])
    n2.shape_metadata = (1000,)
    n3 = LogicalNode(id="out2", op_type="ReduceMean", inputs=["in1"])
    n3.shape_metadata = (1,)
    n3.inputs_nelem = [1000]
    n4 = LogicalNode(id="out3", op_type="Negative", inputs=["in1"])
    n4.shape_metadata = (1000,)
    n6 = LogicalNode(id="out_mat", op_type="MatMul", inputs=["in1", "in1"])
    n6.shape_metadata = ()
    n6_2 = LogicalNode(id="out_mat2", op_type="MatMul", inputs=["in1", "in1"])
    n6_2.shape_metadata = (1, 2, 3)
    n7 = LogicalNode(id="out_red", op_type="ReduceSum", inputs=["in1"])
    n7.shape_metadata = (1,)
    n8 = LogicalNode(id="out_red2", op_type="ReduceMean", inputs=["in1"])
    n8.shape_metadata = (1,)
    n9 = LogicalNode(id="out_red3", op_type="ReduceMax", inputs=["in1"])
    n9.shape_metadata = (1,)
    n10 = LogicalNode(id="out_red4", op_type="ReduceMin", inputs=["in1"])
    n10.shape_metadata = (1,)
    graph.nodes = {"in1": n1, "out": n2, "out2": n3, "out3": n4, "out_mat": n6, "out_mat2": n6_2, "out_red": n7, "out_red2": n8, "out_red3": n9, "out_red4": n10}
    graph.outputs = ["out2", "out_mat", "out_mat2", "out_red", "out_red2", "out_red3", "out_red4"]

    gen = WebGPUCodeGenerator(graph)
    code = gen.generate()

    assert "@workgroup_size(64)" in code
    assert "Math.ceil(1000 / 64)" in code
    assert "-buf_in0_f32[in0_offset]" in code
    assert "max(res, buf_in0_f32[i])" in code
    assert "min(res, buf_in0_f32[i])" in code


def test_webgpu_generator_get_offset():
    graph = IRGraph()
    n1 = LogicalNode(id="in1", op_type="Input")
    n1.shape_metadata = (10, 20)
    n2 = LogicalNode(id="out", op_type="Add", inputs=["in1", "in1"])
    n2.shape_metadata = (10, 20)
    graph.nodes = {"in1": n1, "out": n2}

    gen = WebGPUCodeGenerator(graph)
    code = gen.generate()

    assert "let out_offset_d1 = out_offset_remaining % 20u;" in code
    assert "out_offset_offset = out_offset_offset + out_offset_d1 * 1u;" in code


def test_map_type_float64():
    g = IRGraph()
    gen = WebGPUCodeGenerator(g)
    assert gen._map_type("float64") == "f32"
    assert gen._map_type("unknown") == "f32"


def test_get_shape_and_strides_scalar():
    g = IRGraph()
    gen = WebGPUCodeGenerator(g)
    node = LogicalNode(id="n1", op_type="Input")
    node.shape_metadata = 5
    shape, strides = gen._get_shape_and_strides(node)
    assert shape == [5]
    assert strides == [1]


def test_get_shape_and_strides_empty():
    g = IRGraph()
    gen = WebGPUCodeGenerator(g)
    node = LogicalNode(id="n1", op_type="Input")
    node.shape_metadata = ()
    shape, strides = gen._get_shape_and_strides(node)
    assert shape == []
    assert strides == []


def test_num_elements_empty():
    g = IRGraph()
    gen = WebGPUCodeGenerator(g)
    assert gen._num_elements([]) == 1


def test_generate_empty_graph():
    g = IRGraph()
    gen = WebGPUCodeGenerator(g)
    code = gen.generate()
    assert "async function run" in code


def test_generic_visit_no_id():
    g = IRGraph()
    gen = WebGPUCodeGenerator(g)

    class DummyNode:
        pass

    assert gen.generic_visit(DummyNode(), []) == ""


def test_webgpu_fused_ops():
    from ml_switcheroo_compiler.backends.edge.webgpu import WebGPUCodeGenerator
    from ml_switcheroo_compiler.ir.core import IRGraph, LogicalNode

    g = IRGraph()
    n1 = LogicalNode(id="n1", op_type="Input")
    n1.shape_metadata = (2, 2)
    n2 = LogicalNode(id="n2", op_type="Input")
    n2.shape_metadata = (2, 2)
    n3 = LogicalNode(id="n3", op_type="Input")
    n3.shape_metadata = (2, 2)

    n_relu = LogicalNode(id="relu", op_type="FusedAddRelu", inputs=["n1", "n2"])
    n_relu.shape_metadata = (2, 2)
    n_fma = LogicalNode(id="fma", op_type="FusedMultiplyAdd", inputs=["n1", "n2", "n3"])
    n_fma.shape_metadata = (2, 2)
    n_logexp = LogicalNode(id="logexp", op_type="FusedLogExp", inputs=["n1"])
    n_logexp.shape_metadata = (2, 2)

    g.nodes = {"n1": n1, "n2": n2, "n3": n3, "relu": n_relu, "fma": n_fma, "logexp": n_logexp}
    gen = WebGPUCodeGenerator(g)
    code = gen.generate()
    assert "max(0.0" in code
    assert "log(exp(" in code
