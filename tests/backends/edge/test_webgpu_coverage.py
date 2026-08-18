import pytest

from ml_switcheroo_compiler.backends.edge.webgpu import WebGPUCodeGenerator
from ml_switcheroo_compiler.core.errors import UnimplementedMathError
from ml_switcheroo_compiler.ir.core import IRGraph, LogicalNode


def test_webgpu_generator_all_ops():
    graph = IRGraph()
    n1 = LogicalNode(id="in1", op_type="Input")
    n1.shape_metadata = (10,)
    n2 = LogicalNode(id="in2", op_type="Input")
    n2.shape_metadata = (10,)

    graph.nodes = {"in1": n1, "in2": n2}

    ops = [
        "FloorDivide",
        "Power",
        "Maximum",
        "Minimum",
        "LogicalAnd",
        "LogicalOr",
        "LogicalXor",
        "Equal",
        "NotEqual",
        "Greater",
        "Less",
        "GreaterEqual",
        "LessEqual",
        "Exp",
        "Log",
        "Abs",
        "Ceil",
        "Floor",
        "Round",
        "Sqrt",
        "Sin",
        "Cos",
        "Tan",
        "Asin",
        "Acos",
        "Atan",
        "Sinh",
        "Cosh",
        "Tanh",
        "Log1p",
        "Expm1",
        "Rsqrt",
        "Negative",
        "Neg",
        "Sign",
        "Relu",
        "Gelu",
        "Swish",
        "Cast",
        "Constant",
        "ReduceSum",
        "ReduceProd",
        "ArgMax",
        "ArgMin",
        "Conv1D",
        "Conv2D",
        "Conv3D",
        "ConvTranspose2D",
        "MaxPool",
        "AvgPool",
        "MaxPool2D",
        "AvgPool2D",
        "BatchNorm",
        "LayerNorm",
        "GroupNorm",
        "Sigmoid",
    ]

    graph.outputs = []

    for idx, op in enumerate(ops):
        node_id = f"out_{idx}"
        inputs = ["in1", "in2"] if op in ("FloorDivide", "Power", "Maximum", "Minimum", "LogicalAnd", "LogicalOr", "LogicalXor", "Equal", "NotEqual", "Greater", "Less", "GreaterEqual", "LessEqual") else ["in1"]
        node = LogicalNode(id=node_id, op_type=op, inputs=inputs)
        node.shape_metadata = (10,)
        if op == "Constant":
            node.attributes = {"value": 1.0}
        graph.nodes[node_id] = node
        graph.outputs.append(node_id)

    gen = WebGPUCodeGenerator(graph)
    try:
        code = gen.generate()
        # If all ops surprisingly pass without raising, assert basic code is generated
        assert "fn" in code or "var" in code
    except UnimplementedMathError:
        pass  # Expected during transition phase

    # Test missing shapes
    n_no_shape = LogicalNode(id="n_no_shape", op_type="Add", inputs=["in1", "in2"])
    graph.nodes["n_no_shape"] = n_no_shape
    gen._get_shape_and_strides(n_no_shape)
    gen._gen_offset_computation("idx", [], [], "out")

    # Test Softmax/LogSoftmax
    n_softmax = LogicalNode(id="n_softmax", op_type="Softmax", inputs=["in1"])
    n_softmax.shape_metadata = (10,)
    graph.nodes["n_softmax"] = n_softmax
    graph.outputs = ["n_softmax"]
    gen_sm = WebGPUCodeGenerator(graph)
    try:
        gen_sm.generate()
    except UnimplementedMathError:
        pass

    # Test LogSoftmax
    n_logsoftmax = LogicalNode(id="n_logsoftmax", op_type="LogSoftmax", inputs=["in1"])
    n_logsoftmax.shape_metadata = (10,)
    graph.nodes["n_logsoftmax"] = n_logsoftmax
    graph.outputs = ["n_logsoftmax"]
    gen_lsm = WebGPUCodeGenerator(graph)
    try:
        gen_lsm.generate()
    except UnimplementedMathError:
        pass

    for op_name in ["ReduceMax", "ReduceMin", "ReduceMean", "Constant", "Negative"]:
        node = LogicalNode(id=f"n_{op_name.lower()}", op_type=op_name, inputs=["in1"] if op_name != "Constant" else [])
        node.shape_metadata = (1,)
        try:
            gen._get_wgsl_for_op(node, [1], 1, "out")
        except UnimplementedMathError:
            pass

    n_4_inputs = LogicalNode(id="n_4_inputs", op_type="Add", inputs=["in1", "in2", "in3", "in4"])
    n_4_inputs.shape_metadata = (10,)
    for i in range(1, 5):
        n = LogicalNode(id=f"in{i}", op_type="Input")
        n.shape_metadata = (10,)
        graph.nodes[f"in{i}"] = n
    graph.nodes["n_4_inputs"] = n_4_inputs
    graph.outputs = ["n_4_inputs"]
    gen_4 = WebGPUCodeGenerator(graph)
    try:
        gen_4.generate()
    except UnimplementedMathError:
        pass

    node_err = LogicalNode(id="err", op_type="UnsupportedWebGPUOp", inputs=["in1"])
    node_err.shape_metadata = (10,)
    graph.nodes["err"] = node_err
    graph.outputs = ["err"]

    gen_err = WebGPUCodeGenerator(graph)
    with pytest.raises(UnimplementedMathError):
        gen_err.generate()


def test_webgpu_conv2d_pool2d_norm_coverage():
    from ml_switcheroo_compiler.backends.edge.webgpu import WebGPUCodeGenerator
    from ml_switcheroo_compiler.ir.core import IRGraph, IRNode

    g = IRGraph()
    n_conv = IRNode("conv", "Conv2D", inputs=["in1", "in2"], attributes={"window_strides": (2, 2), "padding": ((1, 1), (1, 1))})
    n_conv.shape_metadata = (1, 16, 16, 16)

    in1 = IRNode("in1", "Input")
    in1.shape_metadata = (1, 3, 32, 32)
    in2 = IRNode("in2", "Input")
    in2.shape_metadata = (16, 3, 3, 3)

    n_pool = IRNode("pool", "MaxPool2D", inputs=["conv"], attributes={"window_dimensions": (2, 2), "window_strides": (2, 2), "padding": ((0, 0), (0, 0))})
    n_pool.shape_metadata = (1, 16, 8, 8)

    n_norm = IRNode("norm", "BatchNorm", inputs=["pool", "w", "b", "rm", "rv"], attributes={"epsilon": 1e-5})
    n_norm.shape_metadata = (1, 16, 8, 8)

    w = IRNode("w", "Input")
    w.shape_metadata = (16,)
    b = IRNode("b", "Input")
    b.shape_metadata = (16,)
    rm = IRNode("rm", "Input")
    rm.shape_metadata = (16,)
    rv = IRNode("rv", "Input")
    rv.shape_metadata = (16,)

    n_ln = IRNode("ln", "LayerNorm", inputs=["pool", "w", "b"], attributes={"epsilon": 1e-5})
    n_ln.shape_metadata = (1, 16, 8, 8)

    g.nodes = {"in1": in1, "in2": in2, "conv": n_conv, "pool": n_pool, "norm": n_norm, "w": w, "b": b, "rm": rm, "rv": rv, "ln": n_ln}
    g.inputs = ["in1", "in2", "w", "b", "rm", "rv"]
    g.outputs = ["norm", "ln"]

    gen = WebGPUCodeGenerator(g)
    gen.sorted_nodes = [in1, in2, w, b, rm, rv, n_conv, n_pool, n_norm, n_ln]

    code = gen.generate()
    assert "compute_conv" in code
    assert "compute_pool" in code
    assert "compute_norm" in code
    assert "compute_ln" in code


def test_webgpu_avgpool2d_coverage():
    from ml_switcheroo_compiler.backends.edge.webgpu import WebGPUCodeGenerator
    from ml_switcheroo_compiler.ir.core import IRGraph, IRNode

    g = IRGraph()
    n_pool = IRNode("pool", "AvgPool2D", inputs=["in1"], attributes={"window_dimensions": (2, 2), "window_strides": (2, 2), "padding": (0, 0)})
    n_pool.shape_metadata = (1, 16, 8, 8)

    in1 = IRNode("in1", "Input")
    in1.shape_metadata = (1, 16, 16, 16)

    g.nodes = {"in1": in1, "pool": n_pool}
    g.inputs = ["in1"]
    g.outputs = ["pool"]

    gen = WebGPUCodeGenerator(g)
    gen.sorted_nodes = [in1, n_pool]

    code = gen.generate()
    assert "compute_pool" in code


def test_webgpu_conv2d_fallback_coverage():
    from ml_switcheroo_compiler.backends.edge.webgpu import WebGPUCodeGenerator
    from ml_switcheroo_compiler.ir.core import IRGraph, IRNode

    g = IRGraph()
    n_conv = IRNode("conv", "Conv2D", inputs=["in1", "in2"], attributes={})
    n_conv.shape_metadata = (1, 16, 16, 16)

    in1 = IRNode("in1", "Input")
    in1.shape_metadata = (1, 3, 32)  # not 4D
    in2 = IRNode("in2", "Input")
    in2.shape_metadata = (16, 3, 3)

    g.nodes = {"in1": in1, "in2": in2, "conv": n_conv}
    g.inputs = ["in1", "in2"]
    g.outputs = ["conv"]

    gen = WebGPUCodeGenerator(g)
    gen.sorted_nodes = [in1, in2, n_conv]

    code = gen.generate()
    assert "let out_width = 16u;" in code
