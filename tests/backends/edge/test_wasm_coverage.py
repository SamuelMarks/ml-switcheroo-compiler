from ml_switcheroo_compiler.backends.edge.wasm import WasmCodeGenerator
from ml_switcheroo_compiler.ir.core import IRGraph, IRNode


def test_wasm_coverage_unknown_op():
    g = IRGraph()
    n = IRNode("dummy", "UnknownOp")
    n.inputs = ["dummy_in"]
    n.shape_metadata = 1
    g.inputs = []
    g.outputs = [n]
    g._nodes = {"dummy": n}
    gen = WasmCodeGenerator(g)
    gen.var_names = {"dummy_in": "dummy_in"}
    gen.sorted_nodes = g.inputs + [n]
    code = gen.generate()
    assert "_scalar_unknownop" in code

    n2 = IRNode("dummy_sigmoid", "Sigmoid")
    n2.inputs = ["dummy_in"]
    n2.shape_metadata = 10
    g._nodes["dummy_sigmoid"] = n2
    gen.sorted_nodes.append(n2)
    code = gen.generate()
    assert "std::exp" in code

    n3 = IRNode("dummy_exp", "Exp")
    n3.inputs = ["dummy_in"]
    n3.shape_metadata = 10
    g._nodes["dummy_exp"] = n3
    gen.sorted_nodes.append(n3)
    code = gen.generate()
    assert "std::exp" in code


def test_wasm_conv2d_pool2d_norm_coverage():
    from ml_switcheroo_compiler.backends.edge.wasm import WasmCodeGenerator
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

    gen = WasmCodeGenerator(g)
    gen.sorted_nodes = [in1, in2, w, b, rm, rv, n_conv, n_pool, n_norm, n_ln]

    code = gen.generate()
    assert "for (int n = 0; n < 1; ++n)" in code
    assert "Naive MaxPool2D" in code
    assert "Naive BatchNorm" in code
    assert "Naive LayerNorm" in code


def test_wasm_avgpool2d_coverage():
    from ml_switcheroo_compiler.backends.edge.wasm import WasmCodeGenerator
    from ml_switcheroo_compiler.ir.core import IRGraph, IRNode

    g = IRGraph()
    n_pool = IRNode("pool", "AvgPool2D", inputs=["in1"], attributes={"window_dimensions": (2, 2), "window_strides": (2, 2), "padding": (0, 0)})
    n_pool.shape_metadata = (1, 16, 8, 8)

    in1 = IRNode("in1", "Input")
    in1.shape_metadata = (1, 16, 16, 16)

    g.nodes = {"in1": in1, "pool": n_pool}
    g.inputs = ["in1"]
    g.outputs = ["pool"]

    gen = WasmCodeGenerator(g)
    gen.sorted_nodes = [in1, n_pool]

    code = gen.generate()
    assert "Naive AvgPool2D" in code
    assert "sum += val;" in code


def test_wasm_conv2d_fallback_coverage():
    from ml_switcheroo_compiler.backends.edge.wasm import WasmCodeGenerator
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

    gen = WasmCodeGenerator(g)
    gen.sorted_nodes = [in1, in2, n_conv]

    code = gen.generate()
    assert "SIMD Fallback Copy" in code


def test_wasm_scalar_shapes3():
    from ml_switcheroo_compiler.backends.edge.wasm import WasmCodeGenerator
    from ml_switcheroo_compiler.ir.core import IRGraph, IRNode

    g = IRGraph()
    n_conv = IRNode("conv", "Conv2D", inputs=["in1", "in2"], attributes={"padding": (1, 1)})
    n_conv.shape_metadata = (1, 16, 16, 16)

    in1 = IRNode("in1", "Input")
    in1.shape_metadata = 5.0
    in2 = IRNode("in2", "Input")
    in2.shape_metadata = 5.0

    g.nodes = {"in1": in1, "in2": in2, "conv": n_conv}
    g.inputs = ["in1", "in2"]
    g.outputs = ["conv"]

    gen = WasmCodeGenerator(g)
    gen.sorted_nodes = [in1, in2, n_conv]
    gen.generate()


def test_wasm_scalar_pool_norm():
    from ml_switcheroo_compiler.backends.edge.wasm import WasmCodeGenerator
    from ml_switcheroo_compiler.ir.core import IRGraph, IRNode

    g = IRGraph()
    n_pool = IRNode("pool", "MaxPool2D", inputs=["in1"], attributes={"padding": (1, 1)})
    n_pool.shape_metadata = (1, 16, 8, 8)

    in1 = IRNode("in1", "Input")
    in1.shape_metadata = 5.0

    g.nodes = {"in1": in1, "pool": n_pool}
    g.inputs = ["in1"]
    g.outputs = ["pool"]

    gen = WasmCodeGenerator(g)
    gen.sorted_nodes = [in1, n_pool]
    gen.generate()

    # LayerNorm with bad inputs
    g2 = IRGraph()
    in_ln = IRNode("in_ln", "Input")
    in_ln.shape_metadata = 5.0
    n_ln = IRNode("ln", "LayerNorm", inputs=["in_ln"], attributes={"padding": (1, 1)})
    n_ln.shape_metadata = (1, 16, 8, 8)
    g2.nodes = {"in_ln": in_ln, "ln": n_ln}
    g2.inputs = ["in_ln"]
    g2.outputs = ["ln"]
    gen2 = WasmCodeGenerator(g2)
    gen2.sorted_nodes = [in_ln, n_ln]
    gen2.generate()


def test_wasm_conv2d_scalar_padding():
    from ml_switcheroo_compiler.backends.edge.wasm import WasmCodeGenerator
    from ml_switcheroo_compiler.ir.core import IRGraph, IRNode

    g = IRGraph()
    n_conv = IRNode("conv", "Conv2D", inputs=["in1", "in2"], attributes={"padding": (1, 1)})
    n_conv.shape_metadata = (1, 16, 16, 16)

    in1 = IRNode("in1", "Input")
    in1.shape_metadata = (1, 3, 32, 32)
    in2 = IRNode("in2", "Input")
    in2.shape_metadata = (16, 3, 3, 3)

    n_pool = IRNode("pool", "MaxPool2D", inputs=["conv"], attributes={"padding": (1, 1)})
    n_pool.shape_metadata = (1, 16, 8, 8)

    g.nodes = {"in1": in1, "in2": in2, "conv": n_conv, "pool": n_pool}
    g.inputs = ["in1", "in2"]
    g.outputs = ["pool"]

    gen = WasmCodeGenerator(g)
    gen.sorted_nodes = [in1, in2, n_conv, n_pool]
    gen.generate()


def test_wasm_norm_no_shape():
    from ml_switcheroo_compiler.backends.edge.wasm import WasmCodeGenerator
    from ml_switcheroo_compiler.ir.core import IRGraph, IRNode

    g = IRGraph()
    in1 = IRNode("in1", "Input")
    in1.shape_metadata = None

    n_ln = IRNode("ln", "LayerNorm", inputs=["in1"], attributes={})
    n_ln.shape_metadata = (1, 16, 8, 8)

    g.nodes = {"in1": in1, "ln": n_ln}
    g.inputs = ["in1"]
    g.outputs = ["ln"]

    gen = WasmCodeGenerator(g)
    gen.sorted_nodes = [in1, n_ln]
    gen.generate()


def test_wasm_missing_attributes_and_strides():
    from ml_switcheroo_compiler.backends.edge.wasm import WasmCodeGenerator
    from ml_switcheroo_compiler.ir.core import IRGraph, IRNode

    g = IRGraph()
    n_conv = IRNode("conv", "Conv2D", inputs=["in1", "in2"])
    del n_conv.attributes  # force hasattr(node, "attributes") to be False
    n_conv.shape_metadata = (1, 16, 16, 16)

    in1 = IRNode("in1", "Input")
    in1.shape_metadata = (1, 3, 32, 32)
    in2 = IRNode("in2", "Input")
    in2.shape_metadata = (16, 3, 3, 3)

    n_conv2 = IRNode("conv2", "Conv2D", inputs=["in1", "in2"], attributes={"window_dimensions": [1], "window_strides": [1], "padding": 1})
    n_conv2.shape_metadata = (1, 16, 16, 16)

    n_pool = IRNode("pool", "MaxPool2D", inputs=["conv"])
    del n_pool.attributes
    n_pool.shape_metadata = (1, 16, 8, 8)

    n_pool2 = IRNode("pool2", "MaxPool2D", inputs=["conv"], attributes={"window_dimensions": [1], "window_strides": [1], "padding": 1})
    n_pool2.shape_metadata = (1, 16, 8, 8)

    n_norm = IRNode("norm", "BatchNorm", inputs=["pool", "w", "b", "rm", "rv"])
    del n_norm.attributes
    n_norm.shape_metadata = (1, 16, 8, 8)

    w = IRNode("w", "Input")
    w.shape_metadata = (16,)
    b = IRNode("b", "Input")
    b.shape_metadata = (16,)
    rm = IRNode("rm", "Input")
    rm.shape_metadata = (16,)
    rv = IRNode("rv", "Input")
    rv.shape_metadata = (16,)

    n_ln = IRNode("ln", "LayerNorm", inputs=["pool", "w", "b"])
    del n_ln.attributes
    n_ln.shape_metadata = (1, 16, 8, 8)

    g.nodes = {"in1": in1, "in2": in2, "conv": n_conv, "conv2": n_conv2, "pool": n_pool, "pool2": n_pool2, "norm": n_norm, "w": w, "b": b, "rm": rm, "rv": rv, "ln": n_ln}
    g.inputs = ["in1", "in2", "w", "b", "rm", "rv"]
    g.outputs = ["norm", "ln"]

    gen = WasmCodeGenerator(g)
    gen.sorted_nodes = [in1, in2, w, b, rm, rv, n_conv, n_conv2, n_pool, n_pool2, n_norm, n_ln]

    code = gen.generate()
    assert "Naive BatchNorm" in code
