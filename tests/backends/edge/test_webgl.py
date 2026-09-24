from ml_switcheroo_compiler.backends.edge.webgl import WebGLCodeGenerator
from ml_switcheroo_compiler.ir.core import IRGraph, IRNode


def test_webgl_generator():
    """Test WebGL code generation."""
    graph = IRGraph()
    graph.nodes = {"matmul": IRNode(id="matmul", op_type="MatMul", inputs=[])}
    gen = WebGLCodeGenerator(graph)
    out = gen.generate()
    assert "shader_matmul" in out


def test_webgl_generator_coverage():
    """Test WebGL generator coverage."""
    graph = IRGraph()
    graph.nodes = {"input": IRNode(id="input", op_type="Input", inputs=[])}
    gen = WebGLCodeGenerator(graph)
    out = gen.generate()
    assert "shader_input" not in out


def test_webgl_generator_shapes_and_inputs():
    graph = IRGraph()
    n0 = IRNode(id="n0", op_type="Input", inputs=[], shape_metadata=(16, 32))
    n1 = IRNode(id="n1", op_type="Input", inputs=[], shape_metadata=(32, 64))
    n2 = IRNode(id="n2", op_type="MatMul", inputs=["n0", "n1"], shape_metadata=(16, 64))
    n3 = IRNode(id="n3", op_type="Add", inputs=["n2", "n2"], shape_metadata=(16, 64))
    n4 = IRNode(id="n4", op_type="Exp", inputs=["n3"], shape_metadata=(64,))
    graph.nodes = {"n0": n0, "n1": n1, "n2": n2, "n3": n3, "n4": n4}
    gen = WebGLCodeGenerator(graph)
    out = gen.generate()
    assert "shader_n2" in out
    assert "shader_n3" in out
    assert "shader_n4" in out
    assert "getUniformLocation" in out


def test_webgl_generator_missing_op():
    import pytest

    graph = IRGraph()
    n = IRNode(id="n", op_type="MissingOp", inputs=[])
    graph.nodes = {"n": n}
    gen = WebGLCodeGenerator(graph)
    with pytest.raises(ValueError, match="Missing WebGL shader template for operation: MissingOp"):
        gen.generate()


def test_webgl_generator_custom_setup():
    graph = IRGraph()
    n = IRNode(id="n", op_type="Add", inputs=[], shape_metadata=(16, 32))
    graph.nodes = {"n": n}
    gen = WebGLCodeGenerator(graph)
    from ml_switcheroo_compiler.backends.edge.config_models import WebglTemplateConfig

    gen.config.templates["add"] = WebglTemplateConfig(body="test", custom_setup="let test = {k_dim};")
    out = gen.generate()
    assert "let test = 32;" in out

    gen.config.templates["add"] = "test_string_body"
    out2 = gen.generate()
    assert "test_string_body" in out2


def test_webgl_generator_custom_setup_no_in_shape():
    graph = IRGraph()
    n_in = IRNode(id="n_in", op_type="Input", inputs=[], shape_metadata=())
    n = IRNode(id="n", op_type="Add", inputs=["n_in"], shape_metadata=(16, 32))
    graph.nodes = {"n_in": n_in, "n": n}
    gen = WebGLCodeGenerator(graph)
    from ml_switcheroo_compiler.backends.edge.config_models import WebglTemplateConfig

    gen.config.templates["add"] = WebglTemplateConfig(body="test", custom_setup="let test = {k_dim};")
    out = gen.generate()
    assert "let test = 32;" in out


def test_webgl_missing_yaml(monkeypatch):
    import os

    monkeypatch.setattr(os.path, "exists", lambda p: False)
    graph = IRGraph()
    gen = WebGLCodeGenerator(graph)
    assert gen.config.templates == {}


def test_webgl_expanded_ops():
    """Test emission of newly added WebGL operations."""
    ops = [
        ("conv", "Conv2D", [IRNode(id="in_x", op_type="Input", shape_metadata=(1, 32, 32)), IRNode(id="in_w", op_type="Input", shape_metadata=(3, 3))]),
        ("maxpool", "MaxPool2D", [IRNode(id="in_m", op_type="Input", shape_metadata=(1, 32, 32))]),
        ("avgpool", "AvgPool2D", [IRNode(id="in_a", op_type="Input", shape_metadata=(1, 32, 32))]),
        ("bn", "BatchNorm", [IRNode(id="in_bn", op_type="Input", shape_metadata=(1, 32))]),
        ("ln", "LayerNorm", [IRNode(id="in_ln", op_type="Input", shape_metadata=(1, 32))]),
        ("sm", "Softmax", [IRNode(id="in_sm", op_type="Input", shape_metadata=(1, 32))]),
        ("tp", "Transpose", [IRNode(id="in_tp", op_type="Input", shape_metadata=(16, 32))]),
        ("bc", "BroadcastTo", [IRNode(id="in_bc", op_type="Input", shape_metadata=(1, 32))]),
        ("dw", "DepthwiseConv2D", [IRNode(id="in_dw_x", op_type="Input", shape_metadata=(1, 32, 32)), IRNode(id="in_dw_w", op_type="Input", shape_metadata=(3, 3))]),
        ("mha", "MultiheadAttention", [IRNode(id="in_q", op_type="Input", shape_metadata=(16, 32)), IRNode(id="in_k", op_type="Input", shape_metadata=(16, 32)), IRNode(id="in_v", op_type="Input", shape_metadata=(16, 32))]),
    ]
    for prefix, op_type, inputs in ops:
        graph = IRGraph()
        input_ids = []
        for inp in inputs:
            graph.nodes[inp.id] = inp
            input_ids.append(inp.id)
        node = IRNode(id=f"{prefix}_node", op_type=op_type, inputs=input_ids, shape_metadata=(32, 32))
        graph.nodes[node.id] = node
        gen = WebGLCodeGenerator(graph)
        out = gen.generate()
        assert f"shader_{prefix}_node" in out


def test_webgl_texture_packing_schema():
    """Test Pydantic schema for WebGL float texture packing and framebuffer attachments."""
    import pytest
    from pydantic import ValidationError

    from ml_switcheroo_compiler.backends.edge.config_models import WebglTexturePackingConfig

    # Valid R32F configuration
    cfg_r32 = WebglTexturePackingConfig(
        internal_format="R32F",
        format="RED",
        type="FLOAT",
        attachment="COLOR_ATTACHMENT0",
        viewport_width=64,
        viewport_height=64,
        channels=1,
    )
    assert cfg_r32.internal_format == "R32F"
    assert cfg_r32.format == "RED"
    assert cfg_r32.channels == 1

    # Valid RGBA32F configuration
    cfg_rgba = WebglTexturePackingConfig(
        internal_format="RGBA32F",
        format="RGBA",
        type="FLOAT",
        attachment="COLOR_ATTACHMENT0",
        viewport_width=128,
        viewport_height=128,
        channels=4,
    )
    assert cfg_rgba.internal_format == "RGBA32F"
    assert cfg_rgba.format == "RGBA"
    assert cfg_rgba.channels == 4

    # Invalid internal_format / format mismatch
    with pytest.raises(ValidationError):
        WebglTexturePackingConfig(internal_format="R32F", format="RGBA", channels=1)

    with pytest.raises(ValidationError):
        WebglTexturePackingConfig(internal_format="RGBA32F", format="RED", channels=4)

    # Invalid channel count for R32F
    with pytest.raises(ValidationError):
        WebglTexturePackingConfig(internal_format="R32F", format="RED", channels=4)

    # Viewport bounds
    with pytest.raises(ValidationError):
        WebglTexturePackingConfig(viewport_width=0)


def test_webgl_multipass_pingpong():
    """Test multi-pass compute graph dispatch with FBO ping-ponging without CPU stalls."""
    graph = IRGraph()
    n_in = IRNode(id="inp", op_type="Input", shape_metadata=(32, 32))
    n1 = IRNode(id="relu1", op_type="ReLU", inputs=["inp"], shape_metadata=(32, 32))
    n2 = IRNode(id="relu2", op_type="ReLU", inputs=["relu1"], shape_metadata=(32, 32))
    n3 = IRNode(id="relu3", op_type="ReLU", inputs=["relu2"], shape_metadata=(32, 32))
    graph.nodes = {"inp": n_in, "relu1": n1, "relu2": n2, "relu3": n3}

    gen = WebGLCodeGenerator(graph)
    out = gen.generate()

    # Verify ping-pong FBO creation and alternating switch
    assert "const main_fbo = gl.createFramebuffer();" in out
    assert "const fboPong = gl.createFramebuffer();" in out
    assert "currentFbo = (currentFbo === main_fbo) ? fboPong : main_fbo;" in out

    # Verify readPixels is only called once at the end (no intermediate CPU stalls)
    assert out.count("return readPixels(gl,") == 1


def test_webgl_generator_five_inputs_and_k_shape_variants():
    """Test webgl node with 5 inputs (exceeding uniform names) and k_shape variants."""
    from ml_switcheroo_compiler.backends.edge.config_models import WebglTemplateConfig

    # 1. Five inputs (exercises i >= len(uniform_names))
    g = IRGraph()
    for idx in range(5):
        g.nodes[f"in{idx}"] = IRNode(id=f"in{idx}", op_type="Input", shape_metadata=(16, 16))
    g.nodes["add5"] = IRNode(id="add5", op_type="Add", inputs=["in0", "in1", "in2", "in3", "in4"], shape_metadata=(16, 16))
    gen = WebGLCodeGenerator(g)
    out = gen.generate()
    assert "shader_add5" in out

    # 2. Custom setup with non-int k_shape (exercises lines 118-122 branches)
    g2 = IRGraph()
    g2.nodes["in0"] = IRNode(id="in0", op_type="Input", shape_metadata=(16, 16))
    g2.nodes["in1_sym"] = IRNode(id="in1_sym", op_type="Input", shape_metadata=("sym_h", "sym_w"))
    g2.nodes["conv_sym"] = IRNode(id="conv_sym", op_type="Add", inputs=["in0", "in1_sym"], shape_metadata=(16, 16))
    gen2 = WebGLCodeGenerator(g2)
    gen2.config.templates["add"] = WebglTemplateConfig(body="test", custom_setup="let kw = {k_w}; let kh = {k_h};")
    out2 = gen2.generate()
    assert "let kw = 3;" in out2

    # 3. Custom setup with 1D k_shape
    g3 = IRGraph()
    g3.nodes["in0"] = IRNode(id="in0", op_type="Input", shape_metadata=(16, 16))
    g3.nodes["in1_1d"] = IRNode(id="in1_1d", op_type="Input", shape_metadata=(5,))
    g3.nodes["conv_1d"] = IRNode(id="conv_1d", op_type="Add", inputs=["in0", "in1_1d"], shape_metadata=(16, 16))
    gen3 = WebGLCodeGenerator(g3)
    gen3.config.templates["add"] = WebglTemplateConfig(body="test", custom_setup="let kw = {k_w}; let kh = {k_h};")
    out3 = gen3.generate()
    assert "let kw = 3;" in out3


def test_webgl_generator_missing_node_pass_template():
    """Verify fallback JS orchestration when node_pass template is missing, and blank lines in custom setup."""
    from ml_switcheroo_compiler.backends.edge.config_models import WebglTemplateConfig

    graph = IRGraph()
    n = IRNode(id="n0", op_type="Add", inputs=[], shape_metadata=(16, 32))
    graph.nodes = {"n0": n}
    gen = WebGLCodeGenerator(graph)

    # Custom setup with a blank line in middle to cover line 127 branch
    gen.config.templates["add"] = WebglTemplateConfig(body="test", custom_setup="let test = 1;\n\nlet test2 = 2;")
    # Remove node_pass to cover lines 141-155
    gen.config.js_orchestration.pop("node_pass", None)
    out = gen.generate()
    assert "createProgram(gl, vsSource, shader_n0)" in out
    assert "let texOut_n0 = createTexture" in out


def test_webgl_nd_packing_and_mrt():
    """Verify arbitrary N-D tensor shape packing, stride descriptors, and MRT drawBuffers."""
    graph = IRGraph()
    in0 = IRNode(id="in0", op_type="Input", shape_metadata=(2, 3, 4, 5))
    mrt_node = IRNode(
        id="split_node",
        op_type="Add",
        inputs=["in0", "in0"],
        shape_metadata=(2, 3, 4, 5),
        attributes={"num_outputs": 2},
    )
    graph.nodes = {"in0": in0, "split_node": mrt_node}
    gen = WebGLCodeGenerator(graph)
    gen.config.js_orchestration.pop("node_pass", None)
    out = gen.generate()
    assert "stride_0" in out
    assert "stride_1" in out
    assert "stride_2" in out
    assert "stride_3" in out
    assert "gl.drawBuffers([gl.COLOR_ATTACHMENT0, gl.COLOR_ATTACHMENT1]);" in out
    assert "texOut_split_node_0" in out
    assert "texOut_split_node_1" in out


def test_webgl_generator_conv2d_shape_branches() -> None:
    """Test WebGLCodeGenerator with 4D Conv2D and non-int symbolic dimensions."""
    from ml_switcheroo_ir import LogicalNode

    graph = IRGraph(name="webgl_conv")
    node_in = LogicalNode(id="in_0", op_type="Input", shape_metadata=(1, 3, 32, 32))
    node_w = LogicalNode(id="w_0", op_type="Input", shape_metadata=(16, 3, 3, 3))
    node_conv = LogicalNode(id="conv_0", op_type="Conv2D", inputs=["in_0", "w_0"], shape_metadata=(1, 16, 30, 30))
    graph.nodes = {"in_0": node_in, "w_0": node_w, "conv_0": node_conv}
    graph.inputs = ["in_0", "w_0"]
    graph.outputs = ["conv_0"]
    gen = WebGLCodeGenerator(graph)
    assert gen.generate() is not None

    graph_sym = IRGraph(name="webgl_sym")
    node_sym = LogicalNode(id="sym_add", op_type="Add", inputs=[], shape_metadata=("batch", 4, 8))
    graph_sym.nodes = {"sym_add": node_sym}
    graph_sym.inputs = []
    graph_sym.outputs = ["sym_add"]
    gen_sym = WebGLCodeGenerator(graph_sym)
    assert gen_sym.generate() is not None


def test_webgl_new_operator_templates() -> None:
    """Verify code emission for newly added linear algebra, activation, and control operations."""
    ops_to_test = [
        ("bmm", "BatchedMatMul", ["in_a", "in_b"], (2, 16, 16)),
        ("dot_node", "Dot", ["in_a", "in_b"], (1,)),
        ("outer_node", "Outer", ["in_a", "in_b"], (16, 16)),
        ("prelu_node", "PReLU", ["in_a", "in_b"], (16, 16)),
        ("trelu_node", "ThresholdedReLU", ["in_a"], (16, 16)),
        ("hsig_node", "HardSigmoid", ["in_a"], (16, 16)),
        ("l1p_node", "Log1p", ["in_a"], (16, 16)),
        ("expm1_node", "Expm1", ["in_a"], (16, 16)),
        ("scan_node", "Scan", ["in_a"], (16, 16)),
        ("cond_node", "Cond", ["in_a", "in_b", "in_c"], (16, 16)),
    ]

    for nid, op, inps, shape in ops_to_test:
        graph = IRGraph(name=f"test_{op}")
        graph.nodes = {
            "in_a": IRNode(id="in_a", op_type="Input", shape_metadata=(16, 16)),
            "in_b": IRNode(id="in_b", op_type="Input", shape_metadata=(16, 16)),
            "in_c": IRNode(id="in_c", op_type="Input", shape_metadata=(16, 16)),
            nid: IRNode(id=nid, op_type=op, inputs=inps, shape_metadata=shape),
        }
        graph.outputs = [nid]
        gen = WebGLCodeGenerator(graph)
        out = gen.generate()
        assert f"shader_{nid}" in out
        bundle = gen._compile_aot_impl(graph)
        assert nid in bundle["shaders"]


def test_webgl_aot_runner_execution() -> None:
    """Test ahead-of-time WebGL runner execution with single, multiple, and empty outputs."""
    import numpy as np

    # 1. Single output
    graph = IRGraph(name="test_webgl_runner_single")
    n_in = IRNode(id="x", op_type="Input", shape_metadata=(2, 2))
    n_add = IRNode(id="y", op_type="Add", inputs=["x", "x"], shape_metadata=(2, 2))
    graph.nodes = {"x": n_in, "y": n_add}
    graph.inputs = ["x"]
    graph.outputs = ["y"]
    gen = WebGLCodeGenerator(graph)
    artifact = gen.compile_aot(graph)
    inp = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    res_single = artifact(inp)
    np.testing.assert_allclose(res_single, inp + inp)
    # Extra argument branch coverage
    res_extra = artifact(inp, "extra_arg")
    np.testing.assert_allclose(res_extra, inp + inp)

    # 2. Multiple outputs
    n_sub = IRNode(id="z", op_type="Sub", inputs=["y", "x"], shape_metadata=(2, 2))
    graph.nodes["z"] = n_sub
    graph.outputs = ["y", "z"]
    # Bypass cache with kwargs or fresh generator
    gen_multi = WebGLCodeGenerator(graph)
    artifact_multi = gen_multi.compile_aot(graph, opt_level=1)
    res_multi = artifact_multi(inp)
    assert isinstance(res_multi, tuple) and len(res_multi) == 2
    np.testing.assert_allclose(res_multi[0], inp + inp)
    np.testing.assert_allclose(res_multi[1], inp)

    # 3. Empty outputs
    graph_empty = IRGraph(name="test_webgl_runner_empty")
    graph_empty.nodes = {"x": n_in}
    graph_empty.inputs = ["x"]
    graph_empty.outputs = []
    gen_empty = WebGLCodeGenerator(graph_empty)
    artifact_empty = gen_empty.compile_aot(graph_empty)
    res_empty = artifact_empty(inp)
    assert isinstance(res_empty, dict)
