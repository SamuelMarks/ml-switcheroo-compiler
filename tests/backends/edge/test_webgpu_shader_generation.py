from unittest.mock import patch

from ml_switcheroo_compiler.backends.edge.webgpu import WebGPUCodeGenerator
from ml_switcheroo_compiler.ir.core import IRGraph, IRNode
from ml_switcheroo_compiler.ir.shape_system import SymInt


def test_webgpu_missing_coverage():
    """Test line coverage for WebGPU."""
    g = IRGraph()
    n_linear = IRNode(id="n_linear", op_type="Linear", inputs=["dummy"], shape_metadata=[2, 2])
    n_attn = IRNode(id="n_attn", op_type="Attention", inputs=["dummy"], shape_metadata=[2, 2])
    n_maxpool = IRNode(id="n_maxpool", op_type="MaxPool", inputs=["dummy"], shape_metadata=[1, 2, 2, 1])
    n_layernorm = IRNode(id="n_layernorm", op_type="LayerNorm", inputs=["dummy"], shape_metadata=[2, 2])
    n_trig = IRNode(id="n_trig", op_type="Trig", inputs=["dummy"], shape_metadata=[2, 2])
    n_reducesum = IRNode(id="n_reducesum", op_type="ReduceSum", inputs=["dummy"], shape_metadata=[2, 2])
    n_dummy = IRNode(id="dummy", op_type="Input", inputs=[], shape_metadata=[2, 2])

    for n in [n_linear, n_attn, n_maxpool, n_layernorm, n_trig, n_reducesum, n_dummy]:
        g.nodes[n.id] = n

    g.inputs = ["dummy"]
    g.outputs = ["n_linear"]

    gen = WebGPUCodeGenerator(g)
    gen.generate()

    # Also test webrtc ops mock coverage
    with patch("ml_switcheroo_compiler.backends.edge.webgpu_webrtc.emit_webrtc_init", return_value="init_line1\ninit_line2"):
        with patch("ml_switcheroo_compiler.backends.edge.webgpu_webrtc.emit_webrtc_op", return_value="op_line1\nop_line2"):
            # Set up a distributed graph
            g2 = IRGraph()
            n_allreduce = IRNode(id="ar", op_type="AllReduce", inputs=["dummy"])
            n_dummy2 = IRNode(id="dummy", op_type="Input")
            g2.nodes["ar"] = n_allreduce
            g2.nodes["dummy"] = n_dummy2
            g2.inputs = ["dummy"]
            g2.outputs = ["ar"]
            g2.attributes = {"distributed": True}
            gen2 = WebGPUCodeGenerator(g2)
            gen2.generate()


def test_webgpu_attention_shared_memory_reduction():
    """Test Multi-Head Attention WGSL compute shader with workgroup shared memory reduction."""
    g = IRGraph()
    q = IRNode(id="q", op_type="Input", inputs=[], shape_metadata=[8, 64])
    k = IRNode(id="k", op_type="Input", inputs=[], shape_metadata=[8, 64])
    v = IRNode(id="v", op_type="Input", inputs=[], shape_metadata=[8, 64])
    attn = IRNode(id="attn_node", op_type="Attention", inputs=["q", "k", "v"], shape_metadata=[8, 64])

    for n in [q, k, v, attn]:
        g.nodes[n.id] = n
    g.inputs = ["q", "k", "v"]
    g.outputs = ["attn_node"]

    gen = WebGPUCodeGenerator(g)
    shader_bundle = gen.generate()

    assert "var<workgroup> s_weights: array<f32, 64>;" in shader_bundle
    assert "var<workgroup> s_reduce: array<f32, 64>;" in shader_bundle
    assert "workgroupBarrier();" in shader_bundle
    assert "scale" in shader_bundle


def test_webgpu_pooling_and_gemm_visitors():
    """Test 2D Pooling and GEMM visitor methods."""
    g = IRGraph()
    gen = WebGPUCodeGenerator(g)

    # Test MaxPool2D and AvgPool2D
    p_node = IRNode(id="p1", op_type="MaxPool2D", inputs=["x"], shape_metadata=[1, 1, 4, 4])
    wgsl_lines, dx, dy, dz = gen.visit_MaxPool2D(p_node, ["x"], shape=[1, 1, 4, 4], nelem=16, clean_id="p1")
    assert len(wgsl_lines) > 0

    wgsl_lines, _, _, _ = gen.visit_AvgPool2D(p_node, ["x"], shape=[1, 1, 4, 4], nelem=16, clean_id="p1")
    assert len(wgsl_lines) > 0

    wgsl_lines, _, _, _ = gen.visit_AvgPool(p_node, ["x"], shape=[1, 1, 4, 4], nelem=16, clean_id="p1")
    assert len(wgsl_lines) > 0

    # Test MatMul and BatchMatMul
    m_node = IRNode(id="m1", op_type="MatMul", inputs=["a", "b"], shape_metadata=[4, 4])
    wgsl_m, _, _, _ = gen.visit_MatMul(m_node, ["a", "b"], shape=[4, 4], nelem=16, clean_id="m1")
    assert len(wgsl_m) > 0

    wgsl_bm, _, _, _ = gen.visit_BatchMatMul(m_node, ["a", "b"], shape=[4, 4], nelem=16, clean_id="m1")
    assert len(wgsl_bm) > 0


def test_webgpu_shape_telemetry_symbolic():
    """Test WebGPU shape telemetry resolving SymInt dimensions."""
    g = IRGraph()
    sym_b = SymInt("B")
    n_in = IRNode(id="in0", op_type="Input", inputs=[], shape_metadata=[sym_b, 64])
    n_out = IRNode(id="out0", op_type="Relu", inputs=["in0"], shape_metadata=[sym_b, 64])
    g.nodes["in0"] = n_in
    g.nodes["out0"] = n_out
    g.inputs = ["in0"]
    g.outputs = ["out0"]

    gen = WebGPUCodeGenerator(g)
    gen.apply_shape_telemetry({"shapes": {"in0": [4, 64]}})

    assert n_in.shape_metadata == (4, 64)
    assert n_out.shape_metadata == (4, 64)
