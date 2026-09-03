def test_webgpu_missing_coverage():
    """Test line coverage for WebGPU."""
    from unittest.mock import patch

    from ml_switcheroo_compiler.backends.edge.webgpu import WebGPUCodeGenerator
    from ml_switcheroo_compiler.ir.core import IRGraph, IRNode

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
    with patch("ml_switcheroo_compiler.backends.edge.webgpu_webrtc.emit_webrtc_init", return_value="init_line1\\ninit_line2"):
        with patch("ml_switcheroo_compiler.backends.edge.webgpu_webrtc.emit_webrtc_op", return_value="op_line1\\nop_line2"):
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
