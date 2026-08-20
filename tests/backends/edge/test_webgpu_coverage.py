from unittest.mock import patch

from ml_switcheroo_compiler.backends.edge.webgpu import WebGPUCodeGenerator
from ml_switcheroo_compiler.ir.core import IRGraph, IRNode


def test_webgpu_tiling_and_webrtc_and_dynamic():
    import copy

    from ml_switcheroo_compiler.ops.registry import _YAML_REGISTRY as OPS_REGISTRY

    saved_registry = copy.deepcopy(OPS_REGISTRY)
    try:
        OPS_REGISTRY["MatMulTiledDummy"] = {"variants": {"edge_wgsl": {"template": "tiled_matmul"}}}

        g = IRGraph()
        # Graph attributes for dynamic schema
        g.attributes = {"dynamic_memory_schema": {"dynamic_offsets": [{"var_name": "B", "symbolic_math": "B_dim * 16"}]}}

        # MatMul with tiling (using our dummy op)
        n_matmul = IRNode("matmul_tiled", "MatMulTiledDummy", inputs=["in1", "in2"], attributes={"tiling": True})
        n_matmul.shape_metadata = (16, 16)

        # Conv2D with tiling
        n_conv = IRNode("conv_tiled", "Conv2D", inputs=["in_img", "in_w"], attributes={"tiling": True, "stride": (1, 1)})
        n_conv.shape_metadata = (1, 16, 16, 16)

        # Conv2D with bad shapes to hit < 4 branch
        n_conv_bad = IRNode("conv_bad", "Conv2D", inputs=["in_img2", "in_w2"], attributes={"stride": 1})
        n_conv_bad.shape_metadata = (16, 16)

        # WebRTC ops
        n_allreduce = IRNode("allreduce", "AllReduce", inputs=["in1"])
        n_allgather = IRNode("allgather", "AllGather", inputs=["in1"])
        n_alltoall = IRNode("alltoall", "AllToAll", inputs=["in1"])

        in1 = IRNode("in1", "Input")
        in1.shape_metadata = (16, 16)
        in2 = IRNode("in2", "Input")
        in2.shape_metadata = (16, 16)
        in_img = IRNode("in_img", "Input")
        in_img.shape_metadata = (1, 3, 16, 16)
        in_w = IRNode("in_w", "Input")
        in_w.shape_metadata = (16, 3, 3, 3)
        in_img2 = IRNode("in_img2", "Input")
        in_img2.shape_metadata = (16, 16)
        in_w2 = IRNode("in_w2", "Input")
        in_w2.shape_metadata = (3, 3)

        g.nodes = {"in1": in1, "in2": in2, "in_img": in_img, "in_w": in_w, "in_img2": in_img2, "in_w2": in_w2, "matmul_tiled": n_matmul, "conv_tiled": n_conv, "conv_bad": n_conv_bad, "allreduce": n_allreduce, "allgather": n_allgather, "alltoall": n_alltoall}
        g.inputs = ["in1", "in2", "in_img", "in_w", "in_img2", "in_w2"]
        g.outputs = ["matmul_tiled", "conv_tiled", "conv_bad", "allreduce", "allgather", "alltoall"]

        gen = WebGPUCodeGenerator(g)
        gen.sorted_nodes = [in1, in2, in_img, in_w, in_img2, in_w2, n_matmul, n_conv, n_conv_bad, n_allreduce, n_allgather, n_alltoall]

        code = gen.generate()
        assert code is not None

    finally:
        OPS_REGISTRY.clear()
        OPS_REGISTRY.update(saved_registry)


def test_webgpu_webrtc_missing():
    from ml_switcheroo_compiler.backends.edge.webgpu_webrtc import emit_webrtc_init, emit_webrtc_op

    with patch("os.path.exists", return_value=False):
        assert emit_webrtc_init() == ""
        assert emit_webrtc_op("AllReduce", "buf", "id") == ""

    # Also test unknown op type
    assert emit_webrtc_op("Unknown", "buf", "id") == ""


def test_webgpu_dynamic_resize_wasm_fallback():
    import ml_switcheroo_compiler.backends.edge.webgpu as webgpu_mod
    from ml_switcheroo_compiler.backends.edge.webgpu import WebGPUCodeGenerator
    from ml_switcheroo_compiler.ir.core import IRGraph, IRNode

    g = IRGraph()
    # Graph attributes for dynamic schema
    g.attributes = {"dynamic_memory_schema": {"dynamic_offsets": [{"var_name": "B", "symbolic_math": "B_dim * 16"}]}}

    in1 = IRNode("in1", "Input")
    in1.shape_metadata = (16, 16)

    g.nodes = {"in1": in1}
    g.inputs = ["in1"]
    g.outputs = ["in1"]

    gen = WebGPUCodeGenerator(g)
    gen.sorted_nodes = [in1]

    with patch.object(webgpu_mod, "__file__", "wasm.py"):
        code = gen.generate()
        assert code is not None
