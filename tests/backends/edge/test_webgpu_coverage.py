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


def test_webgpu_webrtc_empty_coverage():
    import copy
    from unittest.mock import patch

    import yaml

    from ml_switcheroo_compiler.backends.edge.webgpu import WebGPUCodeGenerator
    from ml_switcheroo_compiler.ir.core import IRGraph, IRNode

    with open("src/ml_switcheroo_compiler/backends/edge/memory_schemas.yaml") as f:
        real_data = yaml.safe_load(f)

    mocked_data = copy.deepcopy(real_data)
    mocked_data["schemas"]["js_orchestration_templates"]["dynamic_resize"] = ""

    with patch("yaml.safe_load", return_value=mocked_data), patch("ml_switcheroo_compiler.backends.edge.webgpu_webrtc.emit_webrtc_init", return_value=""), patch("ml_switcheroo_compiler.backends.edge.webgpu_webrtc.emit_webrtc_op", return_value=""):
        g = IRGraph()
        g.attributes = {"dynamic_memory_schema": {"dynamic_offsets": [{"var_name": "B", "symbolic_math": "B_dim * 16"}]}}
        n_allreduce = IRNode("allreduce", "AllReduce", inputs=["in1"])
        in1 = IRNode("in1", "Input", attributes={"buffer_id": 1})
        g.nodes = {"in1": in1, "allreduce": n_allreduce}
        g.inputs = ["in1"]
        g.outputs = ["allreduce"]
        gen = WebGPUCodeGenerator(g)
        gen.sorted_nodes = [in1, n_allreduce]
        gen.generate()


def test_webgpu_force_get_wgsl_for_op_coverage():
    import copy

    from ml_switcheroo_compiler.backends.edge.webgpu import WebGPUCodeGenerator
    from ml_switcheroo_compiler.ir.core import IRGraph, IRNode
    from ml_switcheroo_compiler.ops.registry import _YAML_REGISTRY as OPS_REGISTRY

    saved_registry = copy.deepcopy(OPS_REGISTRY)
    try:
        g = IRGraph()
        in0 = IRNode("in0", "Input")
        in0.shape_metadata = (1, 16, 16, 16)
        in1 = IRNode("in1", "Input")
        in1.shape_metadata = (1, 16, 16, 16)
        in2 = IRNode("in2", "Input")
        in3 = IRNode("in3", "Input")

        n_dummy = IRNode("dummy", "DummyConv", inputs=["in0", "in1", "in2", "in3"], attributes={"window_size": (2, 2), "stride": (1, 1), "TILE_M": 16, "TILE_N": 16, "TILE_K": 16})
        n_dummy.shape_metadata = (1, 16, 16, 16)

        OPS_REGISTRY["DummyConv"] = {"variants": {"edge_wgsl": {"template": "im2col_conv2d", "expr": "test_expr"}}}
        gen = WebGPUCodeGenerator(g)
        gen.sorted_nodes = [in0, in1, in2, in3, n_dummy]
        gen._get_wgsl_for_op(n_dummy, [1, 16, 16, 16], 16 * 16 * 16, "dummy")

        n_dummy2 = IRNode("dummy2", "DummyConv2", inputs=["in0", "in1"], attributes={"window_size": 2, "stride": 2})
        n_dummy2.shape_metadata = (1, 16, 16, 16)
        OPS_REGISTRY["DummyConv2"] = {"variants": {"edge_wgsl": {"template": "conv2d", "expr": "test_expr"}}}
        gen._get_wgsl_for_op(n_dummy2, [1, 16, 16, 16], 16 * 16 * 16, "dummy2")
    finally:
        OPS_REGISTRY.clear()
        OPS_REGISTRY.update(saved_registry)


def test_webgpu_while_loop_cond_reduce_scatter_normal():
    from ml_switcheroo_compiler.backends.edge.webgpu import WebGPUCodeGenerator
    from ml_switcheroo_compiler.ir.core import IRGraph, IRNode

    body_graph = IRGraph()
    b_in = IRNode("b_in", "Input")
    b_out = IRNode("b_out", "Add", inputs=["b_in", "b_in"])
    body_graph.nodes = {"b_in": b_in, "b_out": b_out}
    body_graph.inputs = ["b_in"]
    body_graph.outputs = ["b_out"]

    cond_graph = IRGraph()
    c_in = IRNode("c_in", "Input")
    c_out = IRNode("c_out", "Less", inputs=["c_in", "c_in"])
    cond_graph.nodes = {"c_in": c_in, "c_out": c_out}
    cond_graph.inputs = ["c_in"]
    cond_graph.outputs = ["c_out"]

    g = IRGraph()
    n_while = IRNode("while", "WhileLoop", inputs=["in1"], attributes={"body": body_graph, "cond": cond_graph})
    n_while.shape_metadata = (1,)

    then_graph = IRGraph()
    t_in = IRNode("t_in", "Input")
    t_out = IRNode("t_out", "Exp", inputs=["t_in"])
    then_graph.nodes = {"t_in": t_in, "t_out": t_out}
    then_graph.inputs = ["t_in"]
    then_graph.outputs = ["t_out"]

    n_cond = IRNode("cond", "Cond", inputs=["cond_in", "t_in", "e_in"], attributes={"then_branch": then_graph})
    n_cond.shape_metadata = (1,)

    n_reducescatter = IRNode("reducescatter", "ReduceScatter", inputs=["in1"])
    n_reducescatter.shape_metadata = (1,)

    in1 = IRNode("in1", "Input", attributes={"buffer_id": 1})
    cond_in = IRNode("cond_in", "Input")
    in2 = IRNode("t_in", "Input")
    in3 = IRNode("e_in", "Input")

    g.nodes = {"in1": in1, "cond_in": cond_in, "t_in": in2, "e_in": in3, "while": n_while, "cond": n_cond, "reducescatter": n_reducescatter}
    g.inputs = ["in1", "cond_in", "t_in", "e_in"]
    g.outputs = ["while", "cond", "reducescatter"]

    gen = WebGPUCodeGenerator(g)
    gen.sorted_nodes = [in1, cond_in, in2, in3, n_while, n_cond, n_reducescatter]
    gen.generate()
