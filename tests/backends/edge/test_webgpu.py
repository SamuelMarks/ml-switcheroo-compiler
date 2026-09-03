from unittest.mock import patch

from ml_switcheroo_compiler.backends.edge.webgpu import WebGPUCodeGenerator
from ml_switcheroo_compiler.ir.core import IRGraph, IRNode


def test_webgpu_instantiation():
    graph = IRGraph()
    generator = WebGPUCodeGenerator(graph, [])
    assert generator.graph == graph


def test_webgpu_map_type():
    graph = IRGraph()
    generator = WebGPUCodeGenerator(graph, [])
    assert generator._map_type("float32") == "f32"
    assert generator._map_type("float64") == "f32"
    assert generator._map_type("int32") == "i32"
    assert generator._map_type("bool") == "bool"
    assert generator._map_type("unknown") == "f32"


def test_webgpu_shape_and_strides():
    graph = IRGraph()
    generator = WebGPUCodeGenerator(graph, [])

    node1 = IRNode("Input", "in_1", [])
    node1.shape_metadata = [2, 3]
    shape, strides = generator._get_shape_and_strides(node1)
    assert shape == [2, 3]
    assert strides == [3, 1]

    node2 = IRNode("Input", "in_2", [])
    node2.shape_metadata = 5
    shape, strides = generator._get_shape_and_strides(node2)
    assert shape == [5]
    assert strides == [1]

    node3 = IRNode("Input", "in_3", [])
    node3.shape_metadata = None
    shape, strides = generator._get_shape_and_strides(node3)
    assert shape == []
    assert strides == []

    node4 = IRNode("Input", "in_4", [])
    node4.shape_metadata = []
    shape, strides = generator._get_shape_and_strides(node4)
    assert shape == []
    assert strides == []


def test_webgpu_num_elements():
    graph = IRGraph()
    generator = WebGPUCodeGenerator(graph, [])
    assert generator._num_elements([2, 3]) == 6
    assert generator._num_elements([]) == 1


def test_webgpu_generic_visit():
    graph = IRGraph()
    generator = WebGPUCodeGenerator(graph, [])
    node = IRNode("Input", "in_1", [])
    node.id = "in_1"
    assert generator.generic_visit(node, []) == "in_1"


def test_webgpu_gen_offset():
    graph = IRGraph()
    generator = WebGPUCodeGenerator(graph, [])

    nodes = generator._gen_offset_computation("idx", [2, 3], [3, 1], "out")
    assert len(nodes) > 0

    nodes_empty = generator._gen_offset_computation("idx", [], [], "out")
    assert len(nodes_empty) == 1


def test_webgpu_generate():
    graph = IRGraph()
    node = IRNode("Add", "add_1", ["in_1", "in_2"])
    node.shape_metadata = [2, 2]
    node.id = "add_1"

    in1 = IRNode("Input", "in_1", [])
    in1.shape_metadata = [2, 2]
    in1.id = "in_1"
    in2 = IRNode("Input", "in_2", [])
    in2.shape_metadata = [2, 2]
    in2.id = "in_2"

    graph.nodes = {"in_1": in1, "in_2": in2, "add_1": node}
    graph.sorted_nodes = [in1, in2, node]
    graph.inputs = ["in_1", "in_2"]
    graph.outputs = ["add_1"]

    generator = WebGPUCodeGenerator(graph, [])

    # Just basic generation
    out = generator.generate()
    assert "shaderCode" in out


def test_webgpu_webrtc_methods():
    graph = IRGraph()
    generator = WebGPUCodeGenerator(graph, [])
    node = IRNode("AllReduce", "allr_1", [])
    node.id = "allr_1"
    assert generator.visit_AllReduce(node, [])[0][0].startswith("//")
    assert generator.visit_AllGather(node, [])[0][0].startswith("//")
    assert generator.visit_AllToAll(node, [])[0][0].startswith("//")
    assert generator.visit_ReduceScatter(node, [])[0][0].startswith("//")


def test_webgpu_wgsl_for_op():
    graph = IRGraph()
    generator = WebGPUCodeGenerator(graph, [])
    node = IRNode("Add", "add_1", ["in_1", "in_2"])
    node.shape_metadata = [2, 2]
    node.id = "add_1"

    # Mock some inputs
    in1 = IRNode("Input", "in_1", [])
    in1.id = "in_1"
    in2 = IRNode("Input", "in_2", [])
    in2.id = "in_2"
    generator.sorted_nodes = [in1, in2, node]

    with patch.dict("ml_switcheroo_compiler.ops.registry._YAML_REGISTRY", {"Add": {"variants": {"edge_wgsl": {"template": "elementwise", "expr": "buf_in0_f32[in0_offset] + buf_in1_f32[in1_offset]"}}}}):
        with patch("ml_switcheroo_compiler.backends.edge.wgsl.wgsl_provider.get_wgsl_template", return_value={"body": "fn body({clean_id}) {{}}"}):
            wgsl, x, y, z = generator._get_wgsl_for_op(node, [2, 2], 4, "add_1")
            assert len(wgsl) > 0


def test_webgpu_wgsl_for_op_branches():
    graph = IRGraph()
    generator = WebGPUCodeGenerator(graph, [])

    # Test missing mapping fallback
    node = IRNode("UnknownOp", "u_1", ["in_1"])
    node.shape_metadata = [2, 2, 2, 2]
    node.id = "u_1"

    in1 = IRNode("Input", "in_1", [])
    in1.shape_metadata = [2, 2, 2, 2]
    in1.id = "in_1"

    generator.sorted_nodes = [in1, node]

    with patch("ml_switcheroo_compiler.backends.edge.wgsl.wgsl_provider.get_wgsl_template", return_value={"body": "fn body() {{}}"}):
        wgsl, x, y, z = generator._get_wgsl_for_op(node, [2, 2, 2, 2], 16, "u_1")
        assert len(wgsl) > 0

    # Test matmul
    node2 = IRNode("MatMul", "m_1", ["in_1", "in_2"])
    node2.shape_metadata = [2, 2]
    node2.id = "m_1"
    in2 = IRNode("Input", "in_2", [])
    in2.shape_metadata = [2, 2]
    in2.id = "in_2"
    generator.sorted_nodes = [in1, in2, node2]

    with patch.dict("ml_switcheroo_compiler.ops.registry._YAML_REGISTRY", {"MatMul": {"variants": {"edge_wgsl": {"template": "matmul"}}}}):
        with patch("ml_switcheroo_compiler.backends.edge.wgsl.wgsl_provider.get_wgsl_template", return_value={"body": "fn body() {{}}"}):
            wgsl, x, y, z = generator._get_wgsl_for_op(node2, [2, 2], 4, "m_1")

    # Test tiled_matmul
    with patch.dict("ml_switcheroo_compiler.ops.registry._YAML_REGISTRY", {"MatMul": {"variants": {"edge_wgsl": {"template": "tiled_matmul"}}}}):
        with patch("ml_switcheroo_compiler.backends.edge.wgsl.wgsl_provider.get_wgsl_template", return_value={"body": "fn body() {{}}"}):
            wgsl, x, y, z = generator._get_wgsl_for_op(node2, [2, 2], 4, "m_1")

    # Test conv2d
    with patch.dict("ml_switcheroo_compiler.ops.registry._YAML_REGISTRY", {"MatMul": {"variants": {"edge_wgsl": {"template": "conv2d"}}}}):
        with patch("ml_switcheroo_compiler.backends.edge.wgsl.wgsl_provider.get_wgsl_template", return_value={"body": "fn body() {{}}"}):
            wgsl, x, y, z = generator._get_wgsl_for_op(node2, [2, 2], 4, "m_1")


def test_webgpu_wgsl_for_op_attributes():
    graph = IRGraph()
    generator = WebGPUCodeGenerator(graph, [])

    node = IRNode("MaxPool2D", "max_1", ["in_1"])
    node.shape_metadata = [2, 2, 2, 2]
    node.id = "max_1"

    # window_size as int
    node.attributes = {"window_size": 2, "stride": 2}

    in1 = IRNode("Input", "in_1", [])
    in1.shape_metadata = [2, 2, 2, 2]
    in1.id = "in_1"

    generator.sorted_nodes = [in1, node]

    with patch.dict("ml_switcheroo_compiler.ops.registry._YAML_REGISTRY", {"MaxPool2D": {"variants": {"edge_wgsl": {"template": "MaxPool2D"}}}}):
        with patch("ml_switcheroo_compiler.backends.edge.wgsl.wgsl_provider.get_wgsl_template", return_value={"body": "fn body() {{}}"}):
            wgsl, x, y, z = generator._get_wgsl_for_op(node, [2, 2, 2, 2], 16, "max_1")

    # window_size as list length 1
    node.attributes = {"window_size": [2], "stride": [2]}
    with patch.dict("ml_switcheroo_compiler.ops.registry._YAML_REGISTRY", {"MaxPool2D": {"variants": {"edge_wgsl": {"template": "MaxPool2D"}}}}):
        with patch("ml_switcheroo_compiler.backends.edge.wgsl.wgsl_provider.get_wgsl_template", return_value={"body": "fn body() {{}}"}):
            wgsl, x, y, z = generator._get_wgsl_for_op(node, [2, 2, 2, 2], 16, "max_1")


def test_webgpu_wgsl_for_op_attributes_2():
    graph = IRGraph()
    generator = WebGPUCodeGenerator(graph, [])

    node = IRNode("Add", "add_1", ["in_1", "in_2", "in_3"])
    node.shape_metadata = [2, 2, 2, 2]
    node.id = "add_1"

    in1 = IRNode("Input", "in_1", [])
    in1.shape_metadata = [2, 2, 2, 2]
    in1.id = "in_1"

    in2 = IRNode("Input", "in_2", [])
    in2.shape_metadata = [2, 2, 2, 2]
    in2.id = "in_2"

    in3 = IRNode("Input", "in_3", [])
    in3.shape_metadata = [2, 2, 2, 2]
    in3.id = "in_3"

    generator.sorted_nodes = [in1, in2, in3, node]

    with patch.dict("ml_switcheroo_compiler.ops.registry._YAML_REGISTRY", {"Add": {"variants": {"edge_wgsl": {"template": "elementwise", "global_code": "fn helper() {{}}"}}}}):
        with patch("ml_switcheroo_compiler.backends.edge.wgsl.wgsl_provider.get_wgsl_template", return_value={"body": "fn body() {{}}"}):
            wgsl, x, y, z = generator._get_wgsl_for_op(node, [2, 2, 2, 2], 16, "add_1")


def test_webgpu_wgsl_for_op_attributes_missing_body():
    graph = IRGraph()
    generator = WebGPUCodeGenerator(graph, [])

    node = IRNode("Add", "add_1", ["in_1", "in_2", "in_3"])
    node.shape_metadata = [2, 2, 2, 2]
    node.id = "add_1"

    generator.sorted_nodes = [node]

    with patch.dict("ml_switcheroo_compiler.ops.registry._YAML_REGISTRY", {"Add": {"variants": {"edge_wgsl": {"template": "elementwise", "global_code": "fn helper() {{}}"}}}}):
        with patch("ml_switcheroo_compiler.backends.edge.wgsl.wgsl_provider.get_wgsl_template", return_value={"global_code": "fn helper() {{}}"}):
            wgsl, x, y, z = generator._get_wgsl_for_op(node, [2, 2, 2, 2], 16, "add_1")


def test_webgpu_visit_methods():
    graph = IRGraph()
    generator = WebGPUCodeGenerator(graph, [])
    node = IRNode("If", "if_1", [])
    node.id = "if_1"

    node.attributes = {"then_branch": IRGraph(), "else_branch": IRGraph()}

    node_while = IRNode("WhileLoop", "wl_1", [])
    node_while.id = "wl_1"
    node_while.attributes = {"body": IRGraph(), "cond": IRGraph()}
    generator.visit_WhileLoop(node_while, [])

    node_cond = IRNode("Cond", "cond_1", [])
    node_cond.id = "cond_1"
    node_cond.attributes = {"then_branch": IRGraph(), "else_branch": IRGraph()}
    generator.visit_Cond(node_cond, [])

    node_scan = IRNode("Scan", "scan_1", [])
    node_scan.id = "scan_1"
    generator.visit_Scan(node_scan, [])


def test_webgpu_wgsl_for_op_attributes_cond_while_scan():
    graph = IRGraph()
    generator = WebGPUCodeGenerator(graph, [])

    # Test lowering subgraphs in Cond, WhileLoop, Scan
    # The actual methods handle node attributes

    node_cond = IRNode("Cond", "cond_1", [])
    node_cond.id = "cond_1"
    sub_node = IRNode("Add", "add_2", [])
    sub_node.id = "add_2"
    sub_node.op_type = "Add"
    sub_graph = IRGraph()
    sub_graph.nodes = {"add_2": sub_node}
    sub_graph.sorted_nodes = [sub_node]

    node_cond.attributes = {"then_branch": sub_graph, "else_branch": sub_graph}

    with patch("ml_switcheroo_compiler.backends.edge.wgsl.wgsl_provider.get_wgsl_template", return_value={"body": "fn body() {{}}"}):
        with patch.dict("ml_switcheroo_compiler.ops.registry._YAML_REGISTRY", {"Add": {"variants": {"edge_wgsl": {"expr": "buf_in0_f32[in0_offset] + buf_in1_f32[in1_offset]"}}}}):
            generator.visit_Cond(node_cond, [])

    node_while = IRNode("WhileLoop", "wl_1", [])
    node_while.id = "wl_1"
    node_while.attributes = {"body": sub_graph, "cond": sub_graph}
    with patch("ml_switcheroo_compiler.backends.edge.wgsl.wgsl_provider.get_wgsl_template", return_value={"body": "fn body() {{}}"}):
        with patch.dict("ml_switcheroo_compiler.ops.registry._YAML_REGISTRY", {"Add": {"variants": {"edge_wgsl": {"expr": "buf_in0_f32[in0_offset] + buf_in1_f32[in1_offset]"}}}}):
            generator.visit_WhileLoop(node_while, [])


def test_webgpu_generate_method_details():
    graph = IRGraph()
    generator = WebGPUCodeGenerator(graph, [])

    # We need to test the logic inside generate that uses arenas and WebRTC ops
    # Also dynamic offsets

    node1 = IRNode("Input", "in_1", [])
    node1.id = "in_1"
    node1.attributes = {"buffer_id": "arena1", "buffer_offset": 0}

    node2 = IRNode("AllReduce", "allr_1", ["in_1"])
    node2.id = "allr_1"
    node2.attributes = {"buffer_id": "arena1", "buffer_offset": 4}

    graph.sorted_nodes = [node1, node2]
    graph.outputs = ["allr_1"]

    graph.attributes = {"dynamic_memory_schema": {"dynamic_offsets": [{"var_name": "x", "symbolic_math": "y + 1"}]}}

    # Needs to mock open for the memory_schemas yaml load
    # or mock os.path.exists
    with patch("os.path.exists", return_value=False):
        # Without schema file, it ignores dynamic offset resizing logic
        out = generator.generate()
        pass


def test_webgpu_generate_visit_Conv2D():
    graph = IRGraph()
    generator = WebGPUCodeGenerator(graph, [])

    node = IRNode("Conv2D", "conv_1", ["in1", "in2"])
    node.id = "conv_1"

    in1 = IRNode("Input", "in1", [])
    in1.shape_metadata = [1, 3, 224, 224]
    in1.id = "in1"

    in2 = IRNode("Input", "in2", [])
    in2.shape_metadata = [64, 3, 3, 3]
    in2.id = "in2"

    generator.sorted_nodes = [in1, in2, node]

    with patch("ml_switcheroo_compiler.backends.edge.wgsl.wgsl_provider.get_wgsl_template", return_value={"body": "fn conv() {{}}"}):
        res = generator.visit_Conv2D(node, ["in1", "in2"], shape=[1, 64, 222, 222], nelem=1, clean_id="conv_1")
        assert res is not None


def test_webgpu_wgsl_for_op_attributes_padding():
    graph = IRGraph()
    generator = WebGPUCodeGenerator(graph, [])

    node = IRNode("AvgPool2D", "avg_1", ["in_1"])
    node.shape_metadata = [2, 2, 2, 2]
    node.id = "avg_1"

    # Check what happens if shape has missing dims
    in1 = IRNode("Input", "in_1", [])
    in1.shape_metadata = [2]
    in1.id = "in_1"

    generator.sorted_nodes = [in1, node]

    with patch.dict("ml_switcheroo_compiler.ops.registry._YAML_REGISTRY", {"AvgPool2D": {"variants": {"edge_wgsl": {"template": "AvgPool2D"}}}}):
        with patch("ml_switcheroo_compiler.backends.edge.wgsl.wgsl_provider.get_wgsl_template", return_value={"body": "fn body() {{}}"}):
            wgsl, x, y, z = generator._get_wgsl_for_op(node, [2], 2, "avg_1")


def test_webgpu_wgsl_for_op_attributes_no_shape():
    graph = IRGraph()
    generator = WebGPUCodeGenerator(graph, [])

    node = IRNode("Add", "add_1", ["in_1", "in_2"])
    node.shape_metadata = None
    node.id = "add_1"

    generator.sorted_nodes = [node]

    with patch.dict("ml_switcheroo_compiler.ops.registry._YAML_REGISTRY", {"Add": {"variants": {"edge_wgsl": {"template": "elementwise"}}}}):
        with patch("ml_switcheroo_compiler.backends.edge.wgsl.wgsl_provider.get_wgsl_template", return_value={"body": "fn body() {{}}"}):
            wgsl, x, y, z = generator._get_wgsl_for_op(node, [], 0, "add_1")


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
