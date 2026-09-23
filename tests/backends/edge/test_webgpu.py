from unittest.mock import patch

import pytest

from ml_switcheroo_compiler.backends.edge.webgpu import WebGPUCodeGenerator
from ml_switcheroo_compiler.core.errors import CompilationError
from ml_switcheroo_compiler.ir.core import IRGraph, IRNode


def test_webgpu_instantiation():
    graph = IRGraph()
    generator = WebGPUCodeGenerator(graph, [])
    assert generator.graph == graph


def test_webgpu_map_type():
    graph = IRGraph()
    generator = WebGPUCodeGenerator(graph, [])
    assert generator._map_type("float32") == "f32"
    with pytest.warns(UserWarning, match="lacks native float64"):
        assert generator._map_type("float64") == "vec2<f32>"
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

    out_node = IRNode("Output", "out_1", ["add_1"])
    out_node.shape_metadata = [2, 2]
    out_node.id = "out_1"

    graph.nodes = {"in_1": in1, "in_2": in2, "add_1": node, "out_1": out_node}
    graph.sorted_nodes = [in1, in2, node, out_node]
    graph.inputs = ["in_1", "in_2"]
    graph.outputs = ["out_1"]

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
        with pytest.raises(CompilationError, match="cannot be mapped to a valid WGSL compute kernel"):
            generator._get_wgsl_for_op(node, [2, 2, 2, 2], 16, "u_1")

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

    orig_safe_load = yaml.safe_load

    def _conditional_safe_load(stream):
        name = getattr(stream, "name", "")
        if "memory_schemas" in str(name):
            return mocked_data
        return orig_safe_load(stream)

    with patch("yaml.safe_load", side_effect=_conditional_safe_load), patch("ml_switcheroo_compiler.backends.edge.webgpu_webrtc.emit_webrtc_init", return_value=""), patch("ml_switcheroo_compiler.backends.edge.webgpu_webrtc.emit_webrtc_op", return_value=""):
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


def test_webgpu_softmax_and_layernorm():
    """Test generating WebGPU WGSL compute shaders for Softmax and LayerNorm."""
    in_node = IRNode("in_0", "Input", inputs=[])
    in_node.shape_metadata = (4, 16)

    sm_node = IRNode("sm_0", "Softmax", inputs=["in_0"])
    sm_node.shape_metadata = (4, 16)

    ln_node = IRNode("ln_0", "LayerNorm", inputs=["in_0"])
    ln_node.shape_metadata = (4, 16)

    out_sm = IRNode("out_sm", "Output", inputs=["sm_0"])
    out_ln = IRNode("out_ln", "Output", inputs=["ln_0"])

    graph = IRGraph()
    graph.nodes = {"in_0": in_node, "sm_0": sm_node, "ln_0": ln_node, "out_sm": out_sm, "out_ln": out_ln}
    graph.inputs = ["in_0"]
    graph.outputs = ["out_sm", "out_ln"]

    gen = WebGPUCodeGenerator(graph)
    gen.sorted_nodes = [in_node, sm_node, ln_node, out_sm, out_ln]
    code = gen.generate()

    assert "compute_sm_0" in code
    assert "compute_ln_0" in code
    assert "sum_exp" in code
    assert "variance" in code


def test_webgpu_additional_coverage():
    """Verify missing branches in WebGPU code generator."""
    # 1. generate_js_runner with Input and Output nodes in sorted_nodes
    g = IRGraph()
    g.inputs = ["in_0"]
    in_n = IRNode("in_0", "Input")
    add_n = IRNode("add_0", "Add", inputs=["in_0", "in_0"])
    add_n.shape_metadata = (4,)
    out_n = IRNode("out_0", "Output", inputs=["add_0"])
    g.nodes = {"in_0": in_n, "add_0": add_n, "out_0": out_n}
    g.outputs = ["out_0"]
    gen = WebGPUCodeGenerator(g)
    gen.sorted_nodes = [in_n, add_n, out_n]
    js = gen.generate()
    assert "commandEncoder" in js

    # 2. visit_Attention with [0, 0] shape
    attn_node = IRNode("attn_0", "Attention", inputs=["in_0"])
    attn_node.shape_metadata = (0, 0)
    attn_wgsl, x, y, z = gen.visit_Attention(attn_node, ["in_0"], shape=[0, 0], nelem=0, clean_id="attn_0")
    assert any("attention" in line or "scale" in line for line in attn_wgsl)

    # 3. apply_shape_telemetry with invalid telemetry
    gen.apply_shape_telemetry({"shapes": "not_a_dict"})

    # 4. apply_shape_telemetry with string dimension and dim.node.eval
    class MockDimNode:
        def __init__(self, should_fail=False):
            self.node = self
            self.should_fail = should_fail

        def eval(self, env):
            if self.should_fail:
                raise ValueError("eval failure")
            return 8

    node_eval_ok = IRNode("eval_ok", "Relu")
    node_eval_ok.shape_metadata = (MockDimNode(False),)
    node_eval_fail = IRNode("eval_fail", "Relu")
    node_eval_fail.shape_metadata = (MockDimNode(True),)
    node_str = IRNode("str_node", "Relu")
    node_str.shape_metadata = ("B", 16)
    node_source = IRNode("src_node", "Relu")
    node_source.shape_metadata = ("B", 16)

    gen.sorted_nodes = [node_source, node_str, node_eval_ok, node_eval_fail]
    gen.apply_shape_telemetry({"src_node": [4, 16]})
    assert node_str.shape_metadata == (4, 16)
    assert node_eval_ok.shape_metadata == (8,)
    assert node_eval_fail.shape_metadata == (1,)


def test_webgpu_shape_telemetry_additional_branches():
    """Verify missing telemetry branches in WebGPUCodeGenerator.apply_shape_telemetry."""
    g = IRGraph()
    gen = WebGPUCodeGenerator(g)

    node_not_seq = IRNode("not_seq", "Relu")
    node_not_seq.shape_metadata = (4, 4)
    node_mismatch = IRNode("mismatch", "Relu")
    node_mismatch.shape_metadata = (4, 4, 4)
    node_no_meta = IRNode("no_meta", "Relu")
    node_no_meta.shape_metadata = None
    node_empty_meta = IRNode("empty_meta", "Relu")
    node_empty_meta.shape_metadata = ()
    node_unsolved = IRNode("unsolved", "Relu")
    node_unsolved.shape_metadata = ("unsolved_dim",)

    gen.sorted_nodes = [node_not_seq, node_mismatch, node_no_meta, node_empty_meta, node_unsolved]
    gen.apply_shape_telemetry({"not_seq": 123, "mismatch": [4, 4]})
    assert node_not_seq.shape_metadata == (4, 4)
    assert node_mismatch.shape_metadata == (4, 4)


def test_webgpu_parameterized_control_flow():
    """Verify parameterized WhileLoop, Cond, and Scan emission in WebGPUCodeGenerator."""
    g = IRGraph()
    x = IRNode(id="x", op_type="Input")

    # WhileLoop with custom comparator and threshold
    while_node = IRNode(
        id="while_1",
        op_type="WhileLoop",
        inputs=["x"],
        attributes={"comparator": ">", "threshold": "25.0", "max_iters": 8},
    )

    # Cond with custom comparator and threshold
    cond_node = IRNode(
        id="cond_1",
        op_type="Cond",
        inputs=["x"],
        attributes={"comparator": "<", "threshold": "1.0"},
    )

    # Scan with custom init_val and scan_op_expr
    scan_node = IRNode(
        id="scan_1",
        op_type="Scan",
        inputs=["x"],
        attributes={"init_val": "2.0", "scan_op_expr": "acc * buf_in0_f32[i]"},
    )

    g.nodes = {"x": x, "while_1": while_node, "cond_1": cond_node, "scan_1": scan_node}
    g.inputs = ["x"]
    g.outputs = ["while_1", "cond_1", "scan_1"]

    gen = WebGPUCodeGenerator(g)
    wgsl_while, _, _, _ = gen.visit_WhileLoop(while_node, ["x"], nelem=16, clean_id="while_1")
    assert any("current_state > 25.0" in line for line in wgsl_while)

    wgsl_cond, _, _, _ = gen.visit_Cond(cond_node, ["x"], nelem=16, clean_id="cond_1")
    assert any("buf_in0_f32[idx] < 1.0" in line for line in wgsl_cond)

    wgsl_scan, _, _, _ = gen.visit_Scan(scan_node, ["x"], nelem=16, clean_id="scan_1")
    assert any("var acc: f32 = 2.0;" in line for line in wgsl_scan)
    assert any("acc * buf_in0_f32[i]" in line for line in wgsl_scan)

    # 4. WhileLoop with cond_graph containing Greater and threshold
    cond_sub = IRGraph()
    cond_sub_node = IRNode(id="cmp", op_type="Greater", attributes={"threshold": "100.0"})
    cond_sub.nodes = {"cmp": cond_sub_node}
    body_sub = IRGraph()
    body_sub_node = IRNode(id="step", op_type="Add", attributes={})
    body_sub.nodes = {"step": body_sub_node}
    while_cond_graph_node = IRNode(
        id="while_2",
        op_type="WhileLoop",
        inputs=["x"],
        attributes={"cond": cond_sub, "body": body_sub},
    )
    wgsl_while_sub, _, _, _ = gen.visit_WhileLoop(while_cond_graph_node, ["x"], nelem=16, clean_id="while_2")
    assert any("current_state > 100.0" in line for line in wgsl_while_sub)
    assert any("tmp_step" in line for line in wgsl_while_sub)

    # WhileLoop with body graph containing unmapped op covering branch 445->448
    body_empty = IRGraph()
    body_empty.nodes = {"sub": IRNode(id="sub", op_type="NonExistentOpForLoopBody")}
    while_empty_body_node = IRNode(
        id="while_empty",
        op_type="WhileLoop",
        inputs=["x"],
        attributes={"body": body_empty},
    )
    wgsl_while_empty, _, _, _ = gen.visit_WhileLoop(while_empty_body_node, ["x"], nelem=16, clean_id="while_empty")
    assert any("current_state + buf_in1_f32[idx]" in line for line in wgsl_while_empty)

    # 5. Softmax, AvgPool2D, LayerNorm, and Trig op resolution
    node_softmax = IRNode(id="sm", op_type="Softmax", inputs=["x"], shape_metadata=[4, 4])
    gen.sorted_nodes = [x, node_softmax]
    wgsl_sm, _, _, _ = gen._get_wgsl_for_op(node_softmax, [4, 4], 16, "sm")
    assert len(wgsl_sm) > 0

    node_avgpool = IRNode(id="ap", op_type="AvgPool2D", inputs=["x"], shape_metadata=[1, 1, 4, 4])
    gen.sorted_nodes = [x, node_avgpool]
    wgsl_ap, _, _, _ = gen._get_wgsl_for_op(node_avgpool, [1, 1, 4, 4], 16, "ap")
    assert len(wgsl_ap) > 0

    node_const = IRNode(id="const_1", op_type="Constant", inputs=[], shape_metadata=[1], attributes={"value": 3.14})
    gen.sorted_nodes = [x, node_const]
    wgsl_const, _, _, _ = gen._get_wgsl_for_op(node_const, [1], 1, "const_1")
    assert len(wgsl_const) > 0

    node_conv = IRNode(id="c2d", op_type="Conv2D", inputs=["x"], shape_metadata=[1, 1, 4, 4])
    gen.sorted_nodes = [x, node_conv]
    wgsl_conv, _, _, _ = gen._get_wgsl_for_op(node_conv, [1, 1, 4, 4], 16, "c2d")
    assert len(wgsl_conv) > 0

    node_ln = IRNode(id="ln", op_type="LayerNorm", inputs=["x"], shape_metadata=[4, 4])
    gen.sorted_nodes = [x, node_ln]
    wgsl_ln, _, _, _ = gen._get_wgsl_for_op(node_ln, [4, 4], 16, "ln")
    assert len(wgsl_ln) > 0

    node_trig = IRNode(id="trig", op_type="Trig", inputs=["x"], shape_metadata=[4, 4])
    gen.sorted_nodes = [x, node_trig]
    wgsl_trig, _, _, _ = gen._get_wgsl_for_op(node_trig, [4, 4], 16, "trig")
    assert len(wgsl_trig) > 0

    # 6. _compile_aot_impl with new graph and repeat with same graph
    new_g = IRGraph()
    new_x = IRNode(id="nx", op_type="Input")
    new_g.nodes = {"nx": new_x}
    new_g.inputs = ["nx"]
    aot_bundle = gen._compile_aot_impl(new_g)
    assert aot_bundle["format"] == "webgpu_wgsl_bundle"
    assert aot_bundle["status"] == "ready_to_dispatch"

    aot_bundle_repeat = gen._compile_aot_impl(new_g)
    assert aot_bundle_repeat["format"] == "webgpu_wgsl_bundle"

    # 7. Node with more than 3 inputs to exercise j >= 3 branch
    four_in_g = IRGraph()
    i1 = IRNode(id="i1", op_type="Input")
    i2 = IRNode(id="i2", op_type="Input")
    i3 = IRNode(id="i3", op_type="Input")
    i4 = IRNode(id="i4", op_type="Input")
    four_op = IRNode(id="four_op", op_type="Add", inputs=["i1", "i2", "i3", "i4"])
    four_in_g.nodes = {"i1": i1, "i2": i2, "i3": i3, "i4": i4, "four_op": four_op}
    four_in_g.inputs = ["i1", "i2", "i3", "i4"]
    four_in_g.outputs = ["four_op"]
    gen_four = WebGPUCodeGenerator(four_in_g)
    gen_four.generate()


def test_webgpu_generate_training_step():
    """Verify coupled forward-backward WGSL generation for in-browser training."""
    g = IRGraph()
    x = IRNode(id="x", op_type="Input")
    x.shape_metadata = [2, 2]
    w = IRNode(id="w", op_type="Input")
    w.shape_metadata = [2, 2]
    mul_node = IRNode(id="mul_1", op_type="Mul", inputs=["x", "w"])
    mul_node.shape_metadata = [2, 2]
    out_node = IRNode(id="out_1", op_type="Output", inputs=["mul_1"])
    out_node.shape_metadata = [2, 2]

    g.nodes = {"x": x, "w": w, "mul_1": mul_node, "out_1": out_node}
    g.inputs = ["x", "w"]
    g.outputs = ["out_1"]

    gen = WebGPUCodeGenerator(g)
    code = gen.generate_training_step(wrt_inputs=["x", "w"], target_output="out_1")
    assert "execute_train_step" in code
    assert "bwd_execute" in code
    assert "pBuf[i] -= learning_rate" in code

    # Test error when outputs are empty
    empty_out_g = IRGraph()
    empty_out_g.nodes = {"x": x}
    empty_out_g.inputs = ["x"]
    empty_out_g.outputs = []
    empty_gen = WebGPUCodeGenerator(empty_out_g)
    with pytest.raises(ValueError, match="Target output cannot be identified"):
        empty_gen.generate_training_step()


def test_webgpu_visit_broadcast_and_empty_outputs_exception() -> None:
    """Test WebGPU visit_Broadcast and compile_forward_backward empty outputs."""
    from ml_switcheroo_ir import LogicalNode

    graph = IRGraph(name="webgpu_extra")
    node_bc = LogicalNode(id="bc_0", op_type="Broadcast", inputs=["in_0"], shape_metadata=(4,))
    node_out = LogicalNode(id="out_0", op_type="Relu", inputs=["bc_0"], shape_metadata=(4,))
    graph.nodes = {"bc_0": node_bc, "out_0": node_out}
    graph.inputs = ["in_0"]
    graph.outputs = ["out_0"]
    gen = WebGPUCodeGenerator(graph)
    (lines, x, y, z) = gen.visit_Broadcast(node_bc, ["buf_in0_f32"])
    assert len(lines) > 0
    assert (x, y, z) == ("1", "1", "1")

    mock_bwd = IRGraph(name="mock_bwd", nodes={"out_0": node_out})
    mock_bwd.inputs = []
    with patch("ml_switcheroo_compiler.transforms.autodiff.grad", return_value=mock_bwd):
        step_code = gen.generate_training_step(target_output=None)
        assert len(step_code) > 0

    graph_empty = IRGraph(name="empty_outputs", nodes={})
    graph_empty.inputs = []
    graph_empty.outputs = []
    gen_empty = WebGPUCodeGenerator(graph_empty)
    with pytest.raises(ValueError, match="Target output cannot be identified"):
        gen_empty.generate_training_step(target_output=None)


def test_webgpu_arbitrary_inputs_and_multiple_outputs() -> None:
    """Test WebGPU code generation with >3 inputs and multiple outputs."""
    graph = IRGraph()
    inputs = [IRNode(id=f"in_{i}", op_type="Input", inputs=[]) for i in range(5)]
    for inp in inputs:
        inp.shape_metadata = [2, 2]
        graph.nodes[inp.id] = inp

    multi_op = IRNode(id="multi_1", op_type="Add", inputs=[f"in_{i}" for i in range(5)])
    multi_op.shape_metadata = [2, 2]
    multi_op.attributes["outputs"] = ["out_primary", "out_secondary"]
    graph.nodes[multi_op.id] = multi_op

    out1 = IRNode(id="out_primary", op_type="Output", inputs=["multi_1"])
    out1.shape_metadata = [2, 2]
    out2 = IRNode(id="out_secondary", op_type="Output", inputs=["multi_1"])
    out2.shape_metadata = [2, 2]
    graph.nodes[out1.id] = out1
    graph.nodes[out2.id] = out2

    graph.sorted_nodes = [*inputs, multi_op, out1, out2]
    graph.inputs = [f"in_{i}" for i in range(5)]
    graph.outputs = ["out_primary", "out_secondary"]

    gen = WebGPUCodeGenerator(graph, [])
    js_code = gen.generate()
    assert "binding: 0" in js_code
    assert "binding: 1" in js_code
    assert "binding: 2" in js_code
    assert "binding: 4" in js_code
    assert "binding: 5" in js_code
    assert "binding: 3" in js_code
    assert "binding: 6" in js_code


def test_wgsl_grounding_schema_validation() -> None:
    """Verify validation of WGSL statements against canonical wgsl_ops schema."""
    from ml_switcheroo_compiler.backends.edge.wgsl.wgsl_provider import (
        get_wgsl_grounding_schema,
        validate_wgsl_statement,
    )

    schema = get_wgsl_grounding_schema()
    assert "ops" in schema
    assert len(schema["ops"]) > 0

    assert validate_wgsl_statement("storageStore") is True
    assert validate_wgsl_statement("add") is True
    assert validate_wgsl_statement("non_existent_fake_wgsl_op") is False


def test_webgpu_multi_outputs_attribute_and_empty_nodes() -> None:
    """Verify WebGPU generation when node has attributes['outputs'] and when graph has no nodes."""
    # 1. Node with attributes['outputs'] as list
    graph = IRGraph(name="test_multi_out_attr")
    in_node = IRNode(id="in_0", op_type="Input", inputs=[])
    in_node.shape_metadata = [2, 2]
    add_node = IRNode(id="add_0", op_type="Add", inputs=["in_0", "in_0"])
    add_node.shape_metadata = [2, 2]
    add_node.outputs = None  # type: ignore[assignment]
    add_node.attributes = {}
    graph.nodes["in_0"] = in_node
    graph.nodes["add_0"] = add_node
    graph.sorted_nodes = [in_node, add_node]
    graph.inputs = ["in_0"]
    graph.outputs = ["add_0"]

    gen = WebGPUCodeGenerator(graph, [])
    js_code = gen.generate()
    assert "add_0" in js_code or "buf_arena_" in js_code

    # 2. Graph with empty sorted_nodes
    empty_graph = IRGraph(name="empty_graph")
    empty_graph.nodes = {}
    empty_graph.sorted_nodes = []
    empty_graph.inputs = []
    empty_graph.outputs = []
    gen_empty = WebGPUCodeGenerator(empty_graph, [])
    js_empty = gen_empty.generate()
    assert "@group(0)" in js_empty


def test_webgpu_spatial_3d_and_depthwise_ops() -> None:
    """Verify WGSL code generation for Conv3D, DepthwiseConv2D, MaxPool3D, and AvgPool3D."""
    # 1. Conv3D
    graph_conv3d = IRGraph(name="test_conv3d")
    in_x = IRNode(id="in_x", op_type="Input", inputs=[])
    in_x.shape_metadata = [1, 2, 4, 8, 8]
    in_w = IRNode(id="in_w", op_type="Input", inputs=[])
    in_w.shape_metadata = [4, 2, 2, 3, 3]
    c3d = IRNode(id="c3d", op_type="Conv3D", inputs=["in_x", "in_w"])
    c3d.shape_metadata = [1, 4, 3, 6, 6]
    c3d.attributes = {"strides": (1, 1, 1)}
    graph_conv3d.nodes = {"in_x": in_x, "in_w": in_w, "c3d": c3d}
    graph_conv3d.sorted_nodes = [in_x, in_w, c3d]
    graph_conv3d.inputs = ["in_x", "in_w"]
    graph_conv3d.outputs = ["c3d"]

    gen_c3d = WebGPUCodeGenerator(graph_conv3d, [])
    code_c3d = gen_c3d.generate()
    assert "compute_c3d" in code_c3d
    assert "workgroup_size(8, 8, 4)" in code_c3d

    # 2. DepthwiseConv2D
    graph_dw = IRGraph(name="test_dw_conv2d")
    dw_x = IRNode(id="dw_x", op_type="Input", inputs=[])
    dw_x.shape_metadata = [1, 4, 16, 16]
    dw_w = IRNode(id="dw_w", op_type="Input", inputs=[])
    dw_w.shape_metadata = [3, 3]
    dw_op = IRNode(id="dw_op", op_type="DepthwiseConv2D", inputs=["dw_x", "dw_w"])
    dw_op.shape_metadata = [1, 4, 14, 14]
    dw_op.attributes = {"strides": (1, 1)}
    graph_dw.nodes = {"dw_x": dw_x, "dw_w": dw_w, "dw_op": dw_op}
    graph_dw.sorted_nodes = [dw_x, dw_w, dw_op]
    graph_dw.inputs = ["dw_x", "dw_w"]
    graph_dw.outputs = ["dw_op"]

    gen_dw = WebGPUCodeGenerator(graph_dw, [])
    code_dw = gen_dw.generate()
    assert "compute_dw_op" in code_dw

    # 3. MaxPool3D
    graph_mp3d = IRGraph(name="test_mp3d")
    mp_x = IRNode(id="mp_x", op_type="Input", inputs=[])
    mp_x.shape_metadata = [1, 2, 4, 8, 8]
    mp3d_op = IRNode(id="mp3d_op", op_type="MaxPool3D", inputs=["mp_x"])
    mp3d_op.shape_metadata = [1, 2, 2, 4, 4]
    mp3d_op.attributes = {"strides": (2, 2, 2), "window": (2, 2, 2)}
    graph_mp3d.nodes = {"mp_x": mp_x, "mp3d_op": mp3d_op}
    graph_mp3d.sorted_nodes = [mp_x, mp3d_op]
    graph_mp3d.inputs = ["mp_x"]
    graph_mp3d.outputs = ["mp3d_op"]

    gen_mp3d = WebGPUCodeGenerator(graph_mp3d, [])
    code_mp3d = gen_mp3d.generate()
    assert "compute_mp3d_op" in code_mp3d

    # 4. AvgPool3D
    graph_ap3d = IRGraph(name="test_ap3d")
    ap_x = IRNode(id="ap_x", op_type="Input", inputs=[])
    ap_x.shape_metadata = [1, 2, 4, 8, 8]
    ap3d_op = IRNode(id="ap3d_op", op_type="AvgPool3D", inputs=["ap_x"])
    ap3d_op.shape_metadata = [1, 2, 2, 4, 4]
    ap3d_op.attributes = {"strides": (2, 2, 2), "window": (2, 2, 2)}
    graph_ap3d.nodes = {"ap_x": ap_x, "ap3d_op": ap3d_op}
    graph_ap3d.sorted_nodes = [ap_x, ap3d_op]
    graph_ap3d.inputs = ["ap_x"]
    graph_ap3d.outputs = ["ap3d_op"]

    gen_ap3d = WebGPUCodeGenerator(graph_ap3d, [])
    code_ap3d = gen_ap3d.generate()
    assert "compute_ap3d_op" in code_ap3d


def test_webgpu_custom_strides_and_non_contiguous() -> None:
    """Verify custom memory strides handling in WebGPUCodeGenerator."""
    graph = IRGraph(name="test_strides")
    node = IRNode(id="stride_node", op_type="Input", inputs=[])
    node.shape_metadata = [4, 8]
    node.attributes = {"strides": [16, 2]}
    graph.nodes = {"stride_node": node}
    graph.sorted_nodes = [node]

    gen = WebGPUCodeGenerator(graph, [])
    shape, strides = gen._get_shape_and_strides(node)
    assert shape == [4, 8]
    assert strides == [16, 2]
