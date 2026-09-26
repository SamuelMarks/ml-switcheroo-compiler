"""Unit tests for WebGL 2.0 shader code generation, texture packing, and operators."""

from typing import cast

import numpy as np

from ml_switcheroo_compiler.backends.edge.webgl import WebGLCodeGenerator
from ml_switcheroo_compiler.ir.core import IRGraph, IRNode


def test_webgl_slice_codegen() -> None:
    """Verify WebGL code generation for tensor slicing operations."""
    graph = IRGraph(name="test_slice")
    in_node = IRNode(id="in_0", op_type="Input", shape_metadata=(64, 64))
    slice_node = IRNode(
        id="slice_0",
        op_type="Slice",
        inputs=["in_0"],
        shape_metadata=(16, 32),
        attributes={"starts": [8, 16], "ends": [24, 48]},
    )
    graph.nodes = {"in_0": in_node, "slice_0": slice_node}
    graph.inputs = ["in_0"]
    graph.outputs = ["slice_0"]

    generator = WebGLCodeGenerator(graph)
    js_code = generator.generate()

    assert "start_x" in js_code
    assert "start_y" in js_code
    assert "shader_slice_0" in js_code
    assert "texelFetch(A, inCoord, 0)" in js_code


def test_webgl_concat_codegen() -> None:
    """Verify WebGL code generation for tensor concatenation along dimensions."""
    graph = IRGraph(name="test_concat")
    in_a = IRNode(id="in_a", op_type="Input", shape_metadata=(16, 32))
    in_b = IRNode(id="in_b", op_type="Input", shape_metadata=(16, 32))
    concat_node = IRNode(
        id="concat_0",
        op_type="Concat",
        inputs=["in_a", "in_b"],
        shape_metadata=(32, 32),
        attributes={"axis": 0},
    )
    graph.nodes = {"in_a": in_a, "in_b": in_b, "concat_0": concat_node}
    graph.inputs = ["in_a", "in_b"]
    graph.outputs = ["concat_0"]

    generator = WebGLCodeGenerator(graph)
    js_code = generator.generate()

    assert "split_boundary" in js_code
    assert "concat_axis" in js_code
    assert "shader_concat_0" in js_code


def test_webgl_reduce_codegen() -> None:
    """Verify WebGL code generation for reduction operations with axis support."""
    graph = IRGraph(name="test_reduce")
    in_a = IRNode(id="in_a", op_type="Input", shape_metadata=(16, 32))
    reduce_node = IRNode(
        id="reduce_0",
        op_type="Reduce",
        inputs=["in_a"],
        shape_metadata=(16, 1),
        attributes={"axis": 1},
    )
    graph.nodes = {"in_a": in_a, "reduce_0": reduce_node}
    graph.inputs = ["in_a"]
    graph.outputs = ["reduce_0"]

    generator = WebGLCodeGenerator(graph)
    js_code = generator.generate()

    assert "reduce_axis" in js_code
    assert "shader_reduce_0" in js_code


def test_webgl_texture_packing_3d_and_4d() -> None:
    """Verify texture packing for 3D and 4D tensors without dummy 32x32 clamps."""
    # 4D Conv2D output: (B, C, H, W) = (2, 8, 14, 14) -> height = 2*8*14 = 224, width = 14
    graph_4d = IRGraph(name="test_pack_4d")
    in_0 = IRNode(id="in_0", op_type="Input", shape_metadata=(2, 8, 16, 16))
    conv_0 = IRNode(
        id="conv_0",
        op_type="Conv2D",
        inputs=["in_0"],
        shape_metadata=(2, 8, 14, 14),
    )
    graph_4d.nodes = {"in_0": in_0, "conv_0": conv_0}
    graph_4d.inputs = ["in_0"]
    graph_4d.outputs = ["conv_0"]

    gen_4d = WebGLCodeGenerator(graph_4d)
    js_4d = gen_4d.generate()

    assert "readPixels(gl, 14, 224);" in js_4d
    assert "gl.viewport(0, 0, 14, 224);" in js_4d


def test_webgl_starts_and_dynamic_conv_edge_branches() -> None:
    """Verify starts attribute forms (int, len-1 non-int, len-2 non-int) and non-int Conv2D spatial dimensions."""
    graph = IRGraph(name="test_starts_and_conv")
    in_node = IRNode("in", "Input", shape_metadata=[32, 32])

    # 1. starts as int
    slice_int = IRNode("slice_int", "Slice", inputs=["in"], shape_metadata=[16, 16], attributes={"starts": 5})

    # 2. starts as list of len 1 with non-int
    slice_1 = IRNode("slice_1", "Slice", inputs=["in"], shape_metadata=[16, 16], attributes={"starts": ["str_dim"]})

    # 3. starts as list of len >= 2 with non-ints
    slice_2 = IRNode("slice_2", "Slice", inputs=["in"], shape_metadata=[16, 16], attributes={"starts": ["str_y", "str_x"]})

    # 3b. starts as empty list and non-int/non-sequence
    slice_empty = IRNode("slice_empty", "Slice", inputs=["in"], shape_metadata=[16, 16], attributes={"starts": []})
    slice_str = IRNode("slice_str", "Slice", inputs=["in"], shape_metadata=[16, 16], attributes={"starts": "str_starts"})

    # 4. Conv2D with non-int dimension in eff_shape[:-1]
    in_dyn = IRNode("in_dyn", "Input", shape_metadata=["dynamic_batch", 16, 16])
    conv_dyn = IRNode("conv_dyn", "Conv2D", inputs=["in_dyn"], shape_metadata=["dynamic_batch", 16, 16])

    graph.nodes = {
        "in": in_node,
        "slice_int": slice_int,
        "slice_1": slice_1,
        "slice_2": slice_2,
        "slice_empty": slice_empty,
        "slice_str": slice_str,
        "in_dyn": in_dyn,
        "conv_dyn": conv_dyn,
    }
    graph.inputs = ["in", "in_dyn"]
    graph.outputs = ["slice_int", "slice_1", "slice_2", "slice_empty", "slice_str", "conv_dyn"]

    gen = WebGLCodeGenerator(graph)
    js = gen.generate()
    assert "'start_y'), 5" in js
    assert "'start_y'), 0" in js

    # 3D generic tensor: (4, 10, 20) -> outer = 40, width = 20
    graph_3d = IRGraph(name="test_pack_3d")
    in_3d = IRNode(id="in_3d", op_type="Input", shape_metadata=(4, 10, 20))
    relu_0 = IRNode(
        id="relu_0",
        op_type="Relu",
        inputs=["in_3d"],
        shape_metadata=(4, 10, 20),
    )
    graph_3d.nodes = {"in_3d": in_3d, "relu_0": relu_0}
    graph_3d.inputs = ["in_3d"]
    graph_3d.outputs = ["relu_0"]

    gen_3d = WebGLCodeGenerator(graph_3d)
    js_3d = gen_3d.generate()

    assert "readPixels(gl, 20, 40);" in js_3d


def test_webgl_coordinate_transformations_uniforms() -> None:
    """Verify coordinate transformation uniforms (strides, dims, rank, and viewport bounds)."""
    graph = IRGraph(name="test_coord_uniforms")
    in_0 = IRNode(id="in_0", op_type="Input", shape_metadata=(2, 4, 8))
    node_op = IRNode(id="exp_0", op_type="Exp", inputs=["in_0"], shape_metadata=(2, 4, 8))
    graph.nodes = {"in_0": in_0, "exp_0": node_op}
    graph.inputs = ["in_0"]
    graph.outputs = ["exp_0"]

    gen = WebGLCodeGenerator(graph)
    js = gen.generate()

    assert "'rank'), 3" in js
    assert "'dim_0'), 2" in js
    assert "'dim_1'), 4" in js
    assert "'dim_2'), 8" in js
    assert "'stride_0'), 32" in js
    assert "'stride_1'), 8" in js
    assert "'stride_2'), 1" in js
    assert "'out_width'), 8" in js
    assert "'out_height'), 8" in js


def test_webgl_slice_and_concat_runner() -> None:
    """Verify execution of compiled WebGL artifact for slice and concat with NumPy fallback."""
    graph = IRGraph(name="test_runner_slice")
    in_x = IRNode(id="x", op_type="Input", shape_metadata=(4, 4))
    slice_op = IRNode(
        id="s",
        op_type="Slice",
        inputs=["x"],
        shape_metadata=(2, 2),
        attributes={"dim": 0, "start": 0, "end": 2},
    )
    graph.nodes = {"x": in_x, "s": slice_op}
    graph.inputs = ["x"]
    graph.outputs = ["s"]

    gen = WebGLCodeGenerator(graph)
    artifact = gen.compile_aot(graph)

    inp = np.arange(16, dtype=np.float32).reshape(4, 4)
    res = artifact(inp)
    np.testing.assert_allclose(cast(np.ndarray, res), inp[0:2, :])
