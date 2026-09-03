"""Tests for loop_tiling pass."""

from ml_switcheroo_compiler.ir.core import IRGraph, IRNode, LogicalNode
from ml_switcheroo_compiler.transforms.passes.loop_tiling import _load_heuristics, loop_tiling_pass


def test_load_heuristics():
    """Test loading heuristics."""
    h = _load_heuristics()
    assert isinstance(h, dict)


def test_loop_tiling_pass_no_nodes():
    """Test loop_tiling_pass on empty graph."""
    graph = IRGraph()
    assert not loop_tiling_pass(graph)


def test_loop_tiling_pass_modifies_nodes():
    """Test loop_tiling_pass modifies MatMul and Conv2D nodes."""
    graph = IRGraph()
    # (128, 128) -> TILE_M=8, TILE_N=8 by default_wasm profile
    # outer is 128//8 = 16, 128//8 = 16
    # expected shape for matmul: (16, 8, 16, 8)
    n1 = LogicalNode(id="matmul", op_type="MatMul", shape_metadata=(128, 128))
    # (1, 64, 64, 3) -> TILE_H=8, TILE_W=8
    # expected shape for conv2d: (1, 8, 8, 8, 8, 3)
    n2 = LogicalNode(id="conv", op_type="Conv2D", shape_metadata=(1, 64, 64, 3))
    n3 = LogicalNode(id="other", op_type="Add")

    graph.nodes = {"matmul": n1, "conv": n2, "other": n3}
    assert loop_tiling_pass(graph)
    assert getattr(n1, "attributes", {}).get("tiling") is True
    assert getattr(n2, "attributes", {}).get("tiling") is True
    assert getattr(n3, "attributes", {}).get("tiling") is None

    # Check shape splitting
    assert n1.shape_metadata == (16, 8, 16, 8)
    assert n2.shape_metadata == (1, 8, 8, 8, 8, 3)

    # Run again, should not modify because the shape changed
    assert not loop_tiling_pass(graph)


def test_load_heuristics_missing(monkeypatch):
    import os

    monkeypatch.setattr(os.path, "exists", lambda p: False)
    h = _load_heuristics()
    assert h == {}

    graph = IRGraph()
    n1 = LogicalNode(id="matmul", op_type="MatMul", shape_metadata=(128, 128))
    graph.nodes = {"matmul": n1}
    assert not loop_tiling_pass(graph)


def test_loop_tiling_pass_no_attributes():
    graph = IRGraph()
    n1 = LogicalNode(id="matmul", op_type="MatMul", shape_metadata=(128, 128))
    del n1.attributes
    graph.nodes = {"matmul": n1}
    assert loop_tiling_pass(graph)
    assert getattr(n1, "attributes", {}).get("tiling") is True


def test_loop_tiling_extra_coverage():
    from ml_switcheroo_compiler.ir.core import IRGraph
    from ml_switcheroo_compiler.transforms.passes.loop_tiling import _should_tile, loop_tiling_pass

    # Call _should_tile directly with invalid op_type to hit False
    assert _should_tile("unknown", (1, 1), {}) is False

    # Setup graph to hit continue conditions
    graph = IRGraph()
    # 1. op_type is matmul but no shape_metadata
    graph.nodes["n1"] = IRNode(id="n1", op_type="MatMul", inputs=[])

    # 2. op_type is matmul, has shape_metadata, but below threshold to hit 92->81
    n2 = IRNode(id="n2", op_type="MatMul", inputs=[])
    n2.shape_metadata = (2, 2)
    graph.nodes["n2"] = n2

    # Provide a mock yaml configuration so op_config is valid
    from unittest.mock import mock_open, patch

    import yaml

    mock_yaml = {"profiles": {"default_wasm": {"tiling": {"matmul": {"threshold_M": 100, "threshold_N": 100, "threshold_K": 100, "TILE_M": 16, "TILE_N": 16, "TILE_K": 16}}}}}

    with patch("os.path.exists", return_value=True), patch("builtins.open", mock_open(read_data=yaml.dump(mock_yaml))):
        loop_tiling_pass(graph)

    assert getattr(graph.nodes["n1"], "attributes", {}).get("tiling") is None
    assert getattr(graph.nodes["n2"], "attributes", {}).get("tiling") is None

    # Hit not op_config branch
    mock_yaml2 = {"profiles": {"default_wasm": {"tiling": {"matmul": {}}}}}
    with patch("os.path.exists", return_value=True), patch("builtins.open", mock_open(read_data=yaml.dump(mock_yaml2))):
        loop_tiling_pass(graph)


def test_loop_tiling_missing_coverage_extra():
    from ml_switcheroo_compiler.ir.core import IRGraph, IRNode
    from ml_switcheroo_compiler.transforms.passes.loop_tiling import _should_tile, _split_shape

    assert _split_shape("matmul", ("sym", "sym"), {}) == ("sym", "sym")
    assert _split_shape("conv2d", (1, "sym", "sym", 3), {}) == (1, "sym", "sym", 3)
    assert _split_shape("other_op", (1, 2), {}) == (1, 2)
    assert _should_tile("matmul", "not_tuple", {}) is False
    from unittest.mock import patch

    from ml_switcheroo_compiler.transforms.passes.loop_tiling import loop_tiling_pass

    graph = IRGraph()
    # To hit 90 via not op_config
    n1 = IRNode(id="n1", op_type="MatMul", inputs=[])
    n1.shape_metadata = (100, 100)
    graph.nodes["n1"] = n1

    # To hit 90 via not shape
    n2 = IRNode(id="n2", op_type="MatMul", inputs=[])
    n2.shape_metadata = None
    graph.nodes["n2"] = n2

    # To hit 92 -> 81 (false condition for _should_tile)
    n3 = IRNode(id="n3", op_type="MatMul", inputs=[])
    n3.shape_metadata = (2, 2)
    graph.nodes["n3"] = n3
    graph.nodes["n4"] = IRNode(id="n4", op_type="MatMul", inputs=[])
    graph.nodes["n4"].shape_metadata = (100, 100)

    with patch("ml_switcheroo_compiler.transforms.passes.loop_tiling._get_tiling_config", return_value={"matmul": {"threshold_M": 10, "TILE_M": 2, "TILE_N": 2}}):
        loop_tiling_pass(graph)

    with patch("ml_switcheroo_compiler.transforms.passes.loop_tiling._get_tiling_config", return_value={"matmul": {}}):
        loop_tiling_pass(graph)


def test_loop_tiling_should_not_tile():
    from ml_switcheroo_compiler.ir.core import IRGraph
    from ml_switcheroo_compiler.transforms.passes.loop_tiling import loop_tiling_pass

    graph = IRGraph()
    # A tiny shape that shouldn't tile
    node = IRNode(id="n1", op_type="MatMul", inputs=["a", "b"])
    node.shape_metadata = (2, 2)
    graph.nodes["n1"] = node

    res = loop_tiling_pass(graph)
    assert not res
