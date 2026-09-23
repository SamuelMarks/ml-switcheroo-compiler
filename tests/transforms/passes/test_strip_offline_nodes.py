"""Tests for stripping offline diagnostic and symbolic inspection nodes pass."""

from __future__ import annotations

from ml_switcheroo_compiler.backends.edge.wasm import WasmCodeGenerator
from ml_switcheroo_compiler.ir.core import IRGraph, IRNode
from ml_switcheroo_compiler.transforms.passes.strip_offline_nodes import (
    export_graph_to_dot_string,
    strip_offline_diagnostic_nodes_pass,
)


def test_strip_offline_nodes_empty_and_no_ops() -> None:
    """Test pass behavior on empty graphs and graphs without diagnostic nodes."""
    g_empty = IRGraph()
    res_empty = strip_offline_diagnostic_nodes_pass(g_empty)
    assert res_empty == g_empty

    g_normal = IRGraph()
    n_in = IRNode(id="in0", op_type="Input", inputs=[])
    n_add = IRNode(id="add0", op_type="Add", inputs=["in0", "in0"])
    g_normal.nodes = {"in0": n_in, "add0": n_add}
    g_normal.outputs = ["add0"]

    res_normal = strip_offline_diagnostic_nodes_pass(g_normal)
    assert "in0" in res_normal.nodes
    assert "add0" in res_normal.nodes
    assert len(res_normal.nodes) == 2


def test_strip_offline_diagnostic_nodes_rewiring() -> None:
    """Test rewiring consumers and graph outputs when stripping offline nodes."""
    g = IRGraph()
    n_in = IRNode(id="x", op_type="Input", inputs=[])
    n_diag = IRNode(id="diag0", op_type="ExportToDot", inputs=["x"], attributes={"execute_offline": True})
    n_relu = IRNode(id="relu0", op_type="Relu", inputs=["diag0"])

    g.nodes = {"x": n_in, "diag0": n_diag, "relu0": n_relu}
    g.inputs = ["x"]
    g.outputs = ["relu0"]

    res = strip_offline_diagnostic_nodes_pass(g)
    assert "diag0" not in res.nodes
    assert "x" in res.nodes
    assert "relu0" in res.nodes
    assert res.nodes["relu0"].inputs == ["x"]
    assert "dot_output" in n_diag.attributes
    assert "digraph G" in n_diag.attributes["dot_output"]


def test_strip_offline_diagnostic_nodes_output_rewire() -> None:
    """Test rewiring when diagnostic node is directly in graph outputs."""
    g = IRGraph()
    n_in = IRNode(id="x", op_type="Input", inputs=[])
    n_probe = IRNode(id="probe0", op_type="DiagnosticProbe", inputs=["x"])
    g.nodes = {"x": n_in, "probe0": n_probe}
    g.inputs = ["x"]
    g.outputs = ["probe0"]

    res = strip_offline_diagnostic_nodes_pass(g)
    assert "probe0" not in res.nodes
    assert res.outputs == ["x"]


def test_wasm_generator_ignores_symbolic_ops() -> None:
    """Verify that WasmCodeGenerator strips and excludes offline inspection nodes from kernel emission."""
    g = IRGraph()
    n_in = IRNode(id="in0", op_type="Input", inputs=[], shape_metadata=[4])
    n_dot = IRNode(id="dot0", op_type="ExportToDot", inputs=["in0"], shape_metadata=[4])
    n_exp = IRNode(id="exp0", op_type="Exp", inputs=["dot0"], shape_metadata=[4])

    g.nodes = {"in0": n_in, "dot0": n_dot, "exp0": n_exp}
    g.inputs = ["in0"]
    g.outputs = ["exp0"]

    gen = WasmCodeGenerator(g)
    code = gen.generate()

    assert "ExportToDot" not in code
    assert "buf_dot0" not in code
    assert "buf_exp0" in code


def test_export_graph_to_dot_string() -> None:
    """Verify DOT serialization structure."""
    g = IRGraph()
    n1 = IRNode(id="a", op_type="Input", inputs=[])
    n2 = IRNode(id="b", op_type="Relu", inputs=["a"])
    g.nodes = {"a": n1, "b": n2}

    dot = export_graph_to_dot_string(g)
    assert "digraph G" in dot
    assert '"a" -> "b";' in dot


def test_strip_offline_nodes_chained_and_isolated() -> None:
    """Verify chained transitive rewiring and offline nodes without inputs."""
    g = IRGraph()
    n_in = IRNode(id="in_a", op_type="Input", inputs=[])
    # Chained offline nodes: in_a -> diag1 -> diag2 -> relu
    diag1 = IRNode(id="diag1", op_type="PrintGraph", inputs=["in_a"])
    diag2 = IRNode(id="diag2", op_type="DumpGraph", inputs=["diag1"])
    # Offline node with no inputs at all
    isolated_diag = IRNode(id="iso_diag", op_type="DotExport", inputs=[])
    relu = IRNode(id="relu_out", op_type="Relu", inputs=["diag2"])

    g.nodes = {
        "in_a": n_in,
        "diag1": diag1,
        "diag2": diag2,
        "iso_diag": isolated_diag,
        "relu_out": relu,
    }
    g.inputs = ["in_a"]
    g.outputs = ["relu_out", "iso_diag"]

    res = strip_offline_diagnostic_nodes_pass(g)
    assert "diag1" not in res.nodes
    assert "diag2" not in res.nodes
    assert "iso_diag" not in res.nodes
    assert res.nodes["relu_out"].inputs == ["in_a"]
    assert res.outputs == ["relu_out"]


def test_strip_offline_nodes_topological_sort_exception() -> None:
    """Verify graceful handling when topological_sort fails on cyclic remaining graphs."""
    from unittest.mock import patch

    g = IRGraph()
    n_in = IRNode(id="x", op_type="Input", inputs=[])
    diag = IRNode(id="d", op_type="InspectNode", inputs=["x"])
    g.nodes = {"x": n_in, "d": diag}
    g.inputs = ["x"]
    g.outputs = ["x"]

    with patch("ml_switcheroo_compiler.transforms.passes.strip_offline_nodes.topological_sort", side_effect=ValueError("Cycle")):
        res = strip_offline_diagnostic_nodes_pass(g)
        assert "d" not in res.nodes
        assert "x" in res.nodes


def test_strip_offline_nodes_empty_outputs() -> None:
    """Verify strip pass when graph has no outputs."""
    g = IRGraph()
    n_in = IRNode(id="in0", op_type="Input", inputs=[])
    diag = IRNode(id="d", op_type="PrintGraph", inputs=["in0"])
    g.nodes = {"in0": n_in, "d": diag}
    g.inputs = ["in0"]
    g.outputs = []
    res = strip_offline_diagnostic_nodes_pass(g)
    assert "d" not in res.nodes
    assert res.outputs == []
