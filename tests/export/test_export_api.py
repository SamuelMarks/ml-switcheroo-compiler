"""Test module."""

import os

from ml_switcheroo_compiler.export.export_api import ExportArchive
from ml_switcheroo_compiler.export.pb_utils import encode_varint
from ml_switcheroo_compiler.ir.core import IRGraph, IRNode


def _create_mock_graph() -> IRGraph:
    """Create a mock graph for testing export."""
    graph = IRGraph()

    in1_node = IRNode(id="in1", op_type="Input", shape_metadata=(1, 10))
    in1_node.dtype = "float32"

    const1_node = IRNode(id="const1", op_type="Constant")
    const1_node.dtype = "int32"

    add1_node = IRNode(id="add1", op_type="Add", inputs=["in1", "const1"])

    out1_node = IRNode(id="out1", op_type="Identity", inputs=["add1"])
    out1_node.dtype = "float64"

    graph.nodes = {
        "in1": in1_node,
        "const1": const1_node,
        "add1": add1_node,
        "out1": out1_node,
    }
    graph.inputs = ["in1"]
    graph.outputs = ["out1"]
    return graph


def test_export_api(tmp_path):
    arch = ExportArchive()

    resource = object()
    arch.track(resource)
    assert arch.trackables[id(resource)] is resource

    arch.add_endpoint("name", lambda x: x)
    assert "name" in arch.endpoints

    arch.add_variable_collection("vars", "variables")
    assert arch.collections["vars"] == "variables"

    mock_graph = _create_mock_graph()
    arch.write_out(str(tmp_path), graph=mock_graph)

    assert os.path.exists(os.path.join(tmp_path, "saved_model.pb"))
    assert os.path.exists(os.path.join(tmp_path, "variables", "variables.data-00000-of-00001"))
    assert os.path.exists(os.path.join(tmp_path, "variables", "variables.index"))


def test_export_api_branch(tmp_path):
    arch = ExportArchive()
    arch.write_out(str(tmp_path))  # Tests the None branch
    assert os.path.exists(os.path.join(tmp_path, "saved_model.pb"))

    # Test explicitly passing None to signature def directly
    sig = arch._build_signature_def("test", graph=None)
    assert sig is not None

    # Force missing schema
    from unittest.mock import patch

    with patch("os.path.exists", return_value=False):
        arch_missing = ExportArchive()
        assert "operations" in arch_missing.schema
        arch_missing.write_out(str(tmp_path))

    # Test track missing resource branches
    try:
        arch.track(None)
    except Exception:
        pass


def test_export_api_graph_def(tmp_path):
    """Test generating a graph def with topological sort."""
    arch = ExportArchive()
    mock_graph = _create_mock_graph()

    # We should also test empty collections branch
    arch.write_out(str(tmp_path), graph=mock_graph)
    assert os.path.exists(os.path.join(tmp_path, "variables", "variables.data-00000-of-00001"))

    # Test tracking works with empty
    arch.collections = {}
    arch.write_out(str(tmp_path), graph=mock_graph)
    assert os.path.getsize(os.path.join(tmp_path, "variables", "variables.data-00000-of-00001")) == 0


def test_pb_utils():
    # Value < 0
    assert encode_varint(-1)

    # Large value
    assert encode_varint(1 << 8)


def test_export_api_no_outputs(tmp_path):
    import os

    from ml_switcheroo_compiler.export.export_api import ExportArchive

    # We need a graph object with hasattr(graph, "outputs") but empty outputs
    class DummyGraph:
        def __init__(self):
            self.inputs = []
            self.outputs = []
            self.nodes = {}

    export_dir = str(tmp_path / "saved_model_no_out")
    ExportArchive().write_out(export_dir, graph=DummyGraph())
    assert os.path.exists(os.path.join(export_dir, "saved_model.pb"))
