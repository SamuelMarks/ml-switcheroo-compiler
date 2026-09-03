import os
import tempfile
from unittest import mock

from ml_switcheroo_ir import LogicalGraph, LogicalNode

from ml_switcheroo_compiler.export.export_api import ExportArchive


def test_export_archive_init():
    archive = ExportArchive()
    assert isinstance(archive.trackables, dict)
    assert isinstance(archive.endpoints, dict)
    assert isinstance(archive.collections, dict)
    assert isinstance(archive.schema, dict)


def test_export_archive_init_no_yaml():
    with mock.patch("os.path.exists", return_value=False):
        archive = ExportArchive()
        assert archive.schema == {"types": {}, "operations": {}}


def test_track():
    archive = ExportArchive()
    resource = {"some": "data"}
    archive.track(resource)
    assert id(resource) in archive.trackables


def test_add_endpoint():
    archive = ExportArchive()

    def dummy_fn():
        pass

    archive.add_endpoint("dummy", dummy_fn)
    assert "dummy" in archive.endpoints
    assert archive.endpoints["dummy"] is dummy_fn


def test_get_tf_dtype():
    archive = ExportArchive()
    archive.schema = {"types": {"float32": 1, "int32": 3}}
    assert archive._get_tf_dtype("float32") == 1
    assert archive._get_tf_dtype("int32") == 3
    assert archive._get_tf_dtype("unknown") == 1


def test_get_tf_op():
    archive = ExportArchive()
    archive.schema = {"operations": {"Add": "AddV2", "fallback": "Placeholder"}}
    assert archive._get_tf_op("Add") == "AddV2"
    import pytest

    with pytest.raises(ValueError, match="Operation 'Unknown' cannot be exported to TensorFlow schema."):
        archive._get_tf_op("Unknown")


def test_build_signature_def_no_graph():
    archive = ExportArchive()
    sig = archive._build_signature_def("test_sig")
    assert sig is not None


def test_build_signature_def_with_graph():
    archive = ExportArchive()
    graph = LogicalGraph()
    input_node = LogicalNode(id="n1", op_type="Input")
    input_node.dtype = "float32"

    output_node = LogicalNode(id="n2", op_type="Add")
    output_node.dtype = "float32"

    graph.nodes = {"n1": input_node, "n2": output_node}
    graph.outputs = ["n2"]

    sig = archive._build_signature_def("test_sig", graph)
    assert sig is not None


def test_build_graph_def_no_graph():
    archive = ExportArchive()
    graph_def = archive._build_graph_def()
    assert graph_def is not None


def test_build_graph_def_with_graph():
    archive = ExportArchive()
    graph = LogicalGraph()
    input_node = LogicalNode(id="n1", op_type="Input")

    output_node = LogicalNode(id="n2", op_type="Add")
    output_node.inputs = ["n1"]

    graph.nodes = {"n1": input_node, "n2": output_node}
    graph.outputs = ["n2"]

    graph_def = archive._build_graph_def(graph)
    assert graph_def is not None


def test_build_saved_model():
    archive = ExportArchive()
    archive.add_endpoint("test", lambda: None)
    saved_model = archive._build_saved_model()
    assert isinstance(saved_model, bytes)


def test_write_out():
    archive = ExportArchive()
    archive.add_endpoint("test", lambda: None)
    archive.add_variable_collection("vars", [1, 2, 3])

    with tempfile.TemporaryDirectory() as tmpdir:
        archive.write_out(tmpdir)
        assert os.path.exists(os.path.join(tmpdir, "saved_model.pb"))
        assert os.path.exists(os.path.join(tmpdir, "variables", "variables.data-00000-of-00001"))
        assert os.path.exists(os.path.join(tmpdir, "variables", "variables.index"))


def test_add_variable_collection():
    archive = ExportArchive()
    archive.add_variable_collection("vars", [1, 2, 3])
    assert archive.collections["vars"] == [1, 2, 3]


def test_write_out_no_collections():
    archive = ExportArchive()
    with tempfile.TemporaryDirectory() as tmpdir:
        archive.write_out(tmpdir)
        var_data = os.path.join(tmpdir, "variables", "variables.data-00000-of-00001")
        assert os.path.exists(var_data)
        with open(var_data, "rb") as f:
            assert f.read() == b""
