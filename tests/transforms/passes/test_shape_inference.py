# ruff: noqa: E501
import pytest

from ml_switcheroo_compiler.core.errors import CompilationError
from ml_switcheroo_compiler.ir.core import IRGraph, IRNode
from ml_switcheroo_compiler.transforms.passes.shape_inference import _determine_node_shape, _infer_constant_shape, _infer_op_shape, _infer_output_shape, _prepare_op_kwargs, shape_inference_pass


def test_infer_constant_shape() -> None:
    node = IRNode(id="n1", op_type="Constant", inputs=[], attributes={"value": [1.0, 2.0]})
    assert _infer_constant_shape(node, {}) == (2,)


def test_infer_output_shape() -> None:
    node1 = IRNode(id="n1", op_type="Output", inputs=["in1"], attributes={})
    assert _infer_output_shape(node1, {"in1": (3, 4)}) == (3, 4)
    node2 = IRNode(id="n2", op_type="Output", inputs=[], attributes={})
    assert _infer_output_shape(node2, {}) is None


def test_prepare_op_kwargs() -> None:
    node1 = IRNode(id="n1", op_type="Expand", inputs=[], attributes={"other": 1}, shape_metadata=(1, 2))
    assert _prepare_op_kwargs(node1) == {"other": 1, "shape": (1, 2)}
    node2 = IRNode(id="n2", op_type="BroadcastTo", inputs=[], attributes={}, shape_metadata=(3, 3))
    assert _prepare_op_kwargs(node2) == {"shape": (3, 3)}
    node3 = IRNode(id="n3", op_type="Reshape", inputs=[], attributes={}, shape_metadata=(4,))
    assert _prepare_op_kwargs(node3) == {"newshape": (4,)}
    node4 = IRNode(id="n4", op_type="Add", inputs=[], attributes={"other": 1}, shape_metadata=(1,))
    assert _prepare_op_kwargs(node4) == {"other": 1}


def test_infer_op_shape() -> None:
    import unittest.mock as mock

    node1 = IRNode(id="n1", op_type="Add", inputs=["in1", "in2"], attributes={})
    shapes = {"in1": (1, 2), "in2": (1, 2)}

    class MockOp:
        def infer_shape(self, *args, **kwargs):
            return (1, 2)

    with mock.patch("ml_switcheroo_compiler.transforms.passes.shape_inference.get_op", return_value=lambda: MockOp()):
        assert _infer_op_shape(node1, shapes) == (1, 2)


def test_determine_node_shape() -> None:
    import unittest.mock as mock

    node1 = IRNode(id="n1", op_type="Input", inputs=[], attributes={}, shape_metadata=(5, 5))
    assert _determine_node_shape(node1, {}) == (5, 5)
    node2 = IRNode(id="n2", op_type="UnknownOp", inputs=[], attributes={}, shape_metadata=(1,))
    assert _determine_node_shape(node2, {}) == (1,)
    node3 = IRNode(id="n3", op_type="Add", inputs=["in1", "in2"], attributes={})

    class MockOp2:
        def infer_shape(self, *args, **kwargs):
            raise ValueError("boom")

    with mock.patch("ml_switcheroo_compiler.transforms.passes.shape_inference.get_op", return_value=lambda: MockOp2()):
        with pytest.raises(CompilationError):
            _determine_node_shape(node3, {"in1": (1, 2), "in2": (3, 4)})


def test_shape_inference_pass() -> None:
    import unittest.mock as mock

    node1 = IRNode(id="in1", op_type="Input", inputs=[], attributes={}, shape_metadata=(2, 2))
    node2 = IRNode(id="in2", op_type="Input", inputs=[], attributes={}, shape_metadata=(2, 2))
    node3 = IRNode(id="add", op_type="Add", inputs=["in1", "in2"], attributes={}, shape_metadata=None)
    graph = IRGraph(name="test", nodes={"in1": node1, "in2": node2, "add": node3}, outputs=["add"])

    class MockOp3:
        def infer_shape(self, *args, **kwargs):
            return (2, 2)

    with mock.patch("ml_switcheroo_compiler.transforms.passes.shape_inference.get_op", return_value=lambda: MockOp3()):
        assert shape_inference_pass(graph) is True
        assert graph.nodes["add"].shape_metadata == (2, 2)


def test_determine_node_shape_keyerror_and_not_found() -> None:
    import unittest.mock as mock

    node1 = IRNode(id="n1", op_type="Add", inputs=["in1", "in2"], attributes={}, shape_metadata=(1, 2))

    # Test KeyError
    with mock.patch("ml_switcheroo_compiler.transforms.passes.shape_inference._infer_op_shape", side_effect=KeyError("boom")):
        assert _determine_node_shape(node1, {}) == (1, 2)

    # Test ValueError "Operation ... not found"
    with mock.patch("ml_switcheroo_compiler.transforms.passes.shape_inference._infer_op_shape", side_effect=ValueError("Operation X not found")):
        assert _determine_node_shape(node1, {}) == (1, 2)

    # Test TypeError
    with mock.patch("ml_switcheroo_compiler.transforms.passes.shape_inference._infer_op_shape", side_effect=TypeError("boom")):
        with pytest.raises(CompilationError):
            _determine_node_shape(node1, {})


def test_shape_inference_keyerror():
    # Unknown op -> KeyError -> returns shape_metadata
    node = IRNode(id="n1", op_type="UnknownOp2", inputs=[], attributes={}, shape_metadata=(10, 20))
    assert _determine_node_shape(node, {}) == (10, 20)


def test_shape_inference_valueerror_not_found(monkeypatch):
    # If something raises ValueError with "Operation ... not found", it returns shape_metadata
    from ml_switcheroo_compiler.transforms.passes import shape_inference

    def mock_infer_op_shape(*args, **kwargs):
        raise ValueError("Operation 'Foo' not found")

    monkeypatch.setattr(shape_inference, "_infer_op_shape", mock_infer_op_shape)

    node = IRNode(id="n1", op_type="Foo", inputs=[], attributes={}, shape_metadata=(30, 40))
    assert _determine_node_shape(node, {}) == (30, 40)


def test_shape_inference_valueerror_other(monkeypatch):
    # If something raises ValueError without "Operation not found", raises CompilationError
    from ml_switcheroo_compiler.transforms.passes import shape_inference

    def mock_infer_op_shape(*args, **kwargs):
        raise ValueError("Some other error")

    monkeypatch.setattr(shape_inference, "_infer_op_shape", mock_infer_op_shape)

    node = IRNode(id="n1", op_type="Foo", inputs=[], attributes={}, shape_metadata=(30, 40))
    with pytest.raises(CompilationError):
        _determine_node_shape(node, {})


def test_shape_inference_keyerror_explicit(monkeypatch):
    from ml_switcheroo_compiler.transforms.passes import shape_inference

    def mock_infer_op_shape(*args, **kwargs):
        raise KeyError("foo")

    monkeypatch.setattr(shape_inference, "_infer_op_shape", mock_infer_op_shape)

    node = IRNode(id="n1", op_type="Foo", inputs=[], attributes={}, shape_metadata=(50, 60))
    assert _determine_node_shape(node, {}) == (50, 60)


def test_annotate_learned_shapes_feedback_loop() -> None:
    """Test annotating runtime observed shapes onto IRGraph nodes."""
    from ml_switcheroo_compiler.transforms.passes.shape_inference import annotate_learned_shapes

    node1 = IRNode(id="in1", op_type="Input", inputs=[], attributes={}, shape_metadata=None)
    node2 = IRNode(id="add", op_type="Add", inputs=["in1"], attributes={}, shape_metadata=None)
    graph = IRGraph(name="learned_test", nodes={"in1": node1, "add": node2}, outputs=["add"])

    # First annotation updates in1
    modified = annotate_learned_shapes(graph, {"in1": (8, 16), "missing_node": (1, 2)})
    assert modified is True
    assert graph.nodes["in1"].shape_metadata == (8, 16)

    # Subsequent annotation with identical shape does not modify
    modified_again = annotate_learned_shapes(graph, {"in1": (8, 16)})
    assert modified_again is False


def test_shape_learning_protocol_models_and_payload() -> None:
    """Test shape learning protocol models and annotate_learned_shapes with ShapeInspectionPayload."""
    from ml_switcheroo_compiler.transforms.passes.config_models import (
        ObservedNodeShape,
        RuntimeTensorMetadata,
        ShapeInspectionPayload,
        load_shape_learning_protocol,
    )
    from ml_switcheroo_compiler.transforms.passes.shape_inference import annotate_learned_shapes

    # Test loading protocol yaml
    protocol = load_shape_learning_protocol()
    assert protocol.protocol_version == "1.0.0"
    assert "webgpu" in protocol.supported_runtimes
    assert "shape_inspection_payload" in protocol.message_types

    import os

    explicit_protocol_path = os.path.join(os.path.dirname(__file__), "..", "..", "..", "src", "ml_switcheroo_compiler", "transforms", "passes", "shape_learning_protocol.yaml")
    if os.path.exists(explicit_protocol_path):
        protocol2 = load_shape_learning_protocol(path=explicit_protocol_path)
        assert protocol2.protocol_version == "1.0.0"

    # Test models
    obs_node = ObservedNodeShape(node_id="x", shape=[2, 8], dtype="float32", strides=[8, 1], offset=0, byte_length=64)
    assert obs_node.node_id == "x"
    assert obs_node.shape == [2, 8]

    rt_meta = RuntimeTensorMetadata(buffer_id="buf_0", element_count=16, is_gradient=False, aliased_to=None, is_contiguous=True)
    assert rt_meta.element_count == 16

    payload = ShapeInspectionPayload(
        runtime="webgpu",
        execution_id="exec_1",
        observed_shapes={"in_node": [4, 16]},
        detailed_nodes={"in_node": obs_node},
        tensor_metadata={"buf_0": rt_meta},
        execution_time_ms=1.5,
        memory_usage_bytes=1024,
    )

    node = IRNode(id="in_node", op_type="Input", shape_metadata=(1, 1))
    graph = IRGraph(name="test_graph", nodes={"in_node": node}, outputs=["in_node"])

    modified = annotate_learned_shapes(graph, payload)
    assert modified is True
    assert graph.nodes["in_node"].shape_metadata == (4, 16)
