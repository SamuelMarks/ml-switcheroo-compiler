# ruff: noqa: E501
"""Tests for shape inference pass and dynamic runtime shape learning."""

from typing import Any

import pytest

from ml_switcheroo_compiler.core.errors import CompilationError
from ml_switcheroo_compiler.ir.core import IRGraph, IRNode
from ml_switcheroo_compiler.transforms.passes.shape_inference import (
    _determine_node_shape,
    _infer_constant_shape,
    _infer_op_shape,
    _infer_output_shape,
    _normalize_shape_tuple,
    _prepare_op_kwargs,
    shape_inference_pass,
)


def test_infer_constant_shape() -> None:
    """Verify shape inference for Constant IR nodes."""
    node = IRNode(id="n1", op_type="Constant", inputs=[], attributes={"value": [1.0, 2.0]})
    assert _infer_constant_shape(node, {}) == (2,)


def test_infer_output_shape() -> None:
    """Verify shape inference for Output IR nodes."""
    node1 = IRNode(id="n1", op_type="Output", inputs=["in1"], attributes={})
    assert _infer_output_shape(node1, {"in1": (3, 4)}) == (3, 4)
    node2 = IRNode(id="n2", op_type="Output", inputs=[], attributes={})
    assert _infer_output_shape(node2, {}) is None


def test_prepare_op_kwargs() -> None:
    """Verify preparation of keyword arguments for op shape inference."""
    node1 = IRNode(id="n1", op_type="Expand", inputs=[], attributes={"other": 1}, shape_metadata=(1, 2))
    assert _prepare_op_kwargs(node1) == {"other": 1, "shape": (1, 2)}
    node2 = IRNode(id="n2", op_type="BroadcastTo", inputs=[], attributes={}, shape_metadata=(3, 3))
    assert _prepare_op_kwargs(node2) == {"shape": (3, 3)}
    node3 = IRNode(id="n3", op_type="Reshape", inputs=[], attributes={}, shape_metadata=(4,))
    assert _prepare_op_kwargs(node3) == {"newshape": (4,)}
    node4 = IRNode(id="n4", op_type="Add", inputs=[], attributes={"other": 1}, shape_metadata=(1,))
    assert _prepare_op_kwargs(node4) == {"other": 1}


def test_infer_op_shape() -> None:
    """Verify shape inference delegation to registered Op classes."""
    import unittest.mock as mock

    node1 = IRNode(id="n1", op_type="Add", inputs=["in1", "in2"], attributes={})
    shapes = {"in1": (1, 2), "in2": (1, 2)}

    class MockOp:
        """Mock op implementation."""

        def infer_shape(self, *args: Any, **kwargs: Any) -> tuple[int, ...]:
            """Return static shape."""
            return (1, 2)

    with mock.patch("ml_switcheroo_compiler.transforms.passes.shape_inference.get_op", return_value=lambda: MockOp()):
        assert _infer_op_shape(node1, shapes) == (1, 2)


def test_determine_node_shape() -> None:
    """Verify node shape determination across Input, Unknown, and Error cases."""
    import unittest.mock as mock

    node1 = IRNode(id="n1", op_type="Input", inputs=[], attributes={}, shape_metadata=(5, 5))
    assert _determine_node_shape(node1, {}) == (5, 5)
    node2 = IRNode(id="n2", op_type="UnknownOp", inputs=[], attributes={}, shape_metadata=(1,))
    assert _determine_node_shape(node2, {}) == (1,)
    node3 = IRNode(id="n3", op_type="Add", inputs=["in1", "in2"], attributes={})

    class MockOp2:
        """Mock op that raises error."""

        def infer_shape(self, *args: Any, **kwargs: Any) -> tuple[int, ...]:
            """Raise error."""
            raise ValueError("boom")

    with mock.patch("ml_switcheroo_compiler.transforms.passes.shape_inference.get_op", return_value=lambda: MockOp2()):
        with pytest.raises(CompilationError):
            _determine_node_shape(node3, {"in1": (1, 2), "in2": (3, 4)})


def test_shape_inference_pass() -> None:
    """Verify end-to-end topological shape inference pass execution."""
    import unittest.mock as mock

    node1 = IRNode(id="in1", op_type="Input", inputs=[], attributes={}, shape_metadata=(2, 2))
    node2 = IRNode(id="in2", op_type="Input", inputs=[], attributes={}, shape_metadata=(2, 2))
    node3 = IRNode(id="add", op_type="Add", inputs=["in1", "in2"], attributes={}, shape_metadata=None)
    graph = IRGraph(name="test", nodes={"in1": node1, "in2": node2, "add": node3}, outputs=["add"])

    class MockOp3:
        """Mock op for pass execution."""

        def infer_shape(self, *args: Any, **kwargs: Any) -> tuple[int, ...]:
            """Return static shape."""
            return (2, 2)

    with mock.patch("ml_switcheroo_compiler.transforms.passes.shape_inference.get_op", return_value=lambda: MockOp3()):
        assert shape_inference_pass(graph) is True
        assert graph.nodes["add"].shape_metadata == (2, 2)


def test_determine_node_shape_keyerror_and_not_found() -> None:
    """Verify fallback when op shape raises KeyError or not found."""
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


def test_shape_inference_keyerror() -> None:
    """Verify handling of unknown op types via KeyError fallback."""
    # Unknown op -> KeyError -> returns shape_metadata
    node = IRNode(id="n1", op_type="UnknownOp2", inputs=[], attributes={}, shape_metadata=(10, 20))
    assert _determine_node_shape(node, {}) == (10, 20)


def test_shape_inference_valueerror_not_found(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify operation not found ValueError fallback to shape_metadata."""
    from ml_switcheroo_compiler.transforms.passes import shape_inference

    def mock_infer_op_shape(*args: Any, **kwargs: Any) -> None:
        """Mock raising not found."""
        raise ValueError("Operation 'Foo' not found")

    monkeypatch.setattr(shape_inference, "_infer_op_shape", mock_infer_op_shape)

    node = IRNode(id="n1", op_type="Foo", inputs=[], attributes={}, shape_metadata=(30, 40))
    assert _determine_node_shape(node, {}) == (30, 40)


def test_shape_inference_valueerror_other(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify unexpected ValueError raises CompilationError."""
    from ml_switcheroo_compiler.transforms.passes import shape_inference

    def mock_infer_op_shape(*args: Any, **kwargs: Any) -> None:
        """Mock raising other error."""
        raise ValueError("Some other error")

    monkeypatch.setattr(shape_inference, "_infer_op_shape", mock_infer_op_shape)

    node = IRNode(id="n1", op_type="Foo", inputs=[], attributes={}, shape_metadata=(30, 40))
    with pytest.raises(CompilationError):
        _determine_node_shape(node, {})


def test_shape_inference_keyerror_explicit(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify explicit KeyError returns shape_metadata."""
    from ml_switcheroo_compiler.transforms.passes import shape_inference

    def mock_infer_op_shape(*args: Any, **kwargs: Any) -> None:
        """Mock raising KeyError."""
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


def test_record_runtime_observation_and_packet() -> None:
    """Verify recording runtime observations with validation, packets, and history."""
    from ml_switcheroo_compiler.core.errors import ShapeMismatchError
    from ml_switcheroo_compiler.transforms.passes.config_models import (
        ObservedNodeShape,
        RuntimeShapePacket,
    )
    from ml_switcheroo_compiler.transforms.passes.shape_inference import (
        annotate_learned_shapes,
        get_shape_learning_protocol_config,
        record_runtime_observation,
    )

    proto = get_shape_learning_protocol_config()
    assert proto.protocol_version == "1.0.0"

    node = IRNode(id="n1", op_type="Input", shape_metadata=(4, 8))
    graph = IRGraph(name="test_obs", nodes={"n1": node}, outputs=["n1"])

    # Valid observation matching current shape
    record_runtime_observation(graph, "n1", [4, 8], observed_dtype="float32")
    assert graph.nodes["n1"].shape_metadata == (4, 8)
    assert hasattr(graph, "annotations")
    assert len(graph.annotations["learned_shapes_history"]) == 1

    # Missing node raises KeyError
    with pytest.raises(KeyError, match="not found in computation graph"):
        record_runtime_observation(graph, "missing_node", [4, 8])

    # Rank mismatch raises ShapeMismatchError
    with pytest.raises(ShapeMismatchError, match="Rank mismatch"):
        record_runtime_observation(graph, "n1", [4, 8, 2])

    # Dynamic node can be resolved
    dynamic_node = IRNode(id="n2", op_type="Input", shape_metadata=(-1, 16))
    graph.nodes["n2"] = dynamic_node
    record_runtime_observation(graph, "n2", [2, 16])
    assert graph.nodes["n2"].shape_metadata == (2, 16)

    # Annotate via RuntimeShapePacket
    packet = RuntimeShapePacket(
        packet_id="pkt_001",
        runtime="webgpu",
        graph_name="test_obs",
        observations=[
            ObservedNodeShape(node_id="n1", shape=[4, 8], dtype="float32"),
            ObservedNodeShape(node_id="n2", shape=[4, 16], dtype="float32"),
        ],
        execution_time_ms=2.5,
        memory_usage_bytes=4096,
    )
    modified = annotate_learned_shapes(graph, packet)
    assert modified is True
    assert graph.nodes["n2"].shape_metadata == (4, 16)
    assert graph.annotations.get("convergence_converged") is True


def test_multipass_dynamic_shape_refinement_and_conflicts() -> None:
    """Verify multi-pass shape refinement, downstream propagation, and conflict handling."""
    from ml_switcheroo_compiler.core.errors import ShapeMismatchError
    from ml_switcheroo_compiler.ir.shape_system import ShapeTracker
    from ml_switcheroo_compiler.transforms.passes.config_models import (
        ObservedNodeShape,
        RuntimeShapePacket,
    )
    from ml_switcheroo_compiler.transforms.passes.shape_inference import (
        annotate_learned_shapes,
        record_runtime_observation,
        shape_inference_pass,
    )

    # 1. Build graph with dynamic input feeding downstream op
    inp = IRNode(id="inp", op_type="Input", shape_metadata=(-1, 32))
    weight = IRNode(id="weight", op_type="Constant", inputs=[], attributes={"value": [1.0] * 32}, shape_metadata=(32,))
    add_node = IRNode(id="add", op_type="Add", inputs=["inp", "weight"], shape_metadata=None)
    graph = IRGraph(name="dyn_refine", nodes={"inp": inp, "weight": weight, "add": add_node}, outputs=["add"])

    # First pass: shape_metadata for add is inferred with dynamic shape
    shape_inference_pass(graph)

    # Runtime observation packet arrives from browser/edge runner
    packet_pass1 = RuntimeShapePacket(
        packet_id="cycle_1",
        runtime="webgpu",
        graph_name="dyn_refine",
        observations=[
            ObservedNodeShape(node_id="inp", shape=[8, 32], dtype="float32"),
        ],
    )
    modified = annotate_learned_shapes(graph, packet_pass1)
    assert modified is True
    assert graph.nodes["inp"].shape_metadata == (8, 32)
    assert graph.annotations["convergence_converged"] is True

    # Multi-pass: second observation with different batch size
    packet_pass2 = RuntimeShapePacket(
        packet_id="cycle_2",
        runtime="webgpu",
        graph_name="dyn_refine",
        observations=[
            ObservedNodeShape(node_id="inp", shape=[16, 32], dtype="float32"),
        ],
    )
    modified2 = annotate_learned_shapes(graph, packet_pass2)
    assert modified2 is True
    assert graph.nodes["inp"].shape_metadata == (16, 32)

    # Rank mismatch conflict
    with pytest.raises(ShapeMismatchError, match="Rank mismatch"):
        record_runtime_observation(graph, "inp", [16, 32, 1])

    # ShapeTracker telemetry resolution integration
    telemetry = {"inp": [16, 32], "add": [16, 32]}
    resolved = ShapeTracker.update_from_feedback(telemetry)
    assert resolved["inp"] == (16, 32)
    assert resolved["add"] == (16, 32)


def test_shape_inference_annotation_branches() -> None:
    """Test annotate_learned_shapes when annotations is missing or non-dict, and non-dict shapes."""
    from ml_switcheroo_ir import LogicalGraph, LogicalNode

    from ml_switcheroo_compiler.transforms.passes.shape_inference import annotate_learned_shapes

    graph = LogicalGraph(name="test_graph")
    node = LogicalNode(id="n1", op_type="Relu", shape_metadata=(4, 4))
    graph.nodes[node.id] = node
    graph.annotations = "invalid_annotations"
    res1 = annotate_learned_shapes(graph, {"n1": (4, 4)})
    assert res1 is False
    assert isinstance(graph.annotations, dict)
    assert graph.annotations.get("convergence_converged") is True
    res2 = annotate_learned_shapes(graph, None)
    assert res2 is False


def test_shape_inference_pass_attribute_error_and_none_branches() -> None:
    """Test shape_inference_pass with unhandled op (None shape) and node raising AttributeError on setting strides."""
    from unittest.mock import patch

    from ml_switcheroo_ir import LogicalNode

    from ml_switcheroo_compiler.ir.core import IRGraph
    from ml_switcheroo_compiler.transforms.passes.shape_inference import shape_inference_pass

    class NodeWithAttributeErrorStrides(LogicalNode):
        """Node whose strides attribute raises AttributeError on set."""

        @property
        def strides(self) -> tuple[int, ...]:
            """Return empty strides.

            Returns:
                tuple[int, ...]: Strides tuple.
            """
            return ()

        @strides.setter
        def strides(self, value: tuple[int, ...]) -> None:
            """Raise AttributeError on set.

            Args:
                value (tuple[int, ...]): New strides.

            Raises:
                AttributeError: Always raised.
            """
            raise AttributeError("Read only strides")

    node_none = LogicalNode(id="n_none", op_type="UnknownOpForNoneShape", inputs=[], attributes={}, shape_metadata=None)
    node_custom = NodeWithAttributeErrorStrides(id="n_custom", op_type="Input", inputs=[], attributes={}, shape_metadata=(2, 3))
    graph = IRGraph(name="test_attr_err", nodes={"n_none": node_none, "n_custom": node_custom}, outputs=["n_custom"])
    with patch("ml_switcheroo_compiler.ops.shape_inference.infer_shape", side_effect=[None, (2, 3)]):
        modified = shape_inference_pass(graph)
        assert isinstance(modified, bool)


def test_normalize_shape_edge_cases() -> None:
    """Verify edge cases of _normalize_shape_tuple utility."""
    assert _normalize_shape_tuple(None) is None
    assert _normalize_shape_tuple([1, 2, 3]) == (1, 2, 3)
    assert _normalize_shape_tuple((4, 5)) == (4, 5)
    assert _normalize_shape_tuple(["invalid", "str"]) is None
    assert _normalize_shape_tuple(12345) is None
    assert _normalize_shape_tuple("not_a_list") is None
    assert _normalize_shape_tuple([object()]) is None


def test_shape_inference_subgraphs() -> None:
    """Verify shape propagation across subgraph boundary ports."""
    from ml_switcheroo_ir import LogicalGraph, LogicalNode

    sub = LogicalGraph(name="then_sub")
    sub.inputs = ["sub_in"]
    sub.nodes["sub_in"] = LogicalNode(id="sub_in", op_type="Input")
    sub.nodes["sub_add"] = LogicalNode(id="sub_add", op_type="Add", inputs=["sub_in", "sub_in"])
    sub.outputs = ["sub_add"]

    parent = LogicalGraph(name="parent_graph")
    parent.inputs = ["x"]
    parent.nodes["x"] = LogicalNode(id="x", op_type="Input", shape_metadata=(4, 8))
    parent.nodes["if_node"] = LogicalNode(
        id="if_node",
        op_type="If",
        inputs=["x"],
        subgraphs={"then_branch": sub},
    )
    parent.outputs = ["if_node"]

    modified = shape_inference_pass(parent)
    assert modified is True
    assert sub.nodes["sub_in"].shape_metadata == (4, 8)
    assert sub.nodes["sub_add"].shape_metadata == (4, 8)
    assert parent.nodes["if_node"].shape_metadata == (4, 8)


def test_shape_inference_subgraph_branches_exhaustive() -> None:
    """Test all branches in shape_inference_pass subgraph handling.

    Covers:
    - subgraphs without nodes / inputs attributes
    - parent_inp not in shapes or shapes[parent_inp] is None
    - inp_id not in sub.nodes
    - shape_inference_pass(sub) returns False
    - out_shape is None and inferring out_shape from sub.outputs
    - sub without outputs or empty outputs or sub.outputs[0] not in sub.nodes
    - sub.outputs[0] with None or valid shape_metadata
    """
    from ml_switcheroo_ir import LogicalGraph, LogicalNode

    # 1. Subgraph with no outputs, subgraph not a graph, etc.
    empty_sub = LogicalGraph(name="empty_sub")
    empty_sub.inputs = ["missing_in"]
    # missing_in not in empty_sub.nodes

    sub_no_out = LogicalGraph(name="sub_no_out")
    sub_no_out.inputs = ["in_p"]
    sub_no_out.nodes["in_p"] = LogicalNode(id="in_p", op_type="Input")
    sub_no_out.outputs = []  # empty outputs

    sub_missing_out_node = LogicalGraph(name="sub_missing_out")
    sub_missing_out_node.outputs = ["nonexistent_out_id"]

    sub_none_out_shape = LogicalGraph(name="sub_none_shape")
    sub_none_out_shape.outputs = ["out_n"]
    sub_none_out_shape.nodes["out_n"] = LogicalNode(id="out_n", op_type="Identity", shape_metadata=None)

    sub_valid = LogicalGraph(name="sub_valid")
    sub_valid.outputs = ["out_valid"]
    sub_valid.nodes["out_valid"] = LogicalNode(id="out_valid", op_type="Identity", shape_metadata=(5, 6))

    parent = LogicalGraph(name="parent")
    # parent input with None shape and with concrete shape
    parent.inputs = ["none_input", "untracked_input", "concrete_input"]
    parent.nodes["none_input"] = LogicalNode(id="none_input", op_type="Input", shape_metadata=None)
    parent.nodes["concrete_input"] = LogicalNode(id="concrete_input", op_type="Input", shape_metadata=(2, 3))

    # empty_sub has sub.inputs = ["missing_in"]. Wire parent input "concrete_input" to it.
    # parent_inp is "concrete_input" (which is in shapes and not None),
    # but "missing_in" is NOT in empty_sub.nodes. This tests branch 176 -> 174 (the False branch of `if inp_id in sub.nodes`).
    # sub_no_out has sub.inputs = ["in_p"]. Wire parent input "none_input" (in shapes, but shapes[parent_inp] is None) and
    # "unknown_parent_inp" (not in shapes) to hit the False branch of `if parent_inp in shapes and shapes[parent_inp] is not None:` (175 -> 174).
    sub_no_out.inputs = ["in_p", "in_p2"]

    # Node with non-graph subgraph, empty_sub, sub_no_out, sub_missing_out_node, sub_none_out_shape, sub_valid
    parent.nodes["complex_node"] = LogicalNode(
        id="complex_node",
        op_type="UnknownCustomOp",
        inputs=["concrete_input", "none_input", "untracked_input"],
        subgraphs={
            "not_graph": "plain_string_attr",  # not having nodes/inputs
            "empty_sub": empty_sub,
            "sub_no_out": sub_no_out,
            "sub_missing_out": sub_missing_out_node,
            "sub_none": sub_none_out_shape,
            "sub_valid": sub_valid,
        },
    )
    parent.outputs = ["complex_node"]

    modified = shape_inference_pass(parent)
    assert modified is True
    assert parent.nodes["complex_node"].shape_metadata == (5, 6)
