"""Tests for IR serialization formats and shape learning protocol."""

from ml_switcheroo_compiler.ir.core import IRGraph, IRNode
from ml_switcheroo_compiler.ir.shape_system import ShapeTracker, SymbolicConstraintTracker, SymInt
from ml_switcheroo_compiler.serialization.ir_format import (
    _build_serialization_dict,
    _parse_serialization_dict,
    graph_to_flatbuffers,
    graph_to_json,
    graph_to_protobuf,
    graph_to_yaml,
    json_to_graph,
    yaml_to_graph,
)


def test_ir_format() -> None:
    """Test JSON and binary format round-tripping."""
    graph = IRGraph(name="test_net")
    inp_node = IRNode(id="x", op_type="Input", inputs=[], shape_metadata=(2, 4))
    node = IRNode(id="n1", op_type="add", inputs=["x", "x"], shape_metadata=(2, 4), attributes={"axis": 1})
    graph.nodes["x"] = inp_node
    graph.nodes["n1"] = node
    graph.outputs = ["n1"]

    j = graph_to_json(graph)
    assert '"n1"' in j
    assert '"add"' in j
    assert '"test_net"' in j

    g2 = json_to_graph(j)
    assert "n1" in g2.nodes
    assert g2.nodes["n1"].op_type == "add"
    assert g2.nodes["n1"].inputs == ["x", "x"]
    assert g2.nodes["n1"].shape_metadata == (2, 4)
    assert g2.nodes["n1"].attributes.get("axis") == 1
    assert g2.outputs == ["n1"]

    # Test YAML serialization round-trip
    y = graph_to_yaml(graph)
    assert "name: test_net" in y
    assert "op_type: add" in y

    gy = yaml_to_graph(y)
    assert "n1" in gy.nodes
    assert gy.nodes["n1"].op_type == "add"
    assert gy.nodes["n1"].shape_metadata == (2, 4)
    assert gy.outputs == ["n1"]

    assert graph_to_protobuf(graph) == b""
    assert graph_to_flatbuffers(graph) == b""


def test_ir_format_serialization_edge_branches() -> None:
    """Verify all branching conditions in _build_serialization_dict and _parse_serialization_dict."""
    # 1. Graph with diverse shape metadata types and non-primitive attributes
    g = IRGraph(name="edge_graph")

    # Node with single int shape
    n_int_shape = IRNode(id="n_int", op_type="Const", shape_metadata=42)  # type: ignore[arg-type]
    # Node with single digit str shape
    n_str_digit_shape = IRNode(id="n_str_digit", op_type="Const", shape_metadata="128")  # type: ignore[arg-type]
    # Node with single non-digit str shape
    n_str_sym_shape = IRNode(id="n_str_sym", op_type="Const", shape_metadata="batch")  # type: ignore[arg-type]
    # Node with tuple containing digit str and non-digit str
    n_tuple_mixed = IRNode(
        id="n_mixed",
        op_type="Op",
        shape_metadata=("64", "unknown_dim", 3),  # type: ignore[arg-type]
        attributes={"str_val": "abc", "complex_val": [1, 2, 3], "num_val": 4.5, "bool_val": True},
    )
    # Node with shape_metadata = None and non-dict attributes
    n_none_shape = IRNode(id="n_none", op_type="Op", shape_metadata=None)
    n_none_shape.attributes = "invalid_attr"  # type: ignore[assignment]

    g.nodes["n_int"] = n_int_shape
    g.nodes["n_str_digit"] = n_str_digit_shape
    g.nodes["n_str_sym"] = n_str_sym_shape
    g.nodes["n_mixed"] = n_tuple_mixed
    g.nodes["n_none"] = n_none_shape

    data = _build_serialization_dict(g)
    assert data["nodes"]["n_int"]["shape_metadata"] == [42]
    assert data["nodes"]["n_str_digit"]["shape_metadata"] == [128]
    assert data["nodes"]["n_str_sym"]["shape_metadata"] == ["batch"]
    assert data["nodes"]["n_mixed"]["shape_metadata"] == [64, "unknown_dim", 3]
    assert data["nodes"]["n_mixed"]["attributes"]["complex_val"] == "[1, 2, 3]"
    assert data["nodes"]["n_none"]["shape_metadata"] == []
    assert data["nodes"]["n_none"]["attributes"] == {}

    # 2. Test deserialization edge cases in _parse_serialization_dict
    raw_dict: dict[str, object] = {
        "name": "parsed_graph",
        "outputs": "not_a_list",  # triggers fallback where outputs is not a list
        "nodes": {
            "invalid_node": "not_a_dict",  # triggers continue branch
            "valid_node_1": {
                "op": "LegacyOp",  # triggers op fallback when op_type missing
                "inputs": "not_a_list",  # triggers inputs fallback
                "shape": ["32", "symbolic", 16],  # triggers shape fallback and digit str/non-digit parsing
                "attributes": {
                    "valid_key": 10,
                    "non_prim": {"nested": "value"},  # triggers str(v) branch
                },
            },
            "valid_node_2": {
                "op_type": "Add",
                "shape_metadata": None,  # raw_shape not a list
                "attributes": "not_a_dict",  # raw_attrs not a dict
            },
        },
    }

    parsed_g = _parse_serialization_dict(raw_dict)
    assert parsed_g.name == "parsed_graph"
    assert parsed_g.outputs == []
    assert "invalid_node" not in parsed_g.nodes
    assert "valid_node_1" in parsed_g.nodes
    assert parsed_g.nodes["valid_node_1"].op_type == "LegacyOp"
    assert parsed_g.nodes["valid_node_1"].inputs == []
    assert parsed_g.nodes["valid_node_1"].shape_metadata == (32, "symbolic", 16)
    assert parsed_g.nodes["valid_node_1"].attributes["valid_key"] == 10
    assert parsed_g.nodes["valid_node_1"].attributes["non_prim"] == "{'nested': 'value'}"

    assert "valid_node_2" in parsed_g.nodes
    assert parsed_g.nodes["valid_node_2"].shape_metadata == ()
    assert parsed_g.nodes["valid_node_2"].attributes == {}

    # 3. Test _parse_serialization_dict when nodes is not a dict
    empty_nodes_dict: dict[str, object] = {"nodes": "not_a_dict"}
    g_empty_nodes = _parse_serialization_dict(empty_nodes_dict)
    assert len(g_empty_nodes.nodes) == 0

    # 4. Test yaml_to_graph with empty / whitespace string
    g_empty_yaml = yaml_to_graph("")
    assert g_empty_yaml.name == "graph"
    assert len(g_empty_nodes.nodes) == 0

    # 5. Test initializers with primitive types and custom object fallback, subgraphs without nodes, and invalid subgraph in deserialization
    class CustomInitValue:
        def __str__(self) -> str:
            return "custom_str_repr"

    g_init_edge = IRGraph(name="init_edge")
    g_init_edge.initializers = {
        "int_val": 123,
        "custom_obj": CustomInitValue(),
    }
    # Node with subgraphs containing non-graph item
    n_with_bad_sub = IRNode(
        id="n_bad_sub",
        op_type="CustomOp",
        subgraphs={"invalid_sub": "not_a_subgraph_with_nodes"},  # type: ignore[dict-item]
    )
    g_init_edge.nodes["n_bad_sub"] = n_with_bad_sub
    g_init_edge.inputs = None  # type: ignore[assignment]  # triggers branch 65 where getattr(graph, 'inputs', None) is falsy

    init_edge_data = _build_serialization_dict(g_init_edge)
    assert init_edge_data["initializers"]["int_val"] == 123
    assert init_edge_data["initializers"]["custom_obj"] == "custom_str_repr"
    assert "invalid_sub" not in init_edge_data["nodes"]["n_bad_sub"].get("subgraphs", {})

    # Deserialization with node containing invalid non-dict sub_data or non-dict subgraphs field
    raw_bad_sub_data: dict[str, object] = {
        "name": "g_bad_sub",
        "inputs": "not_a_list",
        "initializers": "not_a_dict",
        "nodes": {
            "node_sub": {
                "op_type": "Identity",
                "subgraphs": {
                    "bad_sub_1": "string_not_dict",
                },
            },
            "node_sub_not_dict": {
                "op_type": "Identity",
                "subgraphs": "not_a_dict_subgraph_field",
            },
        },
    }
    g_parsed_bad_sub = _parse_serialization_dict(raw_bad_sub_data)
    assert "node_sub" in g_parsed_bad_sub.nodes
    assert len(g_parsed_bad_sub.nodes["node_sub"].subgraphs) == 0
    assert "node_sub_not_dict" in g_parsed_bad_sub.nodes


def test_shape_tracker_feedback_and_dynamic_bounds() -> None:
    """Test bidirectional shape learning and SymInt dynamic bounds resolution."""
    tracker = SymbolicConstraintTracker()
    batch_sym = SymInt("batch_dim")

    graph = IRGraph(name="symbolic_net")
    inp_node = IRNode(id="x", op_type="Input", inputs=[], shape_metadata=(batch_sym, 16))
    matmul_node = IRNode(id="dense", op_type="Matmul", inputs=["x", "w"], shape_metadata=(batch_sym, 32))
    graph.nodes["x"] = inp_node
    graph.nodes["dense"] = matmul_node

    # Simulated feedback telemetry observed from browser WASM/WebGPU runtime
    telemetry = {
        "shapes": {
            "x": [4, 16],
            "dense": [4, 32],
        }
    }

    resolved_shapes = ShapeTracker.update_from_feedback(telemetry, tracker)
    assert resolved_shapes["x"] == (4, 16)
    assert resolved_shapes["dense"] == (4, 32)

    # Resolve dynamic bounds across graph
    resolved_bounds = ShapeTracker.resolve_dynamic_bounds(graph, telemetry, tracker)
    assert "batch_dim" in resolved_bounds
    assert resolved_bounds["batch_dim"] == 4

    # Assert node shapes are statically resolved to concrete tuples
    assert graph.nodes["x"].shape_metadata == (4, 16)
    assert graph.nodes["dense"].shape_metadata == (4, 32)


def test_ir_format_initializers_and_subgraphs() -> None:
    """Verify serialization and deserialization of initializers and nested subgraphs."""
    import numpy as np

    sub = IRGraph(name="then_branch")
    sub.inputs = ["sub_in"]
    sub.nodes["sub_in"] = IRNode(id="sub_in", op_type="Input")
    sub.nodes["sub_out"] = IRNode(id="sub_out", op_type="Relu", inputs=["sub_in"])
    sub.outputs = ["sub_out"]

    graph = IRGraph(name="init_subgraph_net")
    graph.inputs = ["x"]
    graph.initializers["weight"] = np.array([1.5, 2.5])
    graph.nodes["x"] = IRNode(id="x", op_type="Input")
    graph.nodes["if_node"] = IRNode(id="if_node", op_type="If", inputs=["x"], subgraphs={"then_branch": sub})
    graph.outputs = ["if_node"]

    serialized_json = graph_to_json(graph)
    assert "initializers" in serialized_json
    assert "weight" in serialized_json
    assert "then_branch" in serialized_json

    deserialized = json_to_graph(serialized_json)
    assert deserialized.inputs == ["x"]
    assert deserialized.outputs == ["if_node"]
    assert "weight" in deserialized.initializers
    assert deserialized.initializers["weight"] == [1.5, 2.5]
    assert "then_branch" in deserialized.nodes["if_node"].subgraphs
    sub_deserialized = deserialized.nodes["if_node"].subgraphs["then_branch"]
    assert sub_deserialized.name == "then_branch"
    assert "sub_out" in sub_deserialized.nodes
