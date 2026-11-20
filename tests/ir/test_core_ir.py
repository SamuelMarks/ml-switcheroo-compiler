"""Tests for compiler-local IR narrowing utilities and Tier-1 protocol interoperability."""

from __future__ import annotations

import json
from collections.abc import Mapping
from types import MappingProxyType

import pytest
from ml_switcheroo_ir import (
    CyclicGraphError,
    DeviceLikeProtocol,
    DTypeLikeProtocol,
    IRGraph,
    IRNode,
    LogicalGraph,
    LogicalNode,
    NoTangent,
    PartitionSpec,
    SymbolicIntWrapperProtocol,
    SymbolicNodeProtocol,
    TensorShape,
    TensorSpec,
    ZeroTangent,
    topological_sort,
)

from ml_switcheroo_compiler.core.device import Device, DeviceType
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.ir import SymInt
from ml_switcheroo_compiler.ir.core import (
    as_node_list,
    as_opt_graph,
    as_opt_pspec,
    clone_logical_node,
    ensure_attr_val,
    ensure_attrs,
    ensure_graph,
    normalize_shape_metadata,
)


def test_clone_logical_node() -> None:
    """Verify clone_logical_node preserves and overrides fields without mutating original."""
    subgraph = LogicalGraph(name="sub1")
    spec = TensorSpec(shape=(2, 3), dtype=DType.float32)
    node = LogicalNode(
        id="id1",
        op_type="op1",
        domain="ai.onnx",
        version=1,
        attributes={"attr1": "val1"},
        inputs=["in1"],
        outputs=["out1"],
        dtype=DType.float32,
        output_specs=[spec],
        subgraphs={"body": subgraph},
        device="cuda:0",
        stream="stream_0",
        strides=(3, 1),
    )
    node2 = clone_logical_node(node, op_type="op2", device="cpu")
    assert node2.op_type == "op2"
    assert node2.id == "id1"
    assert node2.device == "cpu"
    assert node2.stream == "stream_0"
    assert node2.dtype is not None and node2.dtype.value == "float32"
    assert node2.strides == (3, 1)
    assert len(node2.output_specs) == 1
    assert "body" in node2.subgraphs

    node2.attributes["attr1"] = "mutated"
    assert node.attributes["attr1"] == "val1"

    node_min = LogicalNode(id="min1", op_type="min_op", output_specs=[])
    cloned_min = clone_logical_node(node_min, id="min2")
    assert cloned_min.id == "min2"
    assert cloned_min.op_type == "min_op"
    assert cloned_min.output_specs == []

    nested_g = LogicalGraph(name="nested_attr_g", nodes=[LogicalNode(id="ng_x", op_type="Input")])
    nested_n = LogicalNode(id="nested_attr_n", op_type="Relu")
    node_nested = LogicalNode(
        id="parent_n",
        op_type="HigherOrder",
        attributes={
            "graph": nested_g,
            "node": nested_n,
            "mapping": {"k": [1, (2, 3)]},
        },
    )
    cloned_nested = clone_logical_node(node_nested)
    assert isinstance(cloned_nested.attributes["graph"], LogicalGraph)
    assert cloned_nested.attributes["graph"] is not nested_g
    assert isinstance(cloned_nested.attributes["node"], LogicalNode)
    assert cloned_nested.attributes["node"] is not nested_n

    from ml_switcheroo_compiler.core.shape import extract_shape

    explicit_override = clone_logical_node(node_nested, attributes={"explicit": 9})
    assert explicit_override.attributes == {"explicit": 9}
    assert extract_shape(LogicalNode(id="none_meta", op_type="Input", shape_metadata=None), default=(7,)) == (7,)


def test_tier1_structural_protocol_interoperability() -> None:
    """Verify compiler DType, Device, and SymInt satisfy Tier-1 ml_switcheroo_ir protocols."""
    assert IRNode is LogicalNode
    assert IRGraph is LogicalGraph

    dev_enum = Device(DeviceType.GPU, 1)
    dev_str = Device("cpu", 0)
    assert dev_enum.type == "gpu"
    assert dev_str.type == "cpu"
    assert isinstance(dev_enum, DeviceLikeProtocol)
    assert isinstance(DType.float32, DTypeLikeProtocol)

    sym_dim = SymInt("B") + 2
    assert isinstance(sym_dim, SymbolicIntWrapperProtocol)
    assert isinstance(sym_dim.node, SymbolicNodeProtocol)

    ts = TensorShape([4, "N", sym_dim, sym_dim.node])
    assert ts.rank == 4
    assert ts.is_dynamic is True
    assert ts.evaluate({"N": 8, "B": 6}) == (4, 8, 8, 8)

    spec = TensorSpec(shape=ts, dtype=DType.float32, sparsity="coo")
    eval_spec = spec.evaluate({"N": 10, "B": 2})
    assert eval_spec.shape == TensorShape((4, 10, 4, 4))
    assert eval_spec.dtype.value == "float32"
    assert eval_spec.sparsity == "coo"

    zt = ZeroTangent("t1", shape_metadata=(2, 2))
    nt = NoTangent("t2")
    assert zt.op_type == "ZeroTangent"
    assert nt.op_type == "NoTangent"


def types_mapping_proxy() -> Mapping[str, int]:
    """Create an immutable MappingProxyType for testing Mapping branches in `ensure_attrs`.

    Returns:
        Mapping[str, int]: Read-only mapping proxy with sample entry.
    """
    return MappingProxyType({"m": 7})


def test_ir_core_narrowing_helpers() -> None:
    """Verify `ensure_graph`, `as_opt_graph`, `ensure_attrs`, `ensure_attr_val`, `as_node_list`, and `as_opt_pspec`."""
    g = LogicalGraph(name="g_helper")
    n = LogicalNode(id="n_helper", op_type="Relu")
    ps = PartitionSpec(("data",))

    assert ensure_graph(g) is g
    assert ensure_graph(None).nodes == {}
    with pytest.raises(TypeError, match="Expected LogicalGraph"):
        ensure_graph(42)
    assert as_opt_graph(g) is g
    assert as_opt_graph(None) is None
    assert ensure_attrs({"k": 1}) == {"k": 1}
    assert ensure_attrs(types_mapping_proxy()) == {"m": 7}
    assert ensure_attrs(None) == {}
    assert ensure_attr_val(42) == 42
    assert ensure_attr_val((1, [2, {"a": 3}])) == (1, [2, {"a": 3}])
    assert ensure_attr_val(complex(1, 2)) == "(1+2j)"

    def _custom_cb(x: int) -> int:
        """Increment scalar input.

        Args:
            x (int): Input scalar.

        Returns:
            int: Incremented scalar.
        """
        return x + 1

    cb_attr = ensure_attr_val(_custom_cb)
    assert cb_attr != ""
    assert ensure_attr_val(int) == "<class 'int'>"

    assert as_node_list({"n": n}) == [n]
    assert as_node_list([n]) == [n]
    assert as_node_list((n,)) == [n]
    assert as_node_list("not_nodes") == []
    assert as_opt_pspec(ps) is ps
    assert as_opt_pspec(None) is None
    assert normalize_shape_metadata(None) is None
    assert normalize_shape_metadata((2, None, "N")) == (2, -1, "N")


def test_dtype_from_str_and_aliases() -> None:
    """Verify `DType.from_str` normalizes framework prefixes, aliases, and unknown errors."""
    assert DType.from_str("torch.float32") == DType.float32
    assert DType.from_str("np.f64") == DType.float64
    assert DType.from_str("bf16") == DType.bfloat16
    assert DType.from_str("Float16") == DType.float16
    with pytest.raises(ValueError, match="Unknown or unsupported DType"):
        DType.from_str("not_a_dtype")


def test_logical_graph_fields_and_cycle_detection() -> None:
    """Verify `LogicalGraph` sequence init, `strides`, `annotations`, `metadata`, `sorted_nodes`, and cycle check."""
    n_in = LogicalNode(id="x", op_type="Input", shape_metadata=(2, 3), dtype=DType.float32)
    n_add = LogicalNode(id="add", op_type="Add", inputs=["x", "x"], strides=(3, 1))
    assert n_add.strides == (3, 1)
    n_add.strides = [6, 2]
    assert n_add.strides == (6, 2)

    g = LogicalGraph(
        "SeqGraph",
        [n_in, n_add],
        annotations={"stage": "fwd"},
        metadata={"pass": "init"},
        sorted_nodes=[n_in, n_add],
    )
    assert g.annotations == {"stage": "fwd"}
    assert g.metadata == {"pass": "init"}
    assert g.sorted_nodes == [n_in, n_add]
    parsed = json.loads(g.to_json())
    assert parsed["name"] == "SeqGraph"

    c1 = LogicalNode(id="c1", op_type="Identity", inputs=["c2"])
    c2 = LogicalNode(id="c2", op_type="Identity", inputs=["c1"])
    g_cyc = LogicalGraph("Cycle", [c1, c2])
    assert len(topological_sort(g_cyc, strict=False)) == 2
    with pytest.raises(CyclicGraphError):
        topological_sort(g_cyc, strict=True)
