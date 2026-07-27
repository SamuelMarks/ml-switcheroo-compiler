# ruff: noqa: E501
from unittest.mock import MagicMock

import pytest
from ml_switcheroo_ir import LogicalGraph

from ml_switcheroo_compiler.transforms.foreign import _handle_fx_call_function, _handle_fx_output, _handle_fx_placeholder, _translate_jax_equation, ingest_jaxpr, ingest_torch_fx


def test_handle_fx_placeholder():
    graph = LogicalGraph()
    node_map = {}
    node = MagicMock(name="ph")
    _handle_fx_placeholder(node, graph, node_map)
    assert node.name in node_map


def test_handle_fx_call_function():
    graph = LogicalGraph()
    node_map = {"a": "a_mapped"}
    node = MagicMock()
    node.name = "out"
    node.target = MagicMock(__name__="add")
    arg1 = MagicMock(name="arg1")
    arg1.name = "a"
    arg2 = MagicMock(name="arg2")
    arg2.name = "b"
    node.args = [arg1, arg2]
    _handle_fx_call_function(node, graph, node_map)
    assert node.name in graph.nodes
    assert graph.nodes[node.name].op_type == "Add"
    assert graph.nodes[node.name].inputs == ["a_mapped", str(arg2)]
    node.target = "mul"
    _handle_fx_call_function(node, graph, node_map)
    assert graph.nodes[node.name].op_type == "Mul"
    node.target = "unknown_target"
    _handle_fx_call_function(node, graph, node_map)
    assert graph.nodes[node.name].op_type == "Unknown"


def test_handle_fx_output():
    graph = LogicalGraph()
    node_map = {}
    node = MagicMock()
    arg1 = MagicMock()
    arg1.name = "out1"
    arg2 = MagicMock()
    del arg2.name
    node.args = [(arg1, arg2)]
    _handle_fx_output(node, graph, node_map)
    assert graph.outputs == ["out1"]
    node.args = [arg1]
    _handle_fx_output(node, graph, node_map)
    assert graph.outputs == ["out1"]

    # Test args not a tuple and no name
    no_name_arg = MagicMock()
    del no_name_arg.name
    node.args = [no_name_arg]
    _handle_fx_output(node, graph, node_map)
    assert graph.outputs == []  # It resets graph.outputs to empty


def test_ingest_torch_fx():
    with pytest.raises(ValueError):
        ingest_torch_fx(None)
    gm = MagicMock()
    del gm.graph
    g = ingest_torch_fx(gm)
    assert g.name == "torch_fx_ingested"
    gm = MagicMock()
    node1 = MagicMock()
    node1.op = "placeholder"
    node1.name = "p"
    node2 = MagicMock()
    node2.op = "call_function"
    node2.target = "add"
    node2.name = "a"
    node2.args = [node1]
    node3 = MagicMock()
    node3.op = "output"
    node3.args = [node2]
    node4 = MagicMock()
    node4.op = "unknown"
    gm.graph.nodes = [node1, node2, node3, node4]
    g = ingest_torch_fx(gm)
    assert "p" in g.nodes or "a" in g.nodes
    assert g.outputs == ["a"]

    # Test with no nodes
    gm.graph.nodes = []
    g3 = ingest_torch_fx(gm)
    assert len(g3.nodes) == 0


def test_translate_jax_equation():
    graph = LogicalGraph()
    eqn = MagicMock()
    eqn.primitive.name = "add"
    eqn.invars = [MagicMock(), MagicMock()]
    eqn.outvars = [MagicMock()]
    _translate_jax_equation(eqn, graph)
    out_id = str(id(eqn.outvars[0]))
    assert out_id in graph.nodes
    assert graph.nodes[out_id].op_type == "Add"
    eqn.primitive.name = "mul"
    eqn.outvars = []
    _translate_jax_equation(eqn, graph)
    assert "out" in graph.nodes
    assert graph.nodes["out"].op_type == "Mul"

    # Test unknown primitive
    eqn.primitive.name = "unknown"
    _translate_jax_equation(eqn, graph)
    assert graph.nodes["out"].op_type == "Unknown"


def test_ingest_jaxpr():
    with pytest.raises(ValueError):
        ingest_jaxpr(None)
    jaxpr = MagicMock()
    jaxpr.consts = [1.0, 2.0]
    jaxpr.constvars = [MagicMock(), MagicMock()]
    eqn = MagicMock()
    eqn.primitive.name = "add"
    eqn.invars = []
    eqn.outvars = [MagicMock()]
    jaxpr.eqns = [eqn]
    g = ingest_jaxpr(jaxpr)
    assert g.name == "jaxpr_ingested"
    assert len(g.nodes) == 3

    # Test jaxpr with no eqns attribute and no constants
    jaxpr2 = MagicMock()
    del jaxpr2.eqns
    del jaxpr2.consts
    g2 = ingest_jaxpr(jaxpr2)
    assert len(g2.nodes) == 0
