"""Tests for foreign architecture bridging."""

import pytest

from ml_switcheroo_compiler.transforms.foreign import ingest_jaxpr, ingest_torch_fx


class MockFXNode:
    def __init__(self, op, target, name, args):
        self.op = op
        self.target = target
        self.name = name
        self.args = args


class MockFXGraph:
    def __init__(self):
        # x = placeholder
        # y = add(x, x)
        # return y
        node_x = MockFXNode("placeholder", None, "x", ())

        def add():
            pass

        add.__name__ = "add"

        node_add = MockFXNode("call_function", add, "add_1", (node_x, node_x))
        node_out = MockFXNode("output", None, "output", (node_add,))

        self.nodes = [node_x, node_add, node_out]


class MockFXModule:
    def __init__(self):
        self.graph = MockFXGraph()


class MockJaxPrimitive:
    def __init__(self, name):
        self.name = name


class MockJaxEqn:
    def __init__(self):
        self.primitive = MockJaxPrimitive("add")
        self.invars = ["in1", "in2"]
        self.outvars = ["out1"]


class MockJaxpr:
    def __init__(self):
        self.eqns = [MockJaxEqn()]


def test_foreign_ingestion_torch():
    with pytest.raises(ValueError):
        ingest_torch_fx(None)

    module = MockFXModule()
    graph = ingest_torch_fx(module)
    assert graph.name == "torch_fx_ingested"
    assert "add_1" in graph.nodes
    assert graph.nodes["add_1"].op_type == "Add"
    assert graph.outputs == ["add_1"]


def test_foreign_ingestion_jax():
    with pytest.raises(ValueError):
        ingest_jaxpr(None)

    jaxpr = MockJaxpr()
    graph = ingest_jaxpr(jaxpr)
    assert graph.name == "jaxpr_ingested"
    assert len(graph.nodes) == 1
    # Check that there is an add node
    node = list(graph.nodes.values())[0]
    assert node.op_type == "Add"


def test_foreign_ingestion_torch_more():
    module = MockFXModule()

    def mul():
        pass

    mul.__name__ = "mul"

    node_x = MockFXNode("placeholder", None, "x", ())
    node_mul = MockFXNode("call_function", mul, "mul_1", (node_x, "constant"))
    node_out = MockFXNode("output", None, "output", ((node_mul,),))

    module.graph.nodes = [node_x, node_mul, node_out]

    graph = ingest_torch_fx(module)
    assert "mul_1" in graph.nodes
    assert graph.nodes["mul_1"].op_type == "Mul"
    assert "constant" in graph.nodes["mul_1"].inputs
    assert graph.outputs == ["mul_1"]


def test_foreign_ingestion_jax_more():
    jaxpr = MockJaxpr()
    jaxpr.eqns[0].primitive.name = "mul"
    graph = ingest_jaxpr(jaxpr)
    node = list(graph.nodes.values())[0]
    assert node.op_type == "Mul"
