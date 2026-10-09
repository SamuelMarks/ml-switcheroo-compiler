"""Test JAXPR Ingest."""

import pytest

from ml_switcheroo_compiler.transforms.foreign import ingest_jaxpr


class MockInvar:
    """Mock Invar."""

    def __init__(self, id_val):
        """Init."""
        self.id_val = id_val
        self.aval = type("MockAval", (), {"shape": (2, 3)})()


class MockOutvar:
    """Mock Outvar."""

    def __init__(self, id_val):
        """Init."""
        self.id_val = id_val
        self.aval = type("MockAval", (), {"shape": (2, 3)})()


class MockEqn:
    """Mock Eqn."""

    def __init__(self, prim_name, invars, outvars, params=None):
        """Init."""
        self.primitive = type("MockPrim", (), {"name": prim_name})()
        self.invars = invars
        self.outvars = outvars
        self.params = params or {}


class MockJaxpr:
    """Mock Jaxpr."""

    def __init__(self, eqns, consts=None, constvars=None):
        """Init."""
        self.eqns = eqns
        self.consts = consts or []
        self.constvars = constvars or []


def test_ingest_jaxpr_empty():
    """Test empty."""
    with pytest.raises(ValueError):
        ingest_jaxpr(None)

    g = ingest_jaxpr(MockJaxpr([]))
    assert len(g.nodes) == 0


def test_extract_jaxpr_constants():
    """Test constants."""
    consts = [10, 20]
    constvars = [MockInvar(1), MockInvar(2)]
    jaxpr = MockJaxpr([], consts=consts, constvars=constvars)
    g = ingest_jaxpr(jaxpr)

    assert len(g.nodes) == 2
    for node in g.nodes.values():
        assert node.op_type == "Constant"


def test_translate_jax_equation():
    """Test translate eqn."""
    invars = [MockInvar(1), MockInvar(2)]
    outvars = [MockOutvar(3)]

    eqns = [
        MockEqn("add", invars, outvars, {"dim": 1, "tup": (1, 2)}),
        MockEqn("unknown_prim", invars, [MockOutvar(4)]),
    ]
    jaxpr = MockJaxpr(eqns)
    g = ingest_jaxpr(jaxpr)

    out_id = str(id(outvars[0]))
    assert g.nodes[out_id].op_type == "Add"
    assert g.nodes[out_id].attributes["dim"] == 1
    assert g.nodes[out_id].attributes["tup"] == [1, 2]
    assert g.nodes[out_id].shape_metadata == (2, 3)

    out_id_2 = str(id(eqns[1].outvars[0]))
    assert g.nodes[out_id_2].op_type == "Unknown"


def test_translate_jax_equation_no_outvars():
    """Test translate eqn no outvars."""
    invars = [MockInvar(1)]
    eqns = [MockEqn("add", invars, [])]
    jaxpr = MockJaxpr(eqns)
    g = ingest_jaxpr(jaxpr)

    assert "out" in g.nodes
    assert g.nodes["out"].op_type == "Add"


def test_translate_jax_equation_no_aval():
    """Test translate eqn no aval."""
    invars = [MockInvar(1)]
    outvars = [type("MockOutvar", (), {"id_val": 3})()]
    eqns = [MockEqn("add", invars, outvars)]
    jaxpr = MockJaxpr(eqns)
    g = ingest_jaxpr(jaxpr)

    out_id = str(id(outvars[0]))
    assert g.nodes[out_id].shape_metadata == ()


def test_translate_jax_equation_attr_branches():
    """Test attributes branches."""
    invars = [MockInvar(1)]
    outvars = [MockOutvar(2)]

    # Passing an object that isn't int/float/str/bool or tuple/list to hit the implicit else
    class CustomObj:
        """Mock custom object."""

        pass

    eqns = [MockEqn("add", invars, outvars, {"tup": (1, 2), "obj": CustomObj()})]
    jaxpr = MockJaxpr(eqns)
    g = ingest_jaxpr(jaxpr)

    out_id = str(id(outvars[0]))
    assert "tup" in g.nodes[out_id].attributes
    assert "obj" not in g.nodes[out_id].attributes


def test_translate_jax_equation_attr_branches_no_params():
    """Test attributes branches no params."""
    invars = [MockInvar(1)]
    outvars = [MockOutvar(2)]

    # Passing an object without params
    eqn = MockEqn("add", invars, outvars)
    del eqn.params
    eqns = [eqn]
    jaxpr = MockJaxpr(eqns)
    g = ingest_jaxpr(jaxpr)

    out_id = str(id(outvars[0]))
    assert "tup" not in g.nodes[out_id].attributes
