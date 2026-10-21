"""Test JAXPR Ingest."""

from __future__ import annotations

import typing

import pytest

from ml_switcheroo_compiler.transforms.foreign import ingest_jaxpr
from tests.testing_utils import as_callable, as_tensor

if typing.TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import EagerValue


class MockInvar:
    """Mock Invar."""

    def __init__(self, id_val: EagerValue) -> None:
        """Init.

        Args:
            id_val: Test parameter `id_val`.
        """
        self.id_val = id_val
        self.aval = type("MockAval", (), {"shape": (2, 3)})()


class MockOutvar:
    """Mock Outvar."""

    def __init__(self, id_val: EagerValue) -> None:
        """Init.

        Args:
            id_val: Test parameter `id_val`.
        """
        self.id_val = id_val
        self.aval = type("MockAval", (), {"shape": (2, 3)})()


class MockEqn:
    """Mock Eqn."""

    def __init__(self, prim_name: EagerValue, invars: EagerValue, outvars: EagerValue, params: EagerValue = None) -> None:
        """Init.

        Args:
            prim_name: Test parameter `prim_name`.
            invars: Test parameter `invars`.
            outvars: Test parameter `outvars`.
            params: Test parameter `params`.
        """
        self.primitive = type("MockPrim", (), {"name": prim_name})()
        self.invars = invars
        self.outvars = outvars
        self.params = params or {}


class MockJaxpr:
    """Mock Jaxpr."""

    def __init__(self, eqns: EagerValue, consts: EagerValue = None, constvars: EagerValue = None) -> None:
        """Init.

        Args:
            eqns: Test parameter `eqns`.
            consts: Test parameter `consts`.
            constvars: Test parameter `constvars`.
        """
        self.eqns = eqns
        self.consts = consts or []
        self.constvars = constvars or []


def test_ingest_jaxpr_empty() -> None:
    """Test empty."""
    with pytest.raises(ValueError):
        ingest_jaxpr(None)

    g = as_callable(ingest_jaxpr)(as_callable(MockJaxpr)([]))
    assert len(g.nodes) == 0


def test_extract_jaxpr_constants() -> None:
    """Test constants."""
    consts = [10, 20]
    constvars = [MockInvar(1), MockInvar(2)]
    jaxpr = as_callable(MockJaxpr)([], consts=consts, constvars=constvars)
    g = as_callable(ingest_jaxpr)(jaxpr)

    assert len(g.nodes) == 2
    for node in g.nodes.values():
        assert node.op_type == "Constant"


def test_translate_jax_equation() -> None:
    """Test translate eqn."""
    invars = [MockInvar(1), MockInvar(2)]
    outvars = [MockOutvar(3)]

    eqns = [
        as_callable(MockEqn)("add", invars, outvars, {"dim": 1, "tup": (1, 2)}),
        as_callable(MockEqn)("unknown_prim", invars, [as_callable(MockOutvar)(4)]),
    ]
    jaxpr = as_callable(MockJaxpr)(eqns)
    g = as_callable(ingest_jaxpr)(jaxpr)

    out_id = str(id(outvars[0]))
    assert g.nodes[out_id].op_type == "Add"
    assert g.nodes[out_id].attributes["dim"] == 1
    assert g.nodes[out_id].attributes["tup"] == [1, 2]
    assert g.nodes[out_id].shape_metadata == (2, 3)

    out_id_2 = str(id(as_tensor(getattr(as_tensor(getattr(eqns, "__getitem__")(1)).outvars, "__getitem__")(0))))
    assert g.nodes[out_id_2].op_type == "Unknown"


def test_translate_jax_equation_no_outvars() -> None:
    """Test translate eqn no outvars."""
    invars = [MockInvar(1)]
    eqns = [as_callable(MockEqn)("add", invars, [])]
    jaxpr = as_callable(MockJaxpr)(eqns)
    g = as_callable(ingest_jaxpr)(jaxpr)

    assert "out" in g.nodes
    assert g.nodes["out"].op_type == "Add"


def test_translate_jax_equation_no_aval() -> None:
    """Test translate eqn no aval."""
    invars = [MockInvar(1)]
    outvars = [type("MockOutvar", (), {"id_val": 3})()]
    eqns = [as_callable(MockEqn)("add", invars, outvars)]
    jaxpr = as_callable(MockJaxpr)(eqns)
    g = as_callable(ingest_jaxpr)(jaxpr)

    out_id = str(id(outvars[0]))
    assert g.nodes[out_id].shape_metadata == ()


def test_translate_jax_equation_attr_branches() -> None:
    """Test attributes branches."""
    invars = [MockInvar(1)]
    outvars = [MockOutvar(2)]

    # Passing an object that isn't int/float/str/bool or tuple/list to hit the implicit else
    class CustomObj:
        """Mock custom object."""

        pass

    eqns = [as_callable(MockEqn)("add", invars, outvars, {"tup": (1, 2), "obj": as_callable(CustomObj)()})]
    jaxpr = as_callable(MockJaxpr)(eqns)
    g = as_callable(ingest_jaxpr)(jaxpr)

    out_id = str(id(outvars[0]))
    assert "tup" in g.nodes[out_id].attributes
    assert "obj" not in g.nodes[out_id].attributes


def test_translate_jax_equation_attr_branches_no_params() -> None:
    """Test attributes branches no params."""
    invars = [MockInvar(1)]
    outvars = [MockOutvar(2)]

    # Passing an object without params
    eqn = as_callable(MockEqn)("add", invars, outvars)
    del eqn.params
    eqns = [eqn]
    jaxpr = as_callable(MockJaxpr)(eqns)
    g = as_callable(ingest_jaxpr)(jaxpr)

    out_id = str(id(outvars[0]))
    assert "tup" not in g.nodes[out_id].attributes
