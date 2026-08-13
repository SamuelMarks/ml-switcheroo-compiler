# ruff: noqa
from ml_switcheroo_compiler.ir.core import LogicalGraph
from ml_switcheroo_compiler.transforms.foreign import _extract_jaxpr_constants
from ml_switcheroo_compiler.foreign import ForeignCall

"Tests for foreign transform logic."


class MockConst:
    """Mock constant."""


class MockJaxpr:
    """Mock JAX program."""

    def __init__(self, consts: list = None) -> None:
        """Initialize.

        Args:
            consts: Optional constants list.
        """
        self.consts = consts or []
        self.eqns = []


def test_extract_jaxpr_constants() -> None:
    """Test jaxpr constant extraction."""
    graph = LogicalGraph()
    jaxpr_empty = MockJaxpr()
    _extract_jaxpr_constants(jaxpr_empty, graph)
    jaxpr_with = MockJaxpr([MockConst()])
    _extract_jaxpr_constants(jaxpr_with, graph)


def test_extract_jaxpr_constants_nodes_added() -> None:
    """Test that constants are correctly added to the graph."""
    graph = LogicalGraph()
    jaxpr_with = MockJaxpr([MockConst()])
    _extract_jaxpr_constants(jaxpr_with, graph)
    assert "const_0" in graph.nodes
    assert graph.nodes["const_0"].op_type == "Constant"


def test_generic_utils_stubs():
    """Test generic utils stubs."""
    from ml_switcheroo_compiler.utils.generic_utils import (
        CustomObjectScope,
        clear_session,
        deserialize_keras_object,
        disable_interactive_logging,
        enable_interactive_logging,
        get_custom_objects,
        get_registered_name,
        get_registered_object,
        is_interactive_logging_enabled,
        is_keras_tensor,
        register_keras_serializable,
        serialize_keras_object,
        standardize_dtype,
    )

    clear_session()
    with CustomObjectScope():
        pass
    assert deserialize_keras_object() is None
    disable_interactive_logging()
    enable_interactive_logging()
    assert get_custom_objects() == {}
    assert get_registered_name() == ""
    assert get_registered_object() is None
    assert is_interactive_logging_enabled() is False
    assert is_keras_tensor() is False

    @register_keras_serializable()
    class Dummy:
        pass

    assert serialize_keras_object() is None
    assert standardize_dtype() is None


def test_foreign_coverage():
    op = ForeignCall()
    assert op.infer_shape() == ()

    class DummyTensor:
        shape = (1, 2)

    assert op.infer_shape(DummyTensor()) == (1, 2)
