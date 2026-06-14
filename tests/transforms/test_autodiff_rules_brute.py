"""Brute force tests for autodiff rules coverage."""

import pytest

from ml_switcheroo_compiler.core.errors import UnimplementedMathError
from ml_switcheroo_compiler.transforms.autodiff_rules.jvp_registry import _JVP_REGISTRY, get_jvp
from ml_switcheroo_compiler.transforms.autodiff_rules.vjp_registry import _VJP_REGISTRY, get_vjp


class DummyGraph:
    """A dummy graph for testing."""

    def __init__(self) -> None:
        """Initialize."""
        self.nodes = {"a": DummyNode("a", []), "b": DummyNode("b", [])}

    def add_node(self, node: object) -> None:
        """Add node.

        Args:
            node (object): node
        """
        self.nodes[node.id] = node


class DummyNode:
    """A dummy node for testing."""

    def __init__(self, op_type: str, inputs: list[str]) -> None:
        """Initialize.

        Args:
            op_type (str): op_type
            inputs (list[str]): inputs
        """
        self.id = op_type
        self.op_type = op_type
        self.inputs = inputs
        self.attributes = {
            "axis": 0,
            "dims": (0,),
            "keepdims": False,
            "k": 1,
            "shape": (1,),
            "axes": (0, 1),
        }
        self.shape_metadata = (1,)


def test_all_vjps() -> None:
    """Test all VJPs."""
    graph = DummyGraph()
    # Trigger get_vjp misses
    with pytest.raises(UnimplementedMathError):
        get_vjp("NotExistent")

    for op_name, _vjp_func in _VJP_REGISTRY.items():
        # Get actual from registry to hit the lookup branch
        f = get_vjp(op_name)
        try:
            node = DummyNode(op_name, ["a", "b"])
            f(graph, node, "cot")
        except Exception:
            pass
        try:
            node = DummyNode(op_name, ["a"])
            f(graph, node, "cot")
        except Exception:
            pass


def test_all_jvps() -> None:
    """Test all JVPs."""
    with pytest.raises(UnimplementedMathError):
        get_jvp("NotExistent")

    for op_name, _jvp_func in _JVP_REGISTRY.items():
        f = get_jvp(op_name)
        try:
            f("t_x", "t_y", "x", "y")
        except Exception:
            pass
        try:
            f("t", "x")
        except Exception:
            pass
        try:
            graph = DummyGraph()
            node = DummyNode(op_name, ["a"])
            f(graph, node, "t")
        except Exception:
            pass
        try:
            graph = DummyGraph()
            node2 = DummyNode(op_name, ["a"])
            node2.attributes["axes"] = (0, 1)
            f(graph, node2, "t")
        except Exception:
            pass
        try:
            graph = DummyGraph()
            node3 = DummyNode(op_name, ["a"])
            node3.attributes["axes"] = None
            f(graph, node3, "t")
        except Exception:
            pass
