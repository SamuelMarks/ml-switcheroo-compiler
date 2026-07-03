"""Module docstring."""

from unittest.mock import MagicMock, patch

from ml_switcheroo_compiler.transforms.autodiff_rules.binary_rules import (
    atan2_jvp,
    betainc_vjp,
    polygamma_vjp,
    random_gamma_vjp,
    zeta_vjp,
)


def test_binary_rules_extra() -> object:
    """Function docstring."""

    class DummyNode:
        """Class docstring."""

        def __init__(self, inputs: object) -> object:
            """Function docstring."""
            self.id = "id"
            self.inputs = inputs

    class DummyGraph:
        """Class docstring."""

        def __init__(self) -> object:
            """Function docstring."""
            self.nodes = {}

    graph = DummyGraph()
    graph.nodes["x"] = type("N", (), {"shape_metadata": None})
    graph.nodes["q"] = type("N", (), {"shape_metadata": None})
    graph.nodes["n"] = type("N", (), {"shape_metadata": None})
    graph.nodes["a"] = type("N", (), {"shape_metadata": None})
    graph.nodes["b"] = type("N", (), {"shape_metadata": None})
    graph.nodes["alpha"] = type("N", (), {"shape_metadata": None})

    with patch(
        "ml_switcheroo_compiler.transforms.autodiff_rules.binary_rules.emit_ir_node",
        return_value="mock_out",
    ):
        # Zeta
        zeta_node = DummyNode(["x", "q"])
        dx, dq = zeta_vjp(graph, zeta_node, "cotangent")
        assert dq == "mock_out"

        # Polygamma
        poly_node = DummyNode(["n", "x"])
        dn, dx = polygamma_vjp(graph, poly_node, "cotangent")
        assert dx == "mock_out"

        # Betainc
        betainc_node = DummyNode(["a", "b", "x"])
        da, db, dx = betainc_vjp(graph, betainc_node, "cotangent")
        assert dx == "mock_out"

        # RandomGamma
        random_gamma_node = DummyNode(["alpha", "key"])
        da, dk = random_gamma_vjp(graph, random_gamma_node, "cotangent")
        assert da == "mock_out"


def test_atan2_jvp() -> object:
    """Function docstring."""
    graph = MagicMock()
    y_mock = MagicMock()
    y_mock.shape_metadata = {}
    x_mock = MagicMock()
    x_mock.shape_metadata = {}
    graph.nodes = {"y": y_mock, "x": x_mock}

    res = atan2_jvp(tangent_y="ty", tangent_x="tx", y="y", x="x", graph=graph)
    assert res is not None
