from ml_switcheroo_compiler.transforms.autodiff_rules.binary_rules import (
    zeta_vjp,
    polygamma_vjp,
    betainc_vjp,
    random_gamma_vjp,
)
from unittest.mock import patch


def test_binary_rules_extra():
    class DummyNode:
        def __init__(self, inputs):
            self.id = "id"
            self.inputs = inputs

    class DummyGraph:
        def __init__(self):
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
