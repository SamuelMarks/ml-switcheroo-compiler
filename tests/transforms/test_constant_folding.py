"""Unit tests for the constant folding optimization pass on logical graphs."""

from __future__ import annotations

import typing
from unittest import mock
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from ml_switcheroo_ir import LogicalGraph, LogicalNode, NodeDict

from ml_switcheroo_compiler.ir.core import IRGraph, IRNode
from ml_switcheroo_compiler.transforms.passes.constant_folding import (
    _evaluate_constant_node,
    constant_folding_pass,
)
from tests.testing_utils import as_attrs, as_callable

if typing.TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import EagerValue


def test_constant_folding_coverage_brute() -> None:
    """Test constant folding pass across non-constant inputs, constant inputs, and unknown ops."""
    g = IRGraph()
    n1 = IRNode(id="n1", op_type="Input", inputs=[], attributes={}, shape_metadata=None)
    n2 = IRNode(id="n2", op_type="Add", inputs=["n1"], attributes={}, shape_metadata=None)
    g.nodes = NodeDict({"n1": n1, "n2": n2})
    assert constant_folding_pass(g) is False
    assert g.nodes["n2"].op_type == "Add"

    g3 = IRGraph()
    n4 = IRNode(id="n4", op_type="Constant", inputs=[], attributes={"value": [1, 2]}, shape_metadata=None)
    n5 = IRNode(id="n5", op_type="Constant", inputs=[], attributes={"value": [3, 4]}, shape_metadata=None)
    n6 = IRNode(id="n6", op_type="Add", inputs=["n4", "n5"], attributes={}, shape_metadata=None)
    g3.nodes = NodeDict({"n4": n4, "n5": n5, "n6": n6})
    assert constant_folding_pass(g3) is True
    assert g3.nodes["n6"].op_type == "Constant"

    g4 = IRGraph()
    n7 = IRNode(id="n7", op_type="Constant", inputs=[], attributes={"value": 1}, shape_metadata=None)
    n8 = IRNode(id="n8", op_type="UnknownOpThatRaisesException", inputs=["n7"], attributes={}, shape_metadata=None)
    g4.nodes = NodeDict({"n7": n7, "n8": n8})
    assert constant_folding_pass(g4) is False
    assert g4.nodes["n8"].op_type == "UnknownOpThatRaisesException"


def test_constant_folding_numel_branch(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test the constant folding numel branch behavior.

    Args:
        monkeypatch (pytest.MonkeyPatch): Pytest monkeypatch fixture.
    """
    graph = IRGraph()
    n0 = IRNode(id="n0", op_type="Constant", inputs=[], attributes={"value": 1})
    n1 = IRNode(id="n1", op_type="Add", inputs=["n0"], attributes={})
    graph.nodes["n0"] = n0
    graph.nodes["n1"] = n1

    class MockVal:
        """Mock value implementing numel()."""

        def numel(self) -> int:
            """Return element count.

            Returns:
                int: Element count.
            """
            return 1

    def mock_eval(g: EagerValue, inputs: dict[str, EagerValue], backend: EagerValue = None) -> dict[str, EagerValue]:
        """Return mock evaluation output.

        Args:
            g (EagerValue): Graph argument.
            inputs (dict[str, object]): Inputs mapping.
            backend (EagerValue): Backend argument.

        Returns:
            dict[str, object]: Evaluated outputs dictionary.
        """
        _ = (g, inputs, backend)
        return {"n1": as_callable(MockVal)()}

    monkeypatch.setattr("ml_switcheroo_compiler.transforms.passes.constant_folding.evaluate_graph", mock_eval)
    mock_backend = MagicMock()
    mock_backend.item.return_value = 42
    monkeypatch.setattr("ml_switcheroo_compiler.transforms.passes.constant_folding.get_active_backend", lambda: mock_backend)
    assert constant_folding_pass(graph) is True
    assert graph.nodes["n1"].op_type == "Constant"
    assert graph.nodes["n1"].attributes["value"] == 42


def test_constant_folding() -> None:
    """Verify that the constant folding pass folds an Add operation on two Constant nodes."""
    g = LogicalGraph(outputs=["n1"])
    g.nodes["c1"] = LogicalNode(id="c1", op_type="Constant", attributes=as_attrs({"value": np.array([2.0])}))
    g.nodes["c2"] = LogicalNode(id="c2", op_type="Constant", attributes=as_attrs({"value": np.array([3.0])}))
    g.nodes["n1"] = LogicalNode(id="n1", op_type="Add", inputs=["c1", "c2"])
    assert constant_folding_pass(g) is True
    assert g.nodes["n1"].op_type == "Constant"
    val = g.nodes["n1"].attributes["value"]
    np.testing.assert_allclose(typing.cast("list[float]", val), np.array([5.0]))


def test_constant_folding_unsupported_op() -> None:
    """Verify that constant folding ignores unsupported operations."""
    g = LogicalGraph(outputs=["n1"])
    g.nodes["c1"] = LogicalNode(id="c1", op_type="Constant", attributes=as_attrs({"value": np.array([2.0])}))
    g.nodes["n1"] = LogicalNode(id="n1", op_type="UnknownOp", inputs=["c1"])
    assert constant_folding_pass(g) is False
    assert g.nodes["n1"].op_type == "UnknownOp"


def test_constant_folding_scalar_unwrap() -> None:
    """Verify that constant folding unwraps 1-element numpy arrays to Python scalars."""
    g = LogicalGraph(outputs=["n1"])
    g.nodes["c1"] = LogicalNode(id="c1", op_type="Constant", attributes=as_attrs({"value": np.array([2])}))
    g.nodes["c2"] = LogicalNode(id="c2", op_type="Constant", attributes=as_attrs({"value": np.array([3])}))
    g.nodes["n1"] = LogicalNode(id="n1", op_type="Add", inputs=["c1", "c2"])
    assert constant_folding_pass(g) is True
    assert g.nodes["n1"].op_type == "Constant"
    assert not isinstance(g.nodes["n1"].attributes["value"], np.ndarray)
    assert g.nodes["n1"].attributes["value"] == 5


def test_constant_folding_exception(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify that unhandled exceptions in evaluate_graph propagate out of constant_folding_pass.

    Args:
        monkeypatch (pytest.MonkeyPatch): Pytest monkeypatch fixture.
    """
    graph = IRGraph()
    n0 = IRNode(id="n0", op_type="Constant", inputs=[], attributes={"value": 1})
    n1 = IRNode(id="n1", op_type="Add", inputs=["n0"], attributes={})
    graph.nodes["n0"] = n0
    graph.nodes["n1"] = n1

    def mock_eval(*args: EagerValue, **kwargs: EagerValue) -> None:
        """Test function mock_eval.

        Args:
            args: Test parameter `args`.
            kwargs: Test parameter `kwargs`.
        """
        _ = (args, kwargs)
        raise KeyError("Testing other exceptions")

    monkeypatch.setattr("ml_switcheroo_compiler.transforms.passes.constant_folding.evaluate_graph", mock_eval)
    with pytest.raises(KeyError):
        constant_folding_pass(graph)


def test_constant_folding_backend_item_branch() -> None:
    """Test eager node evaluation with and without backend item method."""

    class DummyBackend:
        """Dummy backend implementing item method."""

        @staticmethod
        def item(val: EagerValue) -> int:
            """Extract item.

            Args:
                val (EagerValue): Input value.

            Returns:
                int: Evaluated item.
            """
            _ = val
            return 42

    class ScalarNoItem:
        """Scalar-like object with size == 1 but no item method."""

        size = 1

    graph = LogicalGraph(name="g")
    c1 = LogicalNode(id="c1", op_type="Constant", attributes={"value": 42})
    graph.nodes["c1"] = c1
    node = LogicalNode(id="n1", op_type="Identity", inputs=["c1"], attributes={})
    graph.nodes["n1"] = node

    with mock.patch(
        "ml_switcheroo_compiler.transforms.passes.constant_folding.evaluate_graph",
        return_value={"n1": np.array([42])},
    ):
        val1 = _evaluate_constant_node(node, ["c1"], graph, backend=DummyBackend)
        assert val1 == 42

    with mock.patch(
        "ml_switcheroo_compiler.transforms.passes.constant_folding.evaluate_graph",
        return_value={"n1": ScalarNoItem()},
    ):
        val2 = as_callable(_evaluate_constant_node)(node, ["c1"], graph, backend=None)
        assert isinstance(val2, ScalarNoItem)

    with mock.patch(
        "ml_switcheroo_compiler.transforms.passes.constant_folding.evaluate_graph",
        return_value={"n1": np.array([42])},
    ):
        val3 = as_callable(_evaluate_constant_node)(node, ["c1"], graph, backend=None)
        assert val3 == 42


def test_constant_folding_backend_instance_item_branch() -> None:
    """Test constant folding pass with backend having item method on instance."""

    class MockBackendWithItem:
        """Mock backend with item method."""

        def item(self, val: EagerValue) -> float:
            """Return float item.

            Args:
                val (EagerValue): Input value.

            Returns:
                float: Unwrapped item.
            """
            _ = val
            return 42.0

    node = LogicalNode(id="c0", op_type="Add", inputs=[], attributes={}, shape_metadata=(1,))
    graph = IRGraph(name="cf_backend_item", nodes={"c0": node})
    with patch(
        "ml_switcheroo_compiler.transforms.passes.constant_folding.evaluate_graph",
        return_value={"c0": np.array([42.0])},
    ):
        val = as_callable(_evaluate_constant_node)(node, [], graph, as_callable(MockBackendWithItem)())
        assert val == 42.0


def test_constant_folding_exceptions_and_fallback() -> None:
    """Test constant folding pass with fallback backend, ValueError, and UnimplementedMathError."""

    class UnimplementedMathError(Exception):
        """Custom math exception."""

    g = LogicalGraph(name="g", outputs=["n_out"])
    c1 = LogicalNode(id="c1", op_type="Constant", attributes={"value": 1.0})
    n_val_err = LogicalNode(id="n_err", op_type="Add", inputs=["c1"])
    n_unimpl = LogicalNode(id="n_unimpl", op_type="Sub", inputs=["c1"])
    n_out = LogicalNode(id="n_out", op_type="Relu", inputs=["c1"])
    g.nodes = NodeDict({"c1": c1, "n_err": n_val_err, "n_unimpl": n_unimpl, "n_out": n_out})

    def mock_eval(node: LogicalNode, *args: EagerValue, **kwargs: EagerValue) -> np.ndarray[tuple[int, ...], np.dtype[np.generic]]:
        """Test function mock_eval.

        Args:
            node: Test parameter `node`.
            args: Test parameter `args`.
            kwargs: Test parameter `kwargs`.
        """
        _ = (args, kwargs)
        if node.id == "n_err":
            raise ValueError("Value error in folding")
        if node.id == "n_unimpl":
            raise UnimplementedMathError("Unimplemented math")
        return np.array([2.0])

    with mock.patch("ml_switcheroo_compiler.transforms.passes.constant_folding.get_active_backend", return_value=None):
        with mock.patch("ml_switcheroo_compiler.transforms.passes.constant_folding._evaluate_constant_node", side_effect=mock_eval):
            res = constant_folding_pass(g)
            assert res is True


def test_constant_folding_subgraphs() -> None:
    """Test constant folding pass recursively folds constants inside node.subgraphs."""
    subgraph = LogicalGraph(name="then_branch", outputs=["add_res"])
    c1 = LogicalNode(id="c1", op_type="Constant", attributes={"value": 2.0})
    c2 = LogicalNode(id="c2", op_type="Constant", attributes={"value": 3.0})
    add_node = LogicalNode(id="add_res", op_type="Add", inputs=["c1", "c2"])
    subgraph.nodes = NodeDict({"c1": c1, "c2": c2, "add_res": add_node})

    parent = LogicalGraph(name="parent", outputs=["if_node"])
    parent.nodes["if_node"] = LogicalNode(
        id="if_node",
        op_type="If",
        subgraphs={"then_branch": subgraph},
    )

    res = constant_folding_pass(parent)
    assert res is True
    assert subgraph.nodes["add_res"].op_type == "Constant"


def test_constant_folding_attributes_subgraph_and_exceptions() -> None:
    """Test constant folding pass traversing subgraphs in node.attributes and BackendRegistry fallback."""
    subgraph = LogicalGraph(name="body", outputs=["add_res"])
    c1 = LogicalNode(id="c1", op_type="Constant", attributes={"value": 10.0})
    c2 = LogicalNode(id="c2", op_type="Constant", attributes={"value": 20.0})
    add_node = LogicalNode(id="add_res", op_type="Add", inputs=["c1", "c2"])
    subgraph.nodes = NodeDict({"c1": c1, "c2": c2, "add_res": add_node})

    parent = LogicalGraph(name="parent", outputs=["while_node"])
    parent.nodes["while_node"] = LogicalNode(
        id="while_node",
        op_type="While",
        attributes=as_attrs({"body_graph": subgraph}),
    )

    res = constant_folding_pass(parent)
    assert res is True
    assert subgraph.nodes["add_res"].op_type == "Constant"

    g2 = LogicalGraph(name="g2", outputs=["c1"])
    g2.nodes = NodeDict({"c1": LogicalNode(id="c1", op_type="Constant", attributes={"value": 1.0})})
    with mock.patch("ml_switcheroo_compiler.transforms.passes.constant_folding.get_active_backend", return_value=None):
        with mock.patch("ml_switcheroo_compiler.transforms.passes.constant_folding.BackendRegistry.get", side_effect=Exception("No numpy")):
            assert constant_folding_pass(g2) is False

    subgraph_unmodified = LogicalGraph(name="unmod", outputs=["c1"])
    subgraph_unmodified.nodes = NodeDict({"c1": LogicalNode(id="c1", op_type="Constant", attributes={"value": 1.0})})
    g_unmod = LogicalGraph(name="g_unmod", outputs=["sub_node"])
    g_unmod.nodes = NodeDict(
        {
            "sub_node": LogicalNode(
                id="sub_node",
                op_type="CustomOp",
                attributes=as_attrs({"scalar_val": 42, "child_g": subgraph_unmodified}),
                subgraphs={"sub1": subgraph_unmodified, "non_graph": "not_a_graph"},
            )
        }
    )
    assert constant_folding_pass(g_unmod) is False

    g3 = LogicalGraph(name="g3", outputs=["err_node"])
    c3 = LogicalNode(id="c3", op_type="Constant", attributes={"value": 1.0})
    err_node = LogicalNode(id="err_node", op_type="Add", inputs=["c3"])
    g3.nodes = NodeDict({"c3": c3, "err_node": err_node})

    def raise_custom(*args: EagerValue, **kwargs: EagerValue) -> None:
        """Raise an unhandled KeyError to test exception propagation.

        Args:
            *args: Positional arguments.
            **kwargs: Keyword arguments.

        Raises:
            KeyError: Always raised.
        """
        _ = (args, kwargs)
        raise KeyError("Fatal unhandled error")

    with mock.patch("ml_switcheroo_compiler.transforms.passes.constant_folding._evaluate_constant_node", side_effect=raise_custom):
        with pytest.raises(KeyError, match="Fatal unhandled error"):
            constant_folding_pass(g3)
