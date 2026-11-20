"""Tests for test_lift_state."""

from __future__ import annotations

import typing

import pytest

# ruff: noqa: E501
from ml_switcheroo_ir import IRGraph, IRNode

from ml_switcheroo_compiler.backends.numpy.eager import ensure_ndarray
from ml_switcheroo_compiler.ir.core import ensure_attrs
from ml_switcheroo_compiler.transforms.passes.state_lifting import (
    _get_nodes,
    _lift_block_ir,
    _lift_node,
    flatten_state_dict,
    lift_state_pass,
    unflatten_state_dict,
)
from tests.testing_utils import MockGraphStub, MockTestStub

if typing.TYPE_CHECKING:
    pass


def test_lift_state_branches() -> None:
    """Test function test_lift_state_branches."""
    from ml_switcheroo_ir import NodeDict

    class Block(MockGraphStub):
        """Test class Block."""

        def __init__(self) -> None:
            """Initialize stub attributes."""
            super().__init__()
            self.nodes = NodeDict(
                {"n1": IRNode(id="n1", op_type="Add", inputs=[], attributes={"some_attr": "no_nodes"})}
            )

    _lift_block_ir(Block())
    assert callable(_lift_block_ir)


def test_flatten_unflatten_state_dict() -> None:
    """Test function test_flatten_unflatten_state_dict."""
    from ml_switcheroo_compiler.core.shape import OpValue

    nested: dict[str, OpValue] = {"a": 1, "b": {"c": 2, "d": {"e": 3}}}
    flat = flatten_state_dict(nested)
    assert flat == {"a": 1, "b.c": 2, "b.d.e": 3}
    assert unflatten_state_dict(flat) == nested


def test_get_nodes() -> None:
    """Test function test_get_nodes."""
    from ml_switcheroo_ir import NodeDict

    class DummyBlock(MockGraphStub):
        """Test class DummyBlock."""

        def __init__(self) -> None:
            """Initialize stub attributes."""
            super().__init__()
            self.nodes = NodeDict({"n1": IRNode(id="n1", op_type="Input", inputs=[])})

    assert list(_get_nodes(DummyBlock()))[0].id == "n1"

    class DummyBlock2(MockGraphStub):
        """Test class DummyBlock2."""

        def __init__(self) -> None:
            """Initialize stub attributes."""
            super().__init__()
            pytest.MonkeyPatch().setattr(self, "nodes", [IRNode(id="n2", op_type="Input", inputs=[])], raising=False)

    assert list(_get_nodes(DummyBlock2()))[0].id == "n2"

    class DummyBlock3(MockGraphStub):
        """Test class DummyBlock3."""

        pass

    assert list(_get_nodes(DummyBlock3())) == []


def test_lift_node() -> None:
    """Test function test_lift_node."""

    class DummyBlock(MockGraphStub):
        """Test class DummyBlock."""

        outputs: list[str] = []

    block = DummyBlock()
    node1 = IRNode(id="n1", op_type="ReadVariable", inputs=[])
    assert bool(_lift_node(node1, block)) is True
    assert node1.op_type == "Input"
    node2 = IRNode(id="n2", op_type="AssignVariable", inputs=["in1"])
    assert bool(_lift_node(node2, block)) is True
    assert node2.op_type == "Output"
    assert node2.inputs == ["in1"]
    assert "n2" in block.outputs
    node3 = IRNode(id="n3", op_type="Assign", inputs=["var", "val"])
    assert bool(_lift_node(node3, block)) is True
    assert node3.op_type == "Output"
    assert node3.inputs == ["val"]
    assert "n3" in block.outputs
    node4 = IRNode(id="n4", op_type="Assign", inputs=["var2", "val2"])
    block.outputs = ["n4"]
    assert bool(_lift_node(node4, block)) is True
    assert node4.op_type == "Output"
    assert block.outputs == ["n4"]

    class BlockNoOutputs(MockGraphStub):
        """Test class BlockNoOutputs."""

        pass

    node5 = IRNode(id="n5", op_type="Assign", inputs=["var3", "val3"])
    assert bool(_lift_node(node5, BlockNoOutputs())) is True
    assert node5.op_type == "Output"
    node6 = IRNode(id="n6", op_type="Add", inputs=[])
    assert bool(_lift_node(node6, block)) is False


def test_lift_block_ir() -> None:
    """Test function test_lift_block_ir."""
    sub_block = MockGraphStub(nodes={"n1": IRNode(id="n1", op_type="ReadVariable", inputs=[])})
    node1 = IRNode(id="n2", op_type="Cond", inputs=[], attributes=ensure_attrs({"body": sub_block}))

    class Block(MockGraphStub):
        """Test class Block."""

        def __init__(self) -> None:
            """Initialize stub attributes."""
            from ml_switcheroo_ir import NodeDict

            super().__init__()
            self.nodes = NodeDict({"n2": node1})

    assert bool(_lift_block_ir(Block())) is True
    assert sub_block.nodes["n1"].op_type == "Input"


def test_lift_state_pass() -> None:
    """Test function test_lift_state_pass."""
    graph = IRGraph(name="test", nodes={"n1": IRNode(id="n1", op_type="ReadVariable", inputs=[])}, outputs=[])
    assert lift_state_pass(graph) is True
    assert graph.nodes["n1"].op_type == "Input"


def test_lift_module_state_and_functionalize() -> None:
    """Test lift_module_state and lift_state on stateful variable containers."""
    import numpy as np

    from ml_switcheroo_compiler.transforms.passes.state_lifting import lift_module_state, lift_state

    class MockVariableContainer:
        """Generic stateful variable container."""

        def __init__(self) -> None:
            """Initialize MockVariableContainer."""
            self.w = np.array([[1.0, 2.0]], dtype=np.float32)
            self.b = np.array([0.5], dtype=np.float32)

        def state_dict(self) -> dict[str, np.ndarray[tuple[int, ...], np.dtype[np.float32]]]:
            """Return state dict.

            Returns:
                dict[str, np.ndarray[tuple[int, ...], np.dtype[np.float32]]]: State map.
            """
            return {"w": self.w, "b": self.b}

        def load_state_dict(self, state: dict[str, np.ndarray[tuple[int, ...], np.dtype[np.float32]]]) -> None:
            """Load updated state dict.

            Args:
                state (dict[str, np.ndarray[tuple[int, ...], np.dtype[np.float32]]]): Updated state map.
            """
            if "w" in state:
                self.w = state["w"]
            if "b" in state:
                self.b = state["b"]

        def __call__(
            self,
            x: np.ndarray[
                tuple[int, ...],
                np.dtype[np.float64 | np.float32 | np.int64 | np.int32 | np.uint8 | np.bool_],
            ],
        ) -> np.ndarray[
            tuple[int, ...],
            np.dtype[np.float64 | np.float32 | np.int64 | np.int32 | np.uint8 | np.bool_],
        ]:
            """Evaluate container on x.

            Args:
                x: Test parameter `x`.

            Returns:
                np.ndarray[tuple[int, ...], np.dtype[np.float64 | np.float32 | np.int64 | np.int32 | np.uint8 | np.bool_]]: Computed output value from `__call__`.
            """
            return ensure_ndarray((x @ self.w.T) + self.b)

    mod = MockVariableContainer()
    params, g = lift_module_state(mod)
    assert "w" in params
    assert "b" in params
    assert "param_w" in g.nodes
    assert bool(g.nodes["param_w"].attributes["is_state"]) is True

    params_lifted, pure_fn = lift_state(mod)
    x = np.array([[2.0, 3.0]], dtype=np.float32)
    out, updated_params = pure_fn(params_lifted, x)
    expected = x @ mod.w.T + mod.b
    assert np.allclose(ensure_ndarray(out), expected)
    assert "w" in updated_params

    class NonCallableContainer:
        """Container that is not callable."""

        def __init__(self) -> None:
            """Initialize NonCallableContainer."""
            self.weight = np.ones((2, 2), dtype=np.float32)

        def state_dict(self) -> dict[str, np.ndarray[tuple[int, ...], np.dtype[np.float32]]]:
            """Return state dict.

            Returns:
                dict[str, np.ndarray[tuple[int, ...], np.dtype[np.float32]]]: State map.
            """
            return {"weight": self.weight}

    mod2 = NonCallableContainer()
    params2, _ = lift_module_state(mod2)
    assert "weight" in params2
    _, pure_fn2 = lift_state(mod2)
    with pytest.raises(TypeError, match="is not callable"):
        pure_fn2(params2, x)

    class SlotsModule(MockTestStub):
        """Container using __slots__."""

        __slots__ = ()

    params4, _ = lift_module_state(SlotsModule())
    assert params4 == {}


def test_lift_state_subgraphs() -> None:
    """Verify lifting state inside node.subgraphs."""
    from ml_switcheroo_ir import LogicalGraph, LogicalNode

    sub = LogicalGraph(name="sub")
    sub.nodes["r1"] = LogicalNode(id="r1", op_type="ReadVariable", inputs=[])
    sub.nodes["a1"] = LogicalNode(id="a1", op_type="AssignVariable", inputs=["v1", "v2"])

    parent = LogicalGraph(name="parent")
    parent.nodes["if1"] = LogicalNode(id="if1", op_type="If", subgraphs={"then_branch": sub})

    res = lift_state_pass(parent)
    assert res is True
    assert sub.nodes["r1"].op_type == "Input"
    assert sub.nodes["a1"].op_type == "Output"
    assert "a1" in sub.outputs
