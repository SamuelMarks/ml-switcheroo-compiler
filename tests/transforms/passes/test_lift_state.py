"""Tests for test_lift_state."""

from __future__ import annotations

import typing

# ruff: noqa: E501
from ml_switcheroo_compiler.ir.core import IRGraph, IRNode
from ml_switcheroo_compiler.transforms.passes.lift_state import _get_nodes, _lift_block_ir, _lift_node, flatten_state_dict, lift_state_pass, unflatten_state_dict
from tests.testing_utils import as_attrs, as_callable, as_ndarray, as_tensor

if typing.TYPE_CHECKING:
    pass


def test_lift_state_branches() -> None:
    """Test function test_lift_state_branches."""

    class Block:
        """Test class Block."""

        nodes = {"n1": IRNode(id="n1", op_type="Add", inputs=[], attributes={"some_attr": "no_nodes"})}

    as_callable(_lift_block_ir)(as_callable(Block)())
    assert callable(_lift_block_ir)


def test_flatten_unflatten_state_dict() -> None:
    """Test function test_flatten_unflatten_state_dict."""
    nested = {"a": 1, "b": {"c": 2, "d": {"e": 3}}}
    flat = as_callable(flatten_state_dict)(nested)
    assert flat == {"a": 1, "b.c": 2, "b.d.e": 3}
    assert as_callable(unflatten_state_dict)(flat) == nested


def test_get_nodes() -> None:
    """Test function test_get_nodes."""

    class DummyBlock:
        """Test class DummyBlock."""

        nodes = {"n1": IRNode(id="n1", op_type="Input", inputs=[])}

    assert as_callable(list)(as_callable(_get_nodes)(as_callable(DummyBlock)()))[0].id == "n1"

    class DummyBlock2:
        """Test class DummyBlock2."""

        nodes = [IRNode(id="n2", op_type="Input", inputs=[])]

    assert as_callable(list)(as_callable(_get_nodes)(as_callable(DummyBlock2)()))[0].id == "n2"

    class DummyBlock3:
        """Test class DummyBlock3."""

        pass

    assert as_callable(list)(as_callable(_get_nodes)(as_callable(DummyBlock3)())) == []


def test_lift_node() -> None:
    """Test function test_lift_node."""

    class DummyBlock:
        """Test class DummyBlock."""

        outputs: list[str] = []

    block = DummyBlock()
    node1 = IRNode(id="n1", op_type="ReadVariable", inputs=[])
    assert bool(as_callable(_lift_node)(node1, block)) is True
    assert node1.op_type == "Input"
    node2 = IRNode(id="n2", op_type="AssignVariable", inputs=["in1"])
    assert bool(as_callable(_lift_node)(node2, block)) is True
    assert node2.op_type == "Output"
    assert node2.inputs == ["in1"]
    assert "n2" in block.outputs
    node3 = IRNode(id="n3", op_type="Assign", inputs=["var", "val"])
    assert bool(as_callable(_lift_node)(node3, block)) is True
    assert node3.op_type == "Output"
    assert node3.inputs == ["val"]
    assert "n3" in block.outputs
    node4 = IRNode(id="n4", op_type="Assign", inputs=["var2", "val2"])
    block.outputs = ["n4"]
    assert bool(as_callable(_lift_node)(node4, block)) is True
    assert node4.op_type == "Output"
    assert block.outputs == ["n4"]

    class BlockNoOutputs:
        """Test class BlockNoOutputs."""

        pass

    node5 = IRNode(id="n5", op_type="Assign", inputs=["var3", "val3"])
    assert bool(as_callable(_lift_node)(node5, as_callable(BlockNoOutputs)())) is True
    assert node5.op_type == "Output"
    node6 = IRNode(id="n6", op_type="Add", inputs=[])
    assert bool(as_callable(_lift_node)(node6, block)) is False


def test_lift_block_ir() -> None:
    """Test function test_lift_block_ir."""

    class SubBlock:
        """Test class SubBlock."""

        nodes = {"n1": IRNode(id="n1", op_type="ReadVariable", inputs=[])}

    node1 = IRNode(id="n2", op_type="Cond", inputs=[], attributes=as_attrs({"body": SubBlock()}))

    class Block:
        """Test class Block."""

        nodes = {"n2": node1}

    assert bool(as_callable(_lift_block_ir)(as_callable(Block)())) is True
    assert SubBlock.nodes["n1"].op_type == "Input"


def test_lift_state_pass() -> None:
    """Test function test_lift_state_pass."""
    graph = IRGraph(name="test", nodes={"n1": IRNode(id="n1", op_type="ReadVariable", inputs=[])}, outputs=[])
    assert lift_state_pass(graph) is True
    assert graph.nodes["n1"].op_type == "Input"


def test_lift_module_state_and_functionalize() -> None:
    """Test lift_module_state and lift_state on stateful variable containers."""
    import numpy as np
    import pytest

    from ml_switcheroo_compiler.transforms.passes.lift_state import lift_module_state, lift_state

    class MockVariableContainer:
        """Generic stateful variable container."""

        def __init__(self) -> None:
            """Initialize MockVariableContainer."""
            self.w = np.array([[1.0, 2.0]], dtype=np.float32)
            self.b = np.array([0.5], dtype=np.float32)

        def __call__(self, x: np.ndarray[tuple[int, ...], np.dtype[np.generic]]) -> np.ndarray[tuple[int, ...], np.dtype[np.generic]]:
            """Evaluate container on x.

            Args:
                x: Test parameter `x`.
            """
            return as_ndarray((as_tensor(x) @ as_tensor(self.w.T)) + as_tensor(self.b))

    mod = MockVariableContainer()
    params, g = lift_module_state(as_tensor(mod))
    assert "w" in params
    assert "b" in params
    assert "param_w" in g.nodes
    assert g.nodes["param_w"].attributes["is_state"] is True

    params_lifted, pure_fn = lift_state(as_callable(mod))
    x = np.array([[2.0, 3.0]], dtype=np.float32)
    out, updated_params = pure_fn(params_lifted, x)
    expected = x @ mod.w.T + mod.b
    assert as_callable(np.allclose)(out, expected)
    assert "w" in updated_params

    class NonCallableContainer:
        """Container that is not callable."""

        def __init__(self) -> None:
            """Initialize NonCallableContainer."""
            self.weight = np.ones((2, 2), dtype=np.float32)

    mod2 = NonCallableContainer()
    params2, _ = as_callable(lift_module_state)(mod2)
    assert "weight" in params2
    _, pure_fn2 = as_callable(lift_state)(mod2)
    with pytest.raises(TypeError, match="is not callable"):
        pure_fn2(params2, x)

    class SlotsModule:
        """Container using __slots__."""

        __slots__ = ()

    params4, _ = as_callable(lift_module_state)(as_callable(SlotsModule)())
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
