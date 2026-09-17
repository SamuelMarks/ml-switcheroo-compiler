# ruff: noqa: E501
from ml_switcheroo_compiler.ir.core import IRGraph, IRNode
from ml_switcheroo_compiler.transforms.passes.lift_state import _get_nodes, _lift_block_ir, _lift_node, flatten_state_dict, lift_state_pass, unflatten_state_dict


def test_lift_state_branches() -> None:

    class Block:
        nodes = {"n1": IRNode(id="n1", op_type="Add", inputs=[], attributes={"some_attr": "no_nodes"})}

    _lift_block_ir(Block())


"Test Lift State Pass."


def test_flatten_unflatten_state_dict() -> None:
    nested = {"a": 1, "b": {"c": 2, "d": {"e": 3}}}
    flat = flatten_state_dict(nested)
    assert flat == {"a": 1, "b.c": 2, "b.d.e": 3}
    assert unflatten_state_dict(flat) == nested


def test_get_nodes() -> None:

    class DummyBlock:
        nodes = {"n1": IRNode(id="n1", op_type="Input", inputs=[])}

    assert list(_get_nodes(DummyBlock()))[0].id == "n1"

    class DummyBlock2:
        nodes = [IRNode(id="n2", op_type="Input", inputs=[])]

    assert list(_get_nodes(DummyBlock2()))[0].id == "n2"

    class DummyBlock3:
        pass

    assert list(_get_nodes(DummyBlock3())) == []


def test_lift_node() -> None:

    class DummyBlock:
        outputs: list[str] = []

    block = DummyBlock()
    node1 = IRNode(id="n1", op_type="ReadVariable", inputs=[])
    assert _lift_node(node1, block) is True
    assert node1.op_type == "Input"
    node2 = IRNode(id="n2", op_type="AssignVariable", inputs=["in1"])
    assert _lift_node(node2, block) is True
    assert node2.op_type == "Output"
    assert node2.inputs == ["in1"]
    assert "n2" in block.outputs
    node3 = IRNode(id="n3", op_type="Assign", inputs=["var", "val"])
    assert _lift_node(node3, block) is True
    assert node3.op_type == "Output"
    assert node3.inputs == ["val"]
    assert "n3" in block.outputs
    node4 = IRNode(id="n4", op_type="Assign", inputs=["var2", "val2"])
    block.outputs = ["n4"]
    assert _lift_node(node4, block) is True
    assert node4.op_type == "Output"
    assert block.outputs == ["n4"]

    class BlockNoOutputs:
        pass

    node5 = IRNode(id="n5", op_type="Assign", inputs=["var3", "val3"])
    assert _lift_node(node5, BlockNoOutputs()) is True
    assert node5.op_type == "Output"
    node6 = IRNode(id="n6", op_type="Add", inputs=[])
    assert _lift_node(node6, block) is False


def test_lift_block_ir() -> None:

    class SubBlock:
        nodes = {"n1": IRNode(id="n1", op_type="ReadVariable", inputs=[])}

    node1 = IRNode(id="n2", op_type="Cond", inputs=[], attributes={"body": SubBlock()})

    class Block:
        nodes = {"n2": node1}

    assert _lift_block_ir(Block()) is True
    assert SubBlock.nodes["n1"].op_type == "Input"


def test_lift_state_pass() -> None:
    graph = IRGraph(name="test", nodes={"n1": IRNode(id="n1", op_type="ReadVariable", inputs=[])}, outputs=[])
    assert lift_state_pass(graph) is True
    assert graph.nodes["n1"].op_type == "Input"


def test_lift_module_state_and_functionalize() -> None:
    """Test lift_module_state and lift_state on stateful modules."""
    import numpy as np

    from ml_switcheroo_compiler.transforms.passes.lift_state import lift_module_state, lift_state

    # 1. Custom module with named_parameters
    class MockNamedParamModule:
        def __init__(self) -> None:
            self.w = np.array([[1.0, 2.0]], dtype=np.float32)
            self.b = np.array([0.5], dtype=np.float32)

        def named_parameters(self):
            return [("w", self.w), ("b", self.b)]

        def __call__(self, x: np.ndarray) -> np.ndarray:
            return x @ self.w.T + self.b

    mod = MockNamedParamModule()
    params, g = lift_module_state(mod)
    assert "w" in params
    assert "b" in params
    assert "param_w" in g.nodes
    assert g.nodes["param_w"].attributes["is_state"] is True

    # Test functional execution via lift_state
    params_lifted, pure_fn = lift_state(mod)
    x = np.array([[2.0, 3.0]], dtype=np.float32)
    out, updated_params = pure_fn(params_lifted, x)
    expected = x @ mod.w.T + mod.b
    assert np.allclose(out, expected)

    # 2. Module with state_dict
    class MockStateDictModule:
        def __init__(self) -> None:
            self.weight = np.ones((2, 2), dtype=np.float32)

        def state_dict(self):
            return {"weight": self.weight}

        def forward(self, x: np.ndarray) -> np.ndarray:
            return x @ self.weight

    mod2 = MockStateDictModule()
    params2, g2 = lift_module_state(mod2)
    assert "weight" in params2
    _, pure_fn2 = lift_state(mod2)
    out2, _ = pure_fn2(params2, x)
    assert out2 is not None

    # 3. Simple module relying on __dict__ with private and callable members
    class SimpleStateful:
        def __init__(self) -> None:
            self.param = 42
            self._private = 100

        def helper(self) -> None:
            pass

    mod3 = SimpleStateful()
    params3, g3 = lift_module_state(mod3)
    assert "param" in params3
    assert "_private" not in params3
    assert "helper" not in params3

    # Pass an extra param that module does not have yet to exercise line 197->199
    _, pure_fn3 = lift_state(mod3)
    out3, updated3 = pure_fn3({"param": 99, "new_param": 123})
    assert out3 is None
    assert updated3["param"] == 99

    # 4. Object without __dict__, named_parameters, or state_dict (e.g. using __slots__)
    class SlotsModule:
        __slots__ = ()

    params4, g4 = lift_module_state(SlotsModule())
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
