# ruff: noqa: E501
"""Unit tests for state lifting pass."""

import typing

import pytest

from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.ir.core import IRGraph, IRNode
from ml_switcheroo_compiler.transforms.passes.state_lifting import (
    StateLiftingPass,
    _get_node_items,
    _get_val,
    _lift_block,
    _lift_node_state,
    _navigate,
    _set_val,
    state_lifting_pass,
)


class DummyBlock:
    """Dummy block container with nodes and outputs."""

    def __init__(self) -> None:
        """Initialize DummyBlock."""
        self.nodes: dict[str, IRNode] = {}
        self.outputs: list[str] = []


class MockTensor:
    """Mock tensor object with shape, dtype, and requires_grad."""

    def __init__(self, shape: tuple[int, ...] = (2, 2), dtype: str = "float32", requires_grad: bool = True) -> None:
        """Initialize MockTensor.

        Args:
            shape (tuple[int, ...]): Tensor shape.
            dtype (str): Tensor data type.
            requires_grad (bool): Whether tensor requires gradient.
        """
        self.shape = shape
        self.dtype = dtype
        self.requires_grad = requires_grad


def test_state_lifting() -> None:
    """Test standard state lifting with ReadVariable and AssignVariable nodes."""
    graph = IRGraph()
    n1 = IRNode("n1", "ReadVariable", attributes={"variable_name": "v1"})
    n2 = IRNode("n2", "AssignVariable", attributes={"variable_name": "v2"})
    n3 = IRNode("n3", "Add", attributes={})
    graph.nodes = {"n1": n1, "n2": n2, "n3": n3}
    graph.outputs = []
    res = state_lifting_pass(graph)
    assert res is True
    assert n1.op_type == "Input"
    assert n2.op_type == "Output"
    b = DummyBlock()
    n4 = IRNode("n4", "ReadVariable", attributes={})
    b.nodes = {"n4": n4}
    n5 = IRNode("n5", "If", attributes={"body": b})
    graph.nodes["n5"] = n5
    res = state_lifting_pass(graph)
    assert res is True


def test_get_node_items() -> None:
    """Test _get_node_items with None, list, and dict node representations."""
    assert _get_node_items(None) == []

    class DummyBlockList:
        """Dummy block containing list of nodes."""

        def __init__(self) -> None:
            """Initialize DummyBlockList."""
            n = IRNode("n1", "Add", attributes={})
            self.nodes = [n]

    assert len(list(_get_node_items(DummyBlockList()))) == 1


def test_state_lifting_already_in_outputs() -> None:
    """Test state lifting when node id is already in outputs."""
    nodes = {"assign1": IRNode(id="assign1", op_type="AssignVariable", inputs=["some_input"], attributes={"variable_name": "my_var"})}
    graph = IRGraph(name="test", nodes=nodes, outputs=["assign1"])
    state_lifting_pass(graph)
    assert graph.nodes["assign1"].op_type == "Output"
    assert graph.outputs == ["assign1"]


def test_state_lifting_no_outputs() -> None:
    """Test state lifting when block has no outputs attribute."""

    class DummyBlockNoOutputs:
        """Dummy block lacking outputs attribute."""

        def __init__(self) -> None:
            """Initialize DummyBlockNoOutputs."""
            self.nodes = {"assign1": IRNode(id="assign1", op_type="AssignVariable", inputs=["some_input"], attributes={"variable_name": "my_var"})}

    block = DummyBlockNoOutputs()
    _lift_block(block)
    assert block.nodes["assign1"].op_type == "Output"


def test_get_and_set_val() -> None:
    """Test _get_val and _set_val with dictionaries and objects."""
    d = {"a": 10, "b": 20}
    assert _get_val(d, "a") == 10
    assert _get_val(d, "c") is None
    _set_val(d, "a", 99)
    assert d["a"] == 99

    class Obj:
        """Simple object container."""

        def __init__(self) -> None:
            """Initialize Obj."""
            self.x = "hello"

    obj = Obj()
    assert _get_val(obj, "x") == "hello"
    assert _get_val(obj, "y") is None
    _set_val(obj, "x", "world")
    assert obj.x == "world"


def test_navigate() -> None:
    """Test _navigate traversing dictionaries and objects."""
    d = {"layer1": {"layer2": 42}}
    assert _navigate(d, ["layer1", "layer2"]) == 42
    assert _navigate(d, []) == d

    class Child:
        """Child container with value."""

        def __init__(self) -> None:
            """Initialize Child."""
            self.val = 123

    class Parent:
        """Parent container with child."""

        def __init__(self) -> None:
            """Initialize Parent."""
            self.child = Child()

    parent = Parent()
    assert _navigate(parent, ["child", "val"]) == 123


def test_lift_node_state_defaults() -> None:
    """Test _lift_node_state with default naming fallbacks and unhandled ops."""
    block = DummyBlock()
    read_node = IRNode("read1", "ReadVariable", attributes={})
    assert _lift_node_state(read_node, "read1", block) is True
    assert read_node.op_type == "Input"
    assert read_node.attributes["name"] == "var_read1"

    assign_node = IRNode("assign1", "AssignVariable", attributes={})
    assert _lift_node_state(assign_node, "assign1", block) is True
    assert assign_node.op_type == "Output"
    assert assign_node.attributes["name"] == "var_assign1_out"

    other_node = IRNode("n3", "Add", attributes={})
    assert _lift_node_state(other_node, "n3", block) is False


def test_extract_state_named_parameters_and_buffers() -> None:
    """Test extract_state with named_parameters and named_buffers."""
    pass_obj = StateLiftingPass()

    class ModelWithNamed:
        """Model defining named_parameters and named_buffers."""

        def __init__(self) -> None:
            """Initialize ModelWithNamed."""
            self.w = MockTensor()
            self.b = MockTensor()

        def named_parameters(self) -> list[tuple[str, object]]:
            """Return named parameters."""
            return [("w", self.w)]

        def named_buffers(self) -> list[tuple[str, object]]:
            """Return named buffers."""
            return [("b", self.b)]

    model = ModelWithNamed()
    params, buffers = pass_obj.extract_state(model, prefix="sub.")
    assert "sub.w" in params
    assert "sub.b" in buffers
    assert pass_obj.identity_map[id(model.w)] == "sub.w"
    assert pass_obj.identity_map[id(model.b)] == "sub.b"

    # Test when only named_buffers is provided
    class ModelOnlyBuffers:
        """Model with only named_buffers."""

        def __init__(self) -> None:
            """Initialize ModelOnlyBuffers."""
            self.b = MockTensor()

        def named_buffers(self) -> list[tuple[str, object]]:
            """Return named buffers."""
            return [("buf", self.b)]

    params2, buffers2 = StateLiftingPass().extract_state(ModelOnlyBuffers())
    assert params2 == {}
    assert "buf" in buffers2


def test_extract_state_reflection() -> None:
    """Test extract_state via reflection over __dict__."""
    pass_obj = StateLiftingPass()

    class SubModule:
        """Submodule with tensor attributes."""

        def __init__(self) -> None:
            """Initialize SubModule."""
            self.sub_param = MockTensor(requires_grad=True)
            self.sub_buf = MockTensor(requires_grad=False)

    class RootModel:
        """Root model with various attributes."""

        def __init__(self) -> None:
            """Initialize RootModel."""
            self.weight = MockTensor(requires_grad=True)
            self.bias = MockTensor(requires_grad=False)
            cfg = TensorConfig(shape=(2, 2), dtype="float32", device="cpu", requires_grad=True)
            self.real_tensor = Tensor([1.0], cfg)
            self._private_val = MockTensor()
            self._buffers_explicit = MockTensor(requires_grad=False)
            self.dict_params = {
                "w1": MockTensor(requires_grad=True),
                "b1": MockTensor(requires_grad=False),
                "non_tensor": "string_value",
            }
            self.sub = SubModule()
            self.primitive_int = 42
            self.instance_fn = lambda: 123

        def forward(self) -> None:
            """Dummy forward method."""
            pass

    model = RootModel()
    params, buffers = pass_obj.extract_state(model)
    assert "weight" in params
    assert "bias" in buffers
    assert "real_tensor" in params
    assert "_private_val" not in params
    assert "_private_val" not in buffers
    assert "_buffers_explicit" in buffers
    assert "dict_params.w1" in params
    assert "dict_params.b1" in buffers
    assert "dict_params.non_tensor" not in params
    assert "sub.sub_param" in params
    assert "sub.sub_buf" in buffers


def test_extract_state_no_dict() -> None:
    """Test extract_state on object without __dict__."""

    class SlotsModel:
        """Model using __slots__."""

        __slots__ = ()

    params, buffers = StateLiftingPass().extract_state(SlotsModel())
    assert params == {}
    assert buffers == {}


def test_lift_model_to_graph() -> None:
    """Test lifting module state into functional IRGraph."""
    pass_obj = StateLiftingPass()

    class SimpleModel:
        """Simple model with parameter and buffer."""

        def __init__(self) -> None:
            """Initialize SimpleModel."""
            self.w = MockTensor(shape=(4, 4), dtype="float32", requires_grad=True)
            self.running_mean = MockTensor(shape=(4,), dtype="float32", requires_grad=False)

    model = SimpleModel()
    combined_state, graph = pass_obj.lift(model)
    assert "w" in combined_state
    assert "running_mean" in combined_state
    assert graph.name == "functional_SimpleModel"
    assert "param_w" in graph.nodes
    assert "param_running_mean" in graph.nodes

    w_node = graph.nodes["param_w"]
    assert w_node.attributes["is_state"] is True
    assert w_node.attributes["is_parameter"] is True
    assert w_node.attributes["is_buffer"] is False
    assert w_node.shape_metadata["shape"] == (4, 4)

    mean_node = graph.nodes["param_running_mean"]
    assert mean_node.attributes["is_parameter"] is False
    assert mean_node.attributes["is_buffer"] is True


def test_functionalize_callable() -> None:
    """Test functionalize with callable model and state mutation."""
    pass_obj = StateLiftingPass()

    class CallableModel:
        """Callable model with state mutation."""

        def __init__(self) -> None:
            """Initialize CallableModel."""
            self.weight = [1.0]
            self.counter = 0

        def __call__(self, x: float) -> float:
            """Forward call modifying state.

            Args:
                x (float): Input value.

            Returns:
                float: Computed value.
            """
            self.counter += 1
            return x * self.weight[0] + self.counter

    model = CallableModel()
    model.weight = MockTensor(shape=(1,), dtype="float32", requires_grad=True)
    model.counter = MockTensor(shape=(), dtype="int32", requires_grad=False)

    init_state, pure_fn = pass_obj.functionalize(model)
    assert "weight" in init_state
    assert "counter" in init_state

    # Execute pure_fn
    out, new_state = pure_fn({"weight": [2.0], "counter": 10}, 5.0)
    assert out == 5.0 * 2.0 + 11
    assert new_state["counter"] == 11
    # Verify original attributes were restored in finally block
    assert model.weight is init_state["weight"]
    assert model.counter is init_state["counter"]


def test_functionalize_forward_method() -> None:
    """Test functionalize with non-callable model having forward method."""
    pass_obj = StateLiftingPass()

    class ForwardModel:
        """Model with forward method."""

        def __init__(self) -> None:
            """Initialize ForwardModel."""
            self.val = MockTensor(shape=(1,))

        def forward(self, delta: float) -> float:
            """Forward implementation.

            Args:
                delta (float): Increment value.

            Returns:
                float: Result.
            """
            return typing.cast(float, self.val) + delta

    model = ForwardModel()
    init_state, pure_fn = pass_obj.functionalize(model)
    out, new_state = pure_fn({"val": 100.0}, 25.0)
    assert out == 125.0
    assert new_state["val"] == 100.0
    assert model.val is init_state["val"]


def test_functionalize_type_error_restore() -> None:
    """Test functionalize raises TypeError and restores state if model has no forward or __call__."""
    pass_obj = StateLiftingPass()

    class InertModel:
        """Model without callable or forward."""

        def __init__(self) -> None:
            """Initialize InertModel."""
            self.p = MockTensor()

    model = InertModel()
    init_state, pure_fn = pass_obj.functionalize(model)
    with pytest.raises(TypeError, match="is not callable and has no forward method"):
        pure_fn({"p": "temporary"})
    assert model.p is init_state["p"]


def test_functionalize_nested_state() -> None:
    """Test functionalize with nested module paths."""
    pass_obj = StateLiftingPass()

    class InnerSub:
        """Inner submodule."""

        def __init__(self) -> None:
            """Initialize InnerSub."""
            self.param = MockTensor(requires_grad=True)

    class OuterSub:
        """Outer submodule containing nested submodule."""

        def __init__(self) -> None:
            """Initialize OuterSub."""
            self.inner = InnerSub()

        def forward(self, factor: int) -> int:
            """Forward function.

            Args:
                factor (int): Multiplier factor.

            Returns:
                int: Multiplied parameter.
            """
            return typing.cast(int, self.inner.param) * factor

    outer = OuterSub()
    _, pure_fn = pass_obj.functionalize(outer)
    res, new_st = pure_fn({"inner.param": 7}, 3)
    assert res == 21
    assert new_st["inner.param"] == 7
