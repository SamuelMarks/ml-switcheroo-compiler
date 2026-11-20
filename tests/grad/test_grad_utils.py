"""Behavioral unit tests for grad utils."""

from __future__ import annotations

import typing
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from ml_switcheroo_ir import LogicalGraph, LogicalNode, NodeDict
from pytest_mock import MockerFixture

from ml_switcheroo_compiler.backends.eager_registry import BackendArg
from ml_switcheroo_compiler.core.device import Device
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig, Variable
from ml_switcheroo_compiler.grad.options import GradOptions
from ml_switcheroo_compiler.grad.utils import (
    _compute_grad_and_value,
    _find_wrt_tensors,
    _get_inputs_dict,
    _to_original_type,
)
from ml_switcheroo_compiler.tracing.state import global_tracing_state
from ml_switcheroo_compiler.tracing.tracer import ProxyTensor
from tests.testing_utils import MockTestStub, extract_test_dtype

if typing.TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import BackendArg


def test_get_inputs_dict_missing_input(mocker: MockerFixture) -> None:
    """Verify the behavior and output invariants of test get inputs dict missing input.

    Args:
        mocker: Test parameter `mocker`.
    """

    class DummyGraph(MockTestStub):
        """Test class DummyGraph."""

        inputs = ["missing_inp"]
        nodes: dict[str, LogicalNode] = {}

    with pytest.raises(ValueError, match="Missing input value for node 'missing_inp'"):
        _get_inputs_dict(DummyGraph())


def test_to_original_type_dtypes(mocker: MockerFixture) -> None:
    """Verify the behavior and output invariants of test to original type dtypes.

    Args:
        mocker: Test parameter `mocker`.
    """
    import numpy as np

    from ml_switcheroo_compiler.core.tensor import DType, Tensor, TensorConfig

    orig = Tensor(None, TensorConfig((), DType.Float32, None))

    class DummyBackend:
        """Test class DummyBackend."""

        def asarray(self, val: BackendArg) -> BackendArg:
            """Test function asarray.

            Args:
                val: Test parameter `val`.

            Returns:
                BackendArg: Computed output value from `asarray`.
            """
            return val

    mocker.patch("ml_switcheroo_compiler.grad.utils.get_active_backend", return_value=DummyBackend())

    val_int = np.array(1, dtype=np.int32)
    res_int = _to_original_type(val_int, orig)
    assert extract_test_dtype(res_int) == DType.Int32

    val_bool = np.array(True, dtype=bool)
    res_bool = _to_original_type(val_bool, orig)
    assert extract_test_dtype(res_bool) == DType.Bool


def test_find_wrt_tensors(mocker: MockerFixture) -> None:
    """Verify the behavior and output invariants of test find wrt tensors.

    Args:
        mocker: Test parameter `mocker`.
    """
    from ml_switcheroo_compiler.core.tensor import DType, Tensor, TensorConfig
    from ml_switcheroo_compiler.grad.utils import _find_wrt_tensors

    class DummyData:
        """Test class DummyData."""

        id = "n"

    class DummyGraph(MockTestStub):
        """Test class DummyGraph."""

        nodes: dict[str, dict[str, str]] = {"n": {}}

    class DummyTensor1(Tensor):
        """Test class DummyTensor1."""

        @property
        def requires_grad(self) -> bool:
            """Test function requires_grad.

            Returns:
                bool: Computed output value from `requires_grad`.
            """
            return True

        @property
        def trainable(self) -> BackendArg:
            """Test function trainable.

            Returns:
                BackendArg: Computed output value from `trainable`.
            """
            return False

    class DummyTensor2(Tensor):
        """Test class DummyTensor2."""

        @property
        def requires_grad(self) -> bool:
            """Test function requires_grad.

            Returns:
                bool: Computed output value from `requires_grad`.
            """
            return False

        @property
        def trainable(self) -> BackendArg:
            """Test function trainable.

            Returns:
                BackendArg: Computed output value from `trainable`.
            """
            return True

    t = DummyTensor1(None, TensorConfig((), DType.Float32, None))
    pytest.MonkeyPatch().setattr(t, "_data", DummyData(), raising=False)
    t2 = DummyTensor2(None, TensorConfig((), DType.Float32, None))
    pytest.MonkeyPatch().setattr(t2, "_data", DummyData(), raising=False)

    class DummyTensor3(Tensor):
        """Test class DummyTensor3."""

        @property
        def requires_grad(self) -> bool:
            """Test function requires_grad.

            Returns:
                bool: Computed output value from `requires_grad`.
            """
            return True

    class DummyDataNotInGraph:
        """Test class DummyDataNotInGraph."""

        id = "not_in_graph"

    t3 = DummyTensor3(None, TensorConfig((), DType.Float32, None))
    pytest.MonkeyPatch().setattr(t3, "_data", DummyDataNotInGraph(), raising=False)

    wrt_tensors, wrt_ids = _find_wrt_tensors(DummyGraph())
    assert id(t) in [id(x) for x in wrt_tensors]
    assert id(t2) in [id(x) for x in wrt_tensors]
    assert id(t3) not in [id(x) for x in wrt_tensors]
    assert "n" in wrt_ids


def test_get_concrete_val_variants() -> None:
    """Test _get_concrete_val with non-proxy and proxy objects."""
    from ml_switcheroo_compiler.grad.utils import _get_concrete_val

    class NonProxy(MockTestStub):
        """Test class NonProxy."""

        _data = 42

    assert _get_concrete_val(NonProxy()) == 42


def test_get_inputs_dict_success(mocker: MockerFixture) -> None:
    """Test _get_inputs_dict full branch coverage including successful inputs lookup.

    Args:
        mocker: Test parameter `mocker`.
    """
    from ml_switcheroo_compiler.core.tensor import DType, Tensor, TensorConfig
    from ml_switcheroo_compiler.tracing.tracer import ProxyTensor

    class DummyBackend:
        """Test class DummyBackend."""

        def asarray(self, val: BackendArg) -> BackendArg:
            """Test function asarray.

            Args:
                val: Test parameter `val`.

            Returns:
                BackendArg: Computed output value from `asarray`.
            """
            return val

    mocker.patch("ml_switcheroo_compiler.grad.utils.get_active_backend", return_value=DummyBackend())

    class NodeData:
        """Test class NodeData."""

        def __init__(self, id: BackendArg, concrete_value: BackendArg = None) -> None:
            """Test function __init__.

            Args:
                id: Test parameter `id`.
                concrete_value: Test parameter `concrete_value`.
            """
            self.id = id
            self.concrete_value = concrete_value

    class DummyGraphSuccess(MockTestStub):
        """Test class DummyGraphSuccess."""

        inputs = ["in_valid"]
        nodes: dict[str, dict[str, str]] = {"in_valid": {}, "no_val_node": {}}

    t_valid = Tensor(None, TensorConfig((), DType.Float32, None))
    pytest.MonkeyPatch().setattr(t_valid, "_data", NodeData("in_valid", concrete_value=123.0), raising=False)

    t_no_val = Tensor(None, TensorConfig((), DType.Float32, None))
    proxy = ProxyTensor("no_val_node", ())
    pytest.MonkeyPatch().setattr(proxy, "concrete_value", None, raising=False)
    t_no_val._data = proxy

    t_not_in_graph = Tensor(None, TensorConfig((), DType.Float32, None))
    pytest.MonkeyPatch().setattr(t_not_in_graph, "_data", NodeData("orphan_node", concrete_value=999.0), raising=False)

    Tensor(1.0, TensorConfig((), DType.Float32, None))

    res = _get_inputs_dict(DummyGraphSuccess())
    assert "in_valid" in res
    assert res["in_valid"] == 123.0
    assert "no_val_node" not in res
    assert "orphan_node" not in res

    class GraphNoInputs(MockTestStub):
        """Test class GraphNoInputs."""

        nodes: dict[str, LogicalNode] = {}

    res_no_inputs = _get_inputs_dict(GraphNoInputs())
    assert isinstance(res_no_inputs, dict)


def test_grad_utils_find_wrt_tensors_branches() -> None:
    """Verify _find_wrt_tensors branch coverage when tensors have variable requirements."""
    graph = LogicalGraph(name="test_graph")
    graph.nodes = NodeDict(
        {
            "node_1": LogicalNode(id="node_1", op_type="Input"),
            "node_2": LogicalNode(id="node_2", op_type="Input"),
            "node_3": LogicalNode(id="node_3", op_type="Input"),
        }
    )

    proxy1 = ProxyTensor(id="node_1", shape=(), dtype="float32")
    t1 = Tensor(proxy1, TensorConfig((), DType.Float32, Device("cpu")))
    t1._requires_grad = True

    proxy2 = ProxyTensor(id="node_2", shape=(), dtype="float32")
    t2 = Tensor(proxy2, TensorConfig((), DType.Float32, Device("cpu")))
    pytest.MonkeyPatch().setattr(t2, "trainable", True, raising=False)

    proxy3 = ProxyTensor(id="node_3", shape=(), dtype="float32")
    t3 = Variable(proxy3, TensorConfig((), DType.Float32, Device("cpu")))

    # An unrelated tensor
    proxy_unrelated = ProxyTensor(id="unrelated_node", shape=(), dtype="float32")
    t_unrelated = Tensor(proxy_unrelated, TensorConfig((), DType.Float32, Device("cpu")))
    t_unrelated._requires_grad = True

    # Tensor without .id on data
    t_no_id = Tensor("raw_string", TensorConfig((), DType.Float32, Device("cpu")))

    # Tensor not in graph.nodes
    proxy_not_in_graph = ProxyTensor(id="node_not_in_graph", shape=(), dtype="float32")
    t_not_in_graph = Tensor(proxy_not_in_graph, TensorConfig((), DType.Float32, Device("cpu")))
    t_not_in_graph._requires_grad = True

    # Tensor in graph with no requires_grad/trainable/Variable
    proxy_no_grad = ProxyTensor(id="node_1", shape=(), dtype="float32")
    t_no_grad = Tensor(proxy_no_grad, TensorConfig((), DType.Float32, Device("cpu")))

    # Mock gc.get_objects to return these specific tensors
    with patch(
        "gc.get_objects", return_value=[t1, t2, t3, t_unrelated, t_no_id, t_not_in_graph, t_no_grad, "not_a_tensor"]
    ):
        wrt_tensors, wrt_ids = _find_wrt_tensors(graph)
        assert any(t is t1 for t in wrt_tensors)
        assert any(t is t2 for t in wrt_tensors)
        assert any(t is t3 for t in wrt_tensors)
        assert not any(t is t_unrelated for t in wrt_tensors)
        assert not any(t is t_no_id for t in wrt_tensors)
        assert not any(t is t_not_in_graph for t in wrt_tensors)
        assert not any(t is t_no_grad for t in wrt_tensors)
        assert "node_1" in wrt_ids
        assert "node_2" in wrt_ids
        assert "node_3" in wrt_ids


def test_grad_utils_to_original_type_branches() -> None:
    """Verify _to_original_type branches when converting back to tensors."""
    cfg = TensorConfig((2,), DType.Float32, Device("cpu"))
    orig_tensor = Tensor(np.array([1.0, 2.0], dtype=np.float32), cfg)

    # 1. val is already a Tensor
    already_tensor = Tensor(np.array([3.0, 4.0], dtype=np.float32), cfg)
    res1 = _to_original_type(already_tensor, orig_tensor)
    assert res1 is already_tensor

    # 2. bool array to Tensor
    bool_arr = np.array([True, False], dtype=np.bool_)
    res2 = _to_original_type(bool_arr, orig_tensor)
    assert isinstance(res2, Tensor)
    assert res2.dtype == DType.Bool

    # 3. int array to Tensor
    int_arr = np.array([1, 2], dtype=np.int32)
    res3 = _to_original_type(int_arr, orig_tensor)
    assert isinstance(res3, Tensor)
    assert res3.dtype == DType.Int32


def test_grad_utils_compute_grad_and_value_tracing_branch() -> None:
    """Verify _compute_grad_and_value under global_tracing_state with active graph."""
    graph = LogicalGraph(name="trace_compute_graph")
    graph.nodes = NodeDict({})

    def f_linear(x: Tensor) -> Tensor:
        """Verify the behavior and output invariants of f linear.

        Args:
            x: Test parameter `x`.

        Returns:
            Tensor: Computed output value from `f_linear`.
        """
        return x

    proxy = ProxyTensor(id="in_1", shape=(2,), dtype="float32")
    t_in = Tensor(proxy, TensorConfig((2,), DType.Float32, Device("cpu")))

    global_tracing_state.start_tracing("trace_compute_graph")
    global_tracing_state.active_graph = graph
    try:
        opts = GradOptions()
        # Mock vjp inside _compute_grad_and_value
        vjp_fn_mock = MagicMock(
            return_value=(Tensor(np.array([1.0, 1.0]), TensorConfig((2,), DType.Float32, Device("cpu"))),)
        )
        with patch("ml_switcheroo_compiler.grad.jvp_vjp.vjp", return_value=(t_in, vjp_fn_mock)):
            val, res_grad = _compute_grad_and_value(f_linear, opts, (t_in,))
            assert val is t_in
            assert res_grad is not None
            assert any(node.op_type == "Constant" for node in graph.nodes.values())
    finally:
        global_tracing_state.stop_tracing()


def test_as_primal_tree_mapping_and_array_coercion() -> None:
    """Verify `as_primal_tree` converts mapping leaves and array-coercible inputs."""
    from ml_switcheroo_compiler.core.shape import ShapeProtocol
    from ml_switcheroo_compiler.grad.utils import as_primal_tree

    mapped = as_primal_tree({"w": (1.0, [2.0])})
    assert mapped == {"w": (1.0, [2.0])}
    coerced = as_primal_tree("3.5")
    assert isinstance(coerced, ShapeProtocol)
    assert coerced.shape == ()
