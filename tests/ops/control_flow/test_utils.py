"""Behavioral unit tests for utils."""

from __future__ import annotations

import typing

import numpy as np
import pytest
from ml_switcheroo_ir import LogicalGraph, NodeDict

from ml_switcheroo_compiler.backends.eager_registry import BackendArg
from ml_switcheroo_compiler.core.device import Device
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.ops.control_flow_utils import (
    SubgraphWithId,
    _get_tensor_ids,
    _process_trace_outputs,
    _trace_function,
    _wrap_proxy_inputs,
)
from tests.testing_utils import MockGraphStub, MockTestStub, adapt_backend_module

if typing.TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import BackendArg


class DummyData(MockTestStub):
    """Test class DummyData."""

    id = "id"


def test_control_flow_utils() -> None:
    """Test function test_control_flow_utils."""
    t_cfg = TensorConfig((1,), DType.Float32, "cpu")
    t = Tensor(data=DummyData(), config=t_cfg)

    g = LogicalGraph("g")
    # wrap proxy inputs
    ids, args = _wrap_proxy_inputs(((t,), 42), g)
    assert len(ids) == 1
    assert args[1] == 42
    assert isinstance(args[0], tuple)

    # subgraph without input_specs attribute
    class SubgraphNoSpecs(MockGraphStub):
        """Test class SubgraphNoSpecs."""

        def __init__(self) -> None:
            """Test function __init__."""
            self.nodes: NodeDict = NodeDict({})

    g_nospecs = SubgraphNoSpecs()
    ids_nospecs, args_nospecs = _wrap_proxy_inputs((t,), g_nospecs)
    assert len(ids_nospecs) == 1
    assert len(args_nospecs) == 1

    # get tensor ids
    assert _get_tensor_ids(t) == ["id"]
    assert _get_tensor_ids([t]) == ["id"]

    # Nested Tensor unwrap (line 78)
    t_inner = Tensor(data=DummyData(), config=t_cfg)
    t_outer = Tensor(data=t_inner, config=t_cfg)
    assert _get_tensor_ids(t_outer) == ["id"]

    # Direct id on Tensor (lines 81-82)
    class DirectIdTensor(Tensor):
        """Test class DirectIdTensor."""

        pass

    t_direct = DirectIdTensor(data="no_id", config=t_cfg)
    pytest.MonkeyPatch().setattr(t_direct, "id", "direct_123", raising=False)
    assert _get_tensor_ids(t_direct) == ["direct_123"]

    # No id anywhere (line 83)
    t_no_id = Tensor(data="no_id", config=t_cfg)
    if adapt_backend_module(t_no_id).has_symbol("id"):
        pytest.MonkeyPatch().delattr(t_no_id, "id")
    generated_ids = _get_tensor_ids(t_no_id)
    assert len(generated_ids) == 1
    assert len(generated_ids[0]) > 0

    with pytest.raises(TypeError):
        _get_tensor_ids(42)

    # process trace outputs
    g2 = LogicalGraph("g2")
    out_id = _process_trace_outputs(t, g2)
    assert out_id in g2.nodes

    # trace function
    def my_fn(x: BackendArg) -> BackendArg:
        """Test function my_fn.

        Args:
            x: Test parameter `x`.

        Returns:
            BackendArg: Computed output value from `my_fn`.
        """
        return x

    block = _trace_function(my_fn, (t,), "my_block")
    assert isinstance(block, SubgraphWithId)
    assert block.id == "my_block"
    assert block.name == "my_block"
    assert len(block.inputs) == 1
    assert len(block.outputs) == 1
    assert block.inputs[0] in block.input_specs


class _PayloadWithId(MockTestStub):
    """Payload object with custom id."""

    def __init__(self, obj_id: str) -> None:
        """Initialize payload.

        Args:
            obj_id (str): ID string.
        """
        self.id = obj_id


def test_control_flow_utils_wrap_proxy_inputs_tuple() -> None:
    """Cover _wrap_proxy_inputs branch when arg is a tuple."""
    graph = LogicalGraph(name="test_graph")
    graph.nodes = NodeDict({})
    graph.input_specs = {}

    cfg = TensorConfig(shape=(2,), dtype=DType.Float32, device=Device("cpu"))
    t1 = Tensor(np.array([1.0, 2.0], dtype=np.float32), cfg)
    t2 = Tensor(np.array([3.0, 4.0], dtype=np.float32), cfg)

    nested_args = ((t1, t2), "non_tensor")
    input_ids, proxy_args = _wrap_proxy_inputs(nested_args, graph)

    assert len(input_ids) == 2
    assert isinstance(proxy_args[0], tuple)
    assert proxy_args[1] == "non_tensor"


def test_control_flow_utils_wrap_proxy_inputs_no_input_specs() -> None:
    """Cover _wrap_proxy_inputs branch when subgraph does not have input_specs."""
    graph = LogicalGraph(name="test_graph_no_specs")
    graph.nodes = NodeDict({})
    if adapt_backend_module(graph).has_symbol("input_specs"):
        pytest.MonkeyPatch().delattr(graph, "input_specs")

    cfg = TensorConfig(shape=(2,), dtype=DType.Float32, device=Device("cpu"))
    t1 = Tensor(np.array([1.0, 2.0], dtype=np.float32), cfg)

    input_ids, proxy_args = _wrap_proxy_inputs((t1,), graph)
    assert len(input_ids) == 1
    assert len(proxy_args) == 1


def test_control_flow_utils_get_tensor_ids_branches() -> None:
    """Cover _get_tensor_ids branches for Tensor with id, list/tuple, and invalid types."""
    cfg = TensorConfig(shape=(2,), dtype=DType.Float32, device=Device("cpu"))

    # Nested tensor unwrapping (cur.data is a Tensor)
    inner_payload = _PayloadWithId("inner_id_42")
    inner_t = Tensor(inner_payload, cfg)
    outer_t = Tensor(inner_t, cfg)
    ids_nested = _get_tensor_ids(outer_t)
    assert ids_nested == ["inner_id_42"]

    # Tensor where cur has no data.id but has id on Tensor
    t_no_id_payload = Tensor(np.array([1.0]), cfg)
    pytest.MonkeyPatch().setattr(t_no_id_payload, "id", "tensor_attr_id", raising=False)
    assert _get_tensor_ids(t_no_id_payload) == ["tensor_attr_id"]

    # Tensor where cur has neither data.id nor id (generates uuid)
    pytest.MonkeyPatch().delattr(t_no_id_payload, "id")
    ids_uuid = _get_tensor_ids(t_no_id_payload)
    assert len(ids_uuid) == 1
    assert isinstance(ids_uuid[0], str)

    # list/tuple of tensors
    t_a = Tensor(_PayloadWithId("a"), cfg)
    t_b = Tensor(_PayloadWithId("b"), cfg)
    ids_list = _get_tensor_ids([t_a, (t_b,)])
    assert ids_list == ["a", "b"]

    # Invalid type
    with pytest.raises(TypeError, match="Control flow functions must return a Tensor"):
        _get_tensor_ids("invalid_string_output")
