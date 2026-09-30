"""Tests for test_control_flow_utils_coverage."""

from __future__ import annotations

import numpy as np
import pytest
from ml_switcheroo_ir import LogicalGraph

from ml_switcheroo_compiler.core.device import Device
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.ops.control_flow_utils import (
    _get_tensor_ids,
    _wrap_proxy_inputs,
)


class _PayloadWithId:
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
    graph.nodes = {}
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
    graph.nodes = {}
    if hasattr(graph, "input_specs"):
        delattr(graph, "input_specs")

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
    t_no_id_payload.id = "tensor_attr_id"  # type: ignore[attr-defined]
    assert _get_tensor_ids(t_no_id_payload) == ["tensor_attr_id"]

    # Tensor where cur has neither data.id nor id (generates uuid)
    delattr(t_no_id_payload, "id")
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
