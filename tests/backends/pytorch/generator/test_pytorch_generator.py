"""Tests for test_pytorch_generator."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from ml_switcheroo_compiler.backends.pytorch.generator import PyTorchCodeGenerator
from ml_switcheroo_compiler.backends.pytorch.pytorch_mixins import PyTorchScatterVisitor
from ml_switcheroo_compiler.ir.core import IRGraph

if TYPE_CHECKING:
    from collections.abc import Callable

    from ml_switcheroo_compiler.backends.eager_registry import EagerTensorProtocol, EagerValue


class DummyNode:
    """Test class DummyNode."""

    def __init__(self, attrs: EagerValue = None, op_type: EagerValue = "Unknown") -> None:
        """Test function __init__."""
        self.attributes = attrs or {}
        self.op_type = op_type
        self.id = "n1"


def test_generator_basics(monkeypatch: EagerTensorProtocol) -> None:
    """Test function test_generator_basics."""
    import numpy as np
    import torch

    g = IRGraph()
    gen = PyTorchCodeGenerator(g)
    assert gen.get_fallback_prefix() == "torch"
    n_power = DummyNode({"num_iters": 2}, "PowerIteration")
    res = cast("Callable[..., EagerTensorProtocol]", gen.visit_PowerIteration)(n_power, ["x", "u"])
    assert res == "pt_power_iteration(x, 2, u)"
    res = cast("Callable[..., EagerTensorProtocol]", gen.visit_PowerIteration)(n_power, ["x"])
    assert res == "pt_power_iteration(x, 2, None)"
    n_ragged = DummyNode({}, "RaggedDot")
    assert cast("Callable[..., EagerTensorProtocol]", gen.visit_RaggedDot)(n_ragged, ["x", "y"]) == "pt_ragged_dot(x, y)"
    n_einsum = DummyNode({}, "Einsum")
    assert cast("Callable[..., EagerTensorProtocol]", gen.visit_Einsum)(n_einsum, ["x", "y"], equation="ij,jk->ik") == "torch.einsum('ij,jk->ik', x, y)"
    gen._emit_constant_assignment("var", "val")
    assert "var = self.var" in gen.code[0]
    prefix = gen._get_prefix_code()
    assert "import torch" in prefix[0]
    n_const = DummyNode({"value": 1.0}, "Constant")
    setattr(gen, "sorted_nodes", [n_const])
    gen.code = []
    assert gen._emit_init_body() is True
    assert "register_parameter" in gen.code[0]
    setattr(gen, "sorted_nodes", [cast("Callable[..., EagerTensorProtocol]", DummyNode)({}, "Unknown")])
    assert gen._emit_init_body() is False
    monkeypatch.setattr(torch, "save", lambda *args, **kwargs: "mock_save")
    monkeypatch.setattr(torch, "load", lambda *args, **kwargs: "mock_load")
    assert PyTorchCodeGenerator.save("path", np.array([1, 2, 3])) is None
    assert PyTorchCodeGenerator.load("path.npy") == "mock_load"
    assert PyTorchCodeGenerator.savez("path") is None
    assert PyTorchCodeGenerator.savez_compressed("path") is None


def test_pytorch_scatter_visitor() -> None:
    """Test function test_pytorch_scatter_visitor."""
    vis = PyTorchScatterVisitor()
    node = DummyNode()
    assert cast("Callable[..., EagerTensorProtocol]", vis.visit_TensorScatterUpdate)(node, ["t", "i", "u"]) == "t.clone().index_put_(tuple(i.unbind(-1)), u)"
    assert cast("Callable[..., EagerTensorProtocol]", vis.visit_TensorScatterAdd)(node, ["t", "i", "u"]) == "t.clone().index_put_(tuple(i.unbind(-1)), u, accumulate=True)"
    assert "scatter_reduce_(0, sum(i[..., d] * t.stride(d) for d in range(i.shape[-1])).flatten(), u.flatten(), reduce='amax', include_self=True)" in cast("Callable[..., EagerTensorProtocol]", vis.visit_TensorScatterMax)(node, ["t", "i", "u"])
    assert "scatter_reduce_(0, sum(i[..., d] * t.stride(d) for d in range(i.shape[-1])).flatten(), u.flatten(), reduce='amin', include_self=True)" in cast("Callable[..., EagerTensorProtocol]", vis.visit_TensorScatterMin)(node, ["t", "i", "u"])


def test_pytorch_generator_distributed_visitor() -> None:
    """Test function test_pytorch_generator_distributed_visitor."""
    from ml_switcheroo_compiler.backends.pytorch.pytorch_mixins import PyTorchDistributedVisitor

    vis = PyTorchDistributedVisitor()
    assert cast("Callable[..., EagerTensorProtocol]", vis.visit_AllGather)(None, ["tensor"]) == "torch.distributed.all_gather_into_tensor(output, tensor)"
    assert cast("Callable[..., EagerTensorProtocol]", vis.visit_ReduceScatter)(None, ["tensor"]) == "torch.distributed.reduce_scatter_tensor(torch.empty_like(tensor), tensor)"
    assert cast("Callable[..., EagerTensorProtocol]", vis.visit_AllReduce)(None, ["tensor"]) == "torch.distributed.all_reduce(tensor)"
    assert cast("Callable[..., EagerTensorProtocol]", vis.visit_AllToAll)(None, ["tensor"]) == "torch.distributed.all_to_all_single(torch.empty_like(tensor), tensor)"

    class DummyBcNode:
        """Test class DummyBcNode."""

        attributes = {"root": 2}

    assert cast("Callable[..., EagerTensorProtocol]", vis.visit_Broadcast)(cast("Callable[..., EagerTensorProtocol]", DummyBcNode)(), ["tensor"]) == "torch.distributed.broadcast(tensor, src=2)"
    from ml_switcheroo_compiler.backends.pytorch.pytorch_mixins import PyTorchNNMixin

    nn_vis = PyTorchNNMixin()
    lines = cast("list[EagerTensorProtocol]", [])
    setattr(nn_vis, "add_line", lines.append)

    class DummyNode:
        """Test class DummyNode."""

        def __init__(self, attrs: EagerValue = None) -> None:
            """Test function __init__."""
            self.attributes = attrs or {}
            self.id = "n1"

    res_send = cast("Callable[..., EagerTensorProtocol]", nn_vis.visit_Send)(cast("Callable[..., EagerTensorProtocol]", DummyNode)({"dst_rank": 1, "tag": 2}), ["tensor"])
    assert res_send == ""
    assert "torch.distributed.isend(tensor, dst=1, tag=2)" in lines[0]
    res_recv = cast("Callable[..., EagerTensorProtocol]", nn_vis.visit_Recv)(cast("Callable[..., EagerTensorProtocol]", DummyNode)({"src_rank": 1, "tag": 2, "shape": (2, 2), "dtype": "float32"}), [])
    assert res_recv == "v_n1"
    assert "v_n1 = torch.empty([2, 2], dtype=torch.float32, device=self.device)" in lines[1]
    assert "torch.distributed.irecv(v_n1, src=1, tag=2)" in lines[2]
    from unittest.mock import MagicMock

    mock_gen = MagicMock()
    nn_with_gen = PyTorchNNMixin(generator=mock_gen)
    nn_with_gen._add_line("code_line")
    mock_gen.add_line.assert_called_with("code_line")
    nn_no_gen = PyTorchNNMixin(generator=None)
    nn_no_gen._add_line("noop_line")


def test_missing_methods() -> None:
    """Test function test_missing_methods."""
    g = IRGraph()
    gen = PyTorchCodeGenerator(g)
    assert gen.get_fallback_prefix() == "torch"
    assert gen.get_fallback_axis_kwarg() == "dim"
    assert gen.get_fallback_keepdims_kwarg() == "keepdim"
    assert gen._get_math_ops({}) is not None
    assert gen._get_creation_ops({}) is not None
    assert gen._get_array_ops({}) is not None
    assert gen.get_ops_map({}) is not None
    cast("Callable[..., EagerTensorProtocol]", gen.visit)(cast("Callable[..., EagerTensorProtocol]", DummyNode)({}, "SomeUnknownOp"), [])
