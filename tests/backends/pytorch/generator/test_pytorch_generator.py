from ml_switcheroo_compiler.backends.pytorch.generator import PyTorchAudioVisitor, PyTorchCodeGenerator, PyTorchVisionVisitor
from ml_switcheroo_compiler.backends.pytorch.pytorch_mixins import PyTorchDistributedVisitor, PyTorchLinalgMixin, PyTorchNNMixin, PyTorchScatterVisitor
from ml_switcheroo_compiler.ir.core import IRGraph


class DummyNode:
    def __init__(self, attrs=None, op_type="Unknown"):
        self.attributes = attrs or {}
        self.op_type = op_type
        self.id = "n1"


def test_generator_basics(monkeypatch):
    import numpy as np
    import torch

    g = IRGraph()
    gen = PyTorchCodeGenerator(g)
    assert gen._get_backend_prefix() == "pt"

    # visit_PowerIteration
    n_power = DummyNode({"num_iters": 2}, "PowerIteration")
    res = gen.visit_PowerIteration(n_power, ["x", "u"])
    assert res == "pt_power_iteration(x, 2, u)"
    res = gen.visit_PowerIteration(n_power, ["x"])
    assert res == "pt_power_iteration(x, 2, None)"

    # visit_RaggedDot
    n_ragged = DummyNode({}, "RaggedDot")
    assert gen.visit_RaggedDot(n_ragged, ["x", "y"]) == "pt_ragged_dot(x, y)"

    # visit_Einsum
    n_einsum = DummyNode({}, "Einsum")
    assert gen.visit_Einsum(n_einsum, ["x", "y"], equation="ij,jk->ik") == "torch.einsum('ij,jk->ik', x, y)"

    # _emit_constant_assignment
    gen._emit_constant_assignment("var", "val")
    assert "var = self.var" in gen.code[0]

    # _get_prefix_code
    prefix = gen._get_prefix_code()
    assert "import torch" in prefix[0]

    # _emit_init_body
    n_const = DummyNode({"value": 1.0}, "Constant")
    gen.sorted_nodes = [n_const]
    gen.code = []
    assert gen._emit_init_body() is True
    assert "register_parameter" in gen.code[0]

    gen.sorted_nodes = [DummyNode({}, "Unknown")]
    assert gen._emit_init_body() is False

    # Audio & Vision delegates
    # Add dummy handled ops
    assert gen.visit(DummyNode({}, "Istft"), ["x"]) == "torch.istft(x)"
    assert gen.visit(DummyNode({}, "ElasticTransform"), ["x", "y"]) == "torchvision.transforms.functional.elastic_transform(x, y)"

    # Empty handlers
    assert PyTorchAudioVisitor().visit(DummyNode({}, "Unknown"), []) == ""
    assert PyTorchVisionVisitor().visit(DummyNode({}, "Unknown"), []) == ""

    # Mock torch load and save
    monkeypatch.setattr(torch, "save", lambda *args, **kwargs: "mock_save")
    monkeypatch.setattr(torch, "load", lambda *args, **kwargs: "mock_load")

    # Save/load static methods
    assert PyTorchCodeGenerator.save("path", np.array([1, 2, 3])) is None
    assert PyTorchCodeGenerator.load("path.npy") == "mock_load"
    assert PyTorchCodeGenerator.savez("path") is None
    assert PyTorchCodeGenerator.savez_compressed("path") is None


def test_pytorch_scatter_visitor():
    vis = PyTorchScatterVisitor()
    node = DummyNode()
    assert vis.visit_TensorScatterUpdate(node, ["t", "i", "u"]) == "t.clone().index_put_(tuple(i.unbind(-1)), u)"
    assert vis.visit_TensorScatterAdd(node, ["t", "i", "u"]) == "t.clone().index_put_(tuple(i.unbind(-1)), u, accumulate=True)"
    assert "scatter_reduce_(0, sum(i[..., d] * t.stride(d) for d in range(i.shape[-1])).flatten(), u.flatten(), reduce='amax', include_self=True)" in vis.visit_TensorScatterMax(node, ["t", "i", "u"])
    assert "scatter_reduce_(0, sum(i[..., d] * t.stride(d) for d in range(i.shape[-1])).flatten(), u.flatten(), reduce='amin', include_self=True)" in vis.visit_TensorScatterMin(node, ["t", "i", "u"])


def test_pytorch_distributed_visitor():
    vis = PyTorchDistributedVisitor()
    node = DummyNode()
    assert vis.visit_all_gather(node, ["t"]) == "torch.distributed.all_gather_into_tensor(torch.empty_like(t), t)"
    assert vis.visit_reduce_scatter(node, ["t"]) == "torch.distributed.reduce_scatter_tensor(torch.empty_like(t), t)"
    assert vis.visit_all_reduce(node, ["t"]) == "torch.distributed.all_reduce(t)"


def test_pytorch_linalg_mixin():
    vis = PyTorchLinalgMixin()
    ops = vis._get_linalg_ops({})
    assert "Matmul" in ops
    assert ops["Matmul"] == "torch.matmul({0}, {1})"


def test_pytorch_nn_mixin():
    vis = PyTorchNNMixin()
    assert vis._get_nn_ops({}) == {}


def test_missing_methods():
    g = IRGraph()
    gen = PyTorchCodeGenerator(g)
    assert gen.get_fallback_prefix() == "torch"
    assert gen.get_fallback_axis_kwarg() == "dim"
    assert gen.get_fallback_keepdims_kwarg() == "keepdim"
    assert gen._get_math_ops({}) is not None
    assert gen._get_creation_ops({}) is not None
    assert gen._get_array_ops({}) is not None
    assert gen.get_ops_map({}) is not None

    # cover line 112
    try:
        gen.visit(DummyNode({}, "SomeUnknownOp"), [])
    except Exception:
        pass
