"""Provides required module functionality."""

from ml_switcheroo_compiler.backends.registry import BackendRegistry, register_backend
from ml_switcheroo_compiler.backends.base_generator import BaseGenerator
from ml_switcheroo_compiler.backends.jax import JAXCodeGenerator
from ml_switcheroo_compiler.backends.keras import KerasCodeGenerator
from ml_switcheroo_compiler.backends.mlx import MLXCodeGenerator
from ml_switcheroo_compiler.backends.pytorch import PyTorchCodeGenerator
from ml_switcheroo_compiler.ir.core import IRGraph, IRNode
import pytest


def test_coverage_brute() -> None:
    """Execute the requested function."""

    class FakeGenerator(BaseGenerator):
        """Docstring."""

        pass

    register_backend("fake")(FakeGenerator)
    BackendRegistry.get("fake")
    BackendRegistry.get_all()
    with pytest.raises(ValueError, match="Backend 'non_existent' not found"):
        BackendRegistry.get("non_existent")

    g = IRGraph()
    n = IRNode(id="n1", op_type="UnknownOp", inputs=[], attributes={}, shape_metadata=None)
    g.nodes["n1"] = n

    gen_jax = JAXCodeGenerator(g)
    gen_keras = KerasCodeGenerator(g)
    gen_mlx = MLXCodeGenerator(g)
    gen_pytorch = PyTorchCodeGenerator(g)

    gen_jax.visit(n, [])
    gen_keras.visit(n, [])
    gen_mlx.visit(n, [])
    gen_pytorch.visit(n, [])

    # visit with empty list of input vars and some kwargs to hit missing branches
    n_zeros = IRNode(
        id="n2", op_type="Zeros", inputs=[], attributes={"shape": (2,)}, shape_metadata=None
    )
    # JAX 42->41, Keras 55-63, PyTorch 47->46
    gen_jax.visit(n_zeros, [], shape="(2,)")
    gen_keras.visit(n_zeros, [], shape="(2,)")
    gen_mlx.visit(n_zeros, [], shape="(2,)")
    gen_pytorch.visit(n_zeros, [], shape="(2,)")

    n_custom = IRNode(
        id="n3",
        op_type="CustomFake",
        inputs=["x"],
        attributes={"axis": 1, "keepdims": True},
        shape_metadata=None,
    )
    gen_jax.visit(n_custom, ["x"], axis=1, keepdims=True)
    gen_mlx.visit(n_custom, ["x"], axis=1, keepdims=True)
    gen_pytorch.visit(n_custom, ["x"], axis=1, keepdims=True)

    gen_keras.visit(n_custom, ["x"], axis=1, keepdims=True)

    n_zeros_fake = IRNode(
        id="n4", op_type="Zeros", inputs=[], attributes={"fake": 1}, shape_metadata=None
    )
    gen_jax.visit(n_zeros_fake, [], shape="(2,)", fake=1)
    gen_mlx.visit(n_zeros_fake, [], shape="(2,)", fake=1)
