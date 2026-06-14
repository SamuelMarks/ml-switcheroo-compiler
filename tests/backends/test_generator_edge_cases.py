"""Tests generator edge cases."""

import pytest

from ml_switcheroo_compiler.backends.base_generator import BaseGenerator
from ml_switcheroo_compiler.backends.jax import JAXCodeGenerator
from ml_switcheroo_compiler.backends.keras import KerasCodeGenerator
from ml_switcheroo_compiler.backends.mlx import MLXCodeGenerator
from ml_switcheroo_compiler.backends.pytorch import PyTorchCodeGenerator
from ml_switcheroo_compiler.backends.registry import BackendRegistry, register_backend
from ml_switcheroo_compiler.ir.core import IRGraph, IRNode


def test_coverage_brute() -> None:
    """Execute generator methods for edge cases."""

    class FakeGenerator(BaseGenerator):
        """Docstring."""

    register_backend("fake")(FakeGenerator)
    assert BackendRegistry.get("fake") == FakeGenerator
    assert "fake" in BackendRegistry.get_all()
    with pytest.raises(ValueError, match="Backend 'non_existent' not found"):
        BackendRegistry.get("non_existent")

    g = IRGraph()
    n = IRNode(id="n1", op_type="UnknownOp", inputs=[], attributes={}, shape_metadata=None)
    g.nodes["n1"] = n

    gen_jax = JAXCodeGenerator(g)
    gen_keras = KerasCodeGenerator(g)
    gen_mlx = MLXCodeGenerator(g)
    gen_pytorch = PyTorchCodeGenerator(g)

    assert "unknownop" in gen_jax.visit(n, []).lower()
    assert "unknownop" in gen_keras.visit(n, []).lower()
    assert "unknownop" in gen_mlx.visit(n, []).lower()
    assert "unknownop" in gen_pytorch.visit(n, []).lower()

    # visit with empty list of input vars and some kwargs to hit missing branches
    n_zeros = IRNode(
        id="n2",
        op_type="Zeros",
        inputs=[],
        attributes={"shape": (2,)},
        shape_metadata=None,
    )
    # JAX 42->41, Keras 55-63, PyTorch 47->46
    assert "zeros" in gen_jax.visit(n_zeros, [], shape="(2,)").lower()
    assert "zeros" in gen_keras.visit(n_zeros, [], shape="(2,)").lower()
    assert "zeros" in gen_mlx.visit(n_zeros, [], shape="(2,)").lower()
    assert "zeros" in gen_pytorch.visit(n_zeros, [], shape="(2,)").lower()

    n_custom = IRNode(
        id="n3",
        op_type="CustomFake",
        inputs=["x"],
        attributes={"axis": 1, "keepdims": True},
        shape_metadata=None,
    )
    assert "customfake" in gen_jax.visit(n_custom, ["x"], axis=1, keepdims=True).lower()
    assert "customfake" in gen_mlx.visit(n_custom, ["x"], axis=1, keepdims=True).lower()
    assert "customfake" in gen_pytorch.visit(n_custom, ["x"], axis=1, keepdims=True).lower()
    assert "customfake" in gen_keras.visit(n_custom, ["x"], axis=1, keepdims=True).lower()

    n_zeros_fake = IRNode(
        id="n4",
        op_type="Zeros",
        inputs=[],
        attributes={"fake": 1},
        shape_metadata=None,
    )
    assert "zeros" in gen_jax.visit(n_zeros_fake, [], shape="(2,)", fake=1).lower()
    assert "zeros" in gen_mlx.visit(n_zeros_fake, [], shape="(2,)", fake=1).lower()


def test_cupy_dask_kwargs_only_coverage() -> None:
    """Test cupy and dask missing branches."""
    from ml_switcheroo_compiler.backends.cupy.generator import CupyGenerator
    from ml_switcheroo_compiler.backends.dask.generator import DaskGenerator

    g = IRGraph()
    n = IRNode(
        id="n_kw",
        op_type="KwOp",
        inputs=[],
        attributes={"shape": (2,)},
        shape_metadata=None,
    )
    g.nodes["n_kw"] = n

    # Needs a real mock import so it actually loads cupy and dask classes from somewhere
    # Actually they are already registered if the dependencies are present.
    try:
        gen_cupy = CupyGenerator(g)
        assert "kwop" in gen_cupy.visit(n, [], shape="(2,)").lower()
    except NameError:
        pass  # Not available

    try:
        gen_dask = DaskGenerator(g)
        assert "kwop" in gen_dask.visit(n, [], shape="(2,)").lower()
    except NameError:
        pass
