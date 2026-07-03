"""Tests generator edge cases."""

import pytest

from ml_switcheroo_compiler.backends.base_generator import BaseGenerator
from ml_switcheroo_compiler.backends.common.generator_mixins import SharedASTGeneratorVisitor

# cover line 88 in generator_mixins
from ml_switcheroo_compiler.backends.common.mixins.nn import GroupNormConfig
from ml_switcheroo_compiler.backends.cupy.generator import CupyGenerator
from ml_switcheroo_compiler.backends.dask.generator import DaskGenerator
from ml_switcheroo_compiler.backends.jax import JAXCodeGenerator
from ml_switcheroo_compiler.backends.keras import KerasCodeGenerator
from ml_switcheroo_compiler.backends.mlx import MLXCodeGenerator
from ml_switcheroo_compiler.backends.pytorch import PyTorchCodeGenerator
from ml_switcheroo_compiler.backends.registry import BackendRegistry, register_backend
from ml_switcheroo_compiler.ir.core import IRGraph, IRNode


def test_coverage_brute() -> None:
    """Execute generator methods for edge cases."""

    class FakeGenerator(SharedASTGeneratorVisitor, BaseGenerator):
        """Docstring."""

        def _get_backend_prefix(self) -> str:
            """Function docstring."""
            return "fake"

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

    n_scan = IRNode(
        id="n_scan",
        op_type="Scan",
        inputs=["x", "y"],
        attributes={},
        shape_metadata=None,
    )
    gen_fake = FakeGenerator(g)
    assert "fake_scan" in gen_fake.visit(n_scan, ["x", "y"])

    n_switch = IRNode(
        id="n_switch",
        op_type="Switch",
        inputs=["cond", "a", "b"],
        attributes={},
        shape_metadata=None,
    )
    assert "fake_switch" in gen_fake.visit(n_switch, ["cond", "a", "b"])

    code_lines = gen_fake._get_group_norm_code(
        GroupNormConfig(
            "fake",
            "fake.numpy",
            "fake.reshape",
            "fake.mean",
            "fake.var",
            "fake.sqrt",
            "axis",
            "keepdims",
        )
    )
    assert len(code_lines) > 0


def test_cupy_dask_kwargs_only_coverage() -> None:
    """Test cupy and dask missing branches."""
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


def test_add_n_accumulate_n_generation() -> None:
    """Test AddN and AccumulateN AST generation."""

    class MockGenerator(SharedASTGeneratorVisitor):
        """Class docstring."""

        def _get_backend_prefix(self) -> str:
            """Function docstring."""
            return "mock"

        def visit(self, node: object, input_vars: object, **kwargs: object) -> object:
            """Function docstring."""
            return getattr(self, f"visit_{node.op_type}")(node, input_vars, **kwargs)

    gen = MockGenerator()

    # Test AddN
    node1 = IRNode(id="n1", op_type="AddN", inputs=["a", "b", "c"], attributes={}, shape_metadata=None)
    assert gen.visit_AddN(node1, ["a", "b", "c"]) == "a + b + c"
    assert gen.visit_AddN(node1, []) == "0.0"

    # Test AccumulateN
    node2 = IRNode(id="n2", op_type="AccumulateN", inputs=["x", "y"], attributes={}, shape_metadata=None)
    assert gen.visit_AccumulateN(node2, ["x", "y"]) == "x + y"
