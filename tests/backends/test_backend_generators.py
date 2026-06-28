"""Test backend generator basic functionality."""

import pytest
from unittest.mock import MagicMock

from ml_switcheroo_compiler.backends import jax, keras, mlx, pytorch, registry, tensorflow
from ml_switcheroo_compiler.ir.core import IRGraph, IRNode


def test_backends_coverage() -> None:
    """Execute generators for all backends to ensure they don't error."""
    g = IRGraph()

    n1 = IRNode(
        id="n1",
        op_type="Constant",
        inputs=[],
        attributes={"value": [1.0]},
        shape_metadata=None,
    )
    n2 = IRNode(id="n2", op_type="Input", inputs=[], attributes={}, shape_metadata=(2,))
    n3 = IRNode(id="n3", op_type="Add", inputs=["n1", "n2"], attributes={}, shape_metadata=None)

    for n in [n1, n2, n3]:
        g.nodes[n.id] = n

    g.inputs = ["n2"]
    g.outputs = ["n3"]

    for gen_cls in [
        jax.JAXCodeGenerator,
        keras.KerasCodeGenerator,
        mlx.MLXCodeGenerator,
        pytorch.PyTorchCodeGenerator,
        tensorflow.TensorFlowCodeGenerator,
    ]:
        res = gen_cls(g).generate()
        assert isinstance(res, str)
        assert len(res) > 0


def test_registry_coverage() -> None:
    """Test registry functions properly."""
    with pytest.raises(ValueError, match="Backend 'nonexistent' not found"):
        registry.BackendRegistry.get("nonexistent")

    class FakeGen:
        """Docstring."""

    registry.BackendRegistry.register("fake", FakeGen)
    assert registry.BackendRegistry.get("fake") == FakeGen
    assert "fake" in registry.BackendRegistry.get_all()

    @registry.register_backend("fake2")
    class FakeGen2:
        """Docstring."""

    assert registry.BackendRegistry.get("fake2") == FakeGen2


def test_truncate_ops_generation() -> None:
    """Test generation of TruncateDiv and TruncateMod ops."""
    from ml_switcheroo_compiler.ir.core import IRNode
    from ml_switcheroo_compiler.backends.tensorflow.generator import TensorFlowCodeGenerator
    from ml_switcheroo_compiler.backends.pytorch.generator import PyTorchCodeGenerator
    from ml_switcheroo_compiler.backends.jax.generator import JAXCodeGenerator
    from ml_switcheroo_compiler.backends.mlx.generator import MLXCodeGenerator
    from ml_switcheroo_compiler.backends.keras.generator import KerasCodeGenerator
    from ml_switcheroo_compiler.backends.cupy.generator import CupyGenerator
    from ml_switcheroo_compiler.backends.dask.generator import DaskGenerator

    node_div = IRNode(id="n1", op_type="TruncateDiv", inputs=["x", "y"])
    node_mod = IRNode(id="n2", op_type="TruncateMod", inputs=["x", "y"])

    assert (
        TensorFlowCodeGenerator(MagicMock()).generic_visit(node_div, ["x", "y"])
        == "tf.math.truncatediv(x, y)"
    )
    assert (
        TensorFlowCodeGenerator(MagicMock()).generic_visit(node_mod, ["x", "y"])
        == "tf.math.truncatemod(x, y)"
    )
    from ml_switcheroo_compiler.backends.numpy.generator import NumpyASTVisitor

    assert NumpyASTVisitor.visit_TruncateDiv(node_div, ["x", "y"]) == "np.trunc(np.divide(x, y))"
    assert (
        PyTorchCodeGenerator(MagicMock()).generic_visit(node_div, ["x", "y"])
        == "torch.trunc(x / y)"
    )
    assert (
        JAXCodeGenerator(MagicMock()).generic_visit(node_div, ["x", "y"])
        == "jnp.trunc(jnp.divide(x, y))"
    )
    assert (
        MLXCodeGenerator(MagicMock()).generic_visit(node_div, ["x", "y"])
        == "mx.trunc(mx.divide(x, y))"
    )
    assert (
        KerasCodeGenerator(MagicMock()).generic_visit(node_div, ["x", "y"])
        == "keras.ops.trunc(keras.ops.divide(x, y))"
    )
    assert (
        CupyGenerator(MagicMock()).visit_TruncateDiv(node_div, ["x", "y"])
        == "cp.trunc(cp.divide(x, y))"
    )
    assert (
        DaskGenerator(MagicMock()).visit_TruncateDiv(node_div, ["x", "y"])
        == "da.trunc(da.divide(x, y))"
    )

    assert NumpyASTVisitor.visit_TruncateMod(node_mod, ["x", "y"]) == "np.fmod(x, y)"
    assert (
        PyTorchCodeGenerator(MagicMock()).generic_visit(node_mod, ["x", "y"]) == "torch.fmod(x, y)"
    )
    assert JAXCodeGenerator(MagicMock()).generic_visit(node_mod, ["x", "y"]) == "jnp.fmod(x, y)"
    assert MLXCodeGenerator(MagicMock()).generic_visit(node_mod, ["x", "y"]) == "mx.remainder(x, y)"
    assert (
        KerasCodeGenerator(MagicMock()).generic_visit(node_mod, ["x", "y"]) == "keras.ops.mod(x, y)"
    )
    assert CupyGenerator(MagicMock()).visit_TruncateMod(node_mod, ["x", "y"]) == "cp.fmod(x, y)"
    assert DaskGenerator(MagicMock()).visit_TruncateMod(node_mod, ["x", "y"]) == "da.fmod(x, y)"
