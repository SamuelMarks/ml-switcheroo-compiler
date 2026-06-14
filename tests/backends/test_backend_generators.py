"""Test backend generator basic functionality."""

import pytest

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
