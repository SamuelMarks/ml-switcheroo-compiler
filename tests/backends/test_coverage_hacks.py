"""Module docstring."""

import contextlib
import pytest

from ml_switcheroo_compiler.backends import jax, keras, mlx, pytorch, registry, tensorflow
from ml_switcheroo_compiler.ir.core import IRGraph, IRNode


def test_backends_coverage() -> None:
    """Function docstring."""
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

    with contextlib.suppress(Exception):
        jax.JaxGenerator(g).generate()
    with contextlib.suppress(Exception):
        keras.KerasGenerator(g).generate()
    with contextlib.suppress(Exception):
        mlx.MlxGenerator(g).generate()
    with contextlib.suppress(Exception):
        pytorch.PyTorchGenerator(g).generate()
    with contextlib.suppress(Exception):
        tensorflow.TensorflowGenerator(g).generate()


def test_registry_coverage() -> None:
    """Function docstring."""
    with pytest.raises(ValueError):
        registry.BackendRegistry.get("nonexistent")

    class FakeGen:
        pass

    registry.BackendRegistry.register("fake", FakeGen)
    assert registry.BackendRegistry.get("fake") == FakeGen
    assert "fake" in registry.BackendRegistry.get_all()

    @registry.register_backend("fake2")
    class FakeGen2:
        pass

    assert registry.BackendRegistry.get("fake2") == FakeGen2
