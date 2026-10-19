"""Tests for test_export."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import pytest

from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.ops.export import export_to_dot

if TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import EagerCallableProtocol, EagerValue


def test_export() -> None:
    """Test function test_export."""
    import ml_switcheroo_compiler.tracing.state as state

    class DummyNode:
        """Test class DummyNode."""

        def __init__(self, id: EagerValue, op_type: EagerValue, inputs: EagerValue) -> None:
            """Test function __init__.

            Args:
                id: Test parameter `id`.
                op_type: Test parameter `op_type`.
                inputs: Test parameter `inputs`.
            """
            self.id = id
            self.op_type = op_type
            self.inputs = inputs

    class DummyGraph:
        """Test class DummyGraph."""

        def __init__(self) -> None:
            """Test function __init__."""
            self.nodes = {"n1": DummyNode("n1", "Input", []), "n2": DummyNode("n2", "Add", ["n1"])}

    class DummyData:
        """Test class DummyData."""

        id = "n2"

    t = cast("EagerCallableProtocol", Tensor)(cast("EagerCallableProtocol", DummyData)(), cast("EagerCallableProtocol", TensorConfig)((1,), "float32", "cpu"))

    state.global_tracing_state.active_graph = None
    with pytest.raises(RuntimeError):
        cast("EagerCallableProtocol", export_to_dot)("test.dot", t)

    setattr(state.global_tracing_state, "active_graph", DummyGraph())

    import io

    f = io.StringIO()
    cast("EagerCallableProtocol", export_to_dot)(f, t)

    lines = f.getvalue().split("\n")
    assert "digraph G {" in lines
    assert '  "n2" [label="Add\\nn2"];' in lines
    assert '  "n1" -> "n2";' in lines
    assert "}" in lines

    # Test string file path
    import os

    cast("EagerCallableProtocol", export_to_dot)("test_export.dot", t)
    assert os.path.exists("test_export.dot")
    os.remove("test_export.dot")

    # Test invalid node
    class DummyDataInvalid:
        """Test class DummyDataInvalid."""

        id = "n3"

    t_inv = cast("EagerCallableProtocol", Tensor)(cast("EagerCallableProtocol", DummyDataInvalid)(), cast("EagerCallableProtocol", TensorConfig)((1,), "float32", "cpu"))

    f2 = io.StringIO()
    cast("EagerCallableProtocol", export_to_dot)(f2, t_inv)
    assert "n3" not in f2.getvalue()
