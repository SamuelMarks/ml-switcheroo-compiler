import pytest

from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.ops.export import export_to_dot


def test_export():
    import ml_switcheroo_compiler.tracing.state as state

    class DummyNode:
        def __init__(self, id, op_type, inputs):
            self.id = id
            self.op_type = op_type
            self.inputs = inputs

    class DummyGraph:
        def __init__(self):
            self.nodes = {"n1": DummyNode("n1", "Input", []), "n2": DummyNode("n2", "Add", ["n1"])}

    class DummyData:
        id = "n2"

    t = Tensor(DummyData(), TensorConfig((1,), "float32", "cpu"))

    state.global_tracing_state.active_graph = None
    with pytest.raises(RuntimeError):
        export_to_dot("test.dot", t)

    state.global_tracing_state.active_graph = DummyGraph()

    import io

    f = io.StringIO()
    export_to_dot(f, t)

    lines = f.getvalue().split("\n")
    assert "digraph G {" in lines
    assert '  "n2" [label="Add\\nn2"];' in lines
    assert '  "n1" -> "n2";' in lines
    assert "}" in lines

    # Test string file path
    import os

    export_to_dot("test_export.dot", t)
    assert os.path.exists("test_export.dot")
    os.remove("test_export.dot")

    # Test invalid node
    class DummyDataInvalid:
        id = "n3"

    t_inv = Tensor(DummyDataInvalid(), TensorConfig((1,), "float32", "cpu"))

    f2 = io.StringIO()
    export_to_dot(f2, t_inv)
    assert "n3" not in f2.getvalue()
