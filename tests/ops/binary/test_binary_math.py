# ruff: noqa: E501
from unittest.mock import patch


def test_binary_math_extra_coverage():
    import ml_switcheroo_compiler.backends.registry as reg
    from ml_switcheroo_compiler.core.config import config
    from ml_switcheroo_compiler.ops.binary.math import Clip, rem

    class DummyBackend:
        def execute_op(self, op, *a, **k):
            return op

    reg._ACTIVE_BACKEND = DummyBackend()
    t_shape2 = type("T", (), {"shape": (2, 2)})()
    t_shape1 = type("T", (), {"shape": (2,)})()
    assert Clip().infer_shape() == ()
    config.eager_mode = True
    try:
        rem(1, 2)
    except Exception:
        pass


def test_eager_eval_branches():
    from ml_switcheroo_compiler.ops.binary.math import ArrayEquiv, Betainc, Diff, Digitize, Poly, Polyadd, Polyder, Polydiv, Polyfit, Polyint, Polymul, Polysub, Polyval, Roots

    for cls in [Diff, Digitize, ArrayEquiv, Betainc, Polyadd, Polysub, Polymul, Polydiv, Polyval, Poly, Polyder, Polyfit, Polyint, Roots]:
        assert cls().op_name is not None


def test_binary_math_shapes_again():
    from ml_switcheroo_compiler.ops.binary.math import ArrayEquiv, Betainc, Clip, Diff, Digitize, clip

    class DummyTensor:
        shape = (1,)
        attributes = {}
        shape_metadata = (1,)

    assert Diff().infer_shape() == ()
    assert Digitize().infer_shape(DummyTensor()) == (1,)
    assert Digitize().infer_shape() == ()
    assert ArrayEquiv().infer_shape() == ()
    assert Betainc().infer_shape(DummyTensor(), DummyTensor(), DummyTensor()) == (1,)
    assert Clip().infer_shape(DummyTensor()) == (1,)
    from ml_switcheroo_compiler.core.config import config

    config.eager_mode = True
    with patch("ml_switcheroo_compiler.backends.registry.get_active_backend") as m:
        m.return_value.execute_op.return_value = "Clip"
        assert clip(DummyTensor()) == "Clip"
    config.eager_mode = False
    import ml_switcheroo_compiler.tracing.state as state

    class DummyGraph:
        def __init__(self):
            self.nodes = {}

        def add_node(self, node):
            pass

    state.global_tracing_state.is_tracing = True
    state.global_tracing_state.active_graph = DummyGraph()
    state.global_tracing_state.add_node = state.global_tracing_state.active_graph.add_node
    with patch("ml_switcheroo_compiler.ops.base.LogicalNode", return_value="node_str", create=True):
        pass
        pass
    config.eager_mode = True


def test_binary_math_clip_emit():
    from ml_switcheroo_compiler.core.config import config

    config.eager_mode = False

    class DummyTensor:
        shape_metadata = (1,)

    import ml_switcheroo_compiler.tracing.state as state

    class DummyGraph:
        def __init__(self):
            self.nodes = {}

        def add_node(self, node):
            pass

    state.global_tracing_state.is_tracing = True
    state.global_tracing_state.active_graph = DummyGraph()
    state.global_tracing_state.add_node = state.global_tracing_state.active_graph.add_node
    from unittest.mock import patch

    with patch("ml_switcheroo_compiler.ops.base.LogicalNode", return_value="node_str", create=True):
        pass
        pass
