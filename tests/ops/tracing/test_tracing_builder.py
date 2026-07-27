# ruff: noqa: E501
from ml_switcheroo_compiler.tracing.builder import TracingNodeBuilder


def test_extract_from_constant(mocker):

    class Proxy:
        def __init__(self, id, shape):
            self.id = id
            self.shape = shape

    Proxy.__name__ = "ProxyTensor"
    p = type("ProxyTensor", (), {"id": "1", "shape": (2,)})()
    assert TracingNodeBuilder.extract_from_constant([p]) == (["1"], [(2,)])
    mocker.patch("ml_switcheroo_compiler.tracing.builder.global_tracing_state.is_tracing", True)
    mocker.patch("ml_switcheroo_compiler.tracing.builder.global_tracing_state.active_graph", True)
    mocker.patch("ml_switcheroo_compiler.tracing.builder.global_tracing_state.add_node")
    assert TracingNodeBuilder.extract_from_constant([1, p])[1] == [(), (2,)]


def test_extract_from_tensor(mocker):
    mocker.patch("ml_switcheroo_compiler.tracing.builder.global_tracing_state.is_tracing", True)
    mocker.patch("ml_switcheroo_compiler.tracing.builder.global_tracing_state.active_graph", True)
    mocker.patch("ml_switcheroo_compiler.tracing.builder.global_tracing_state.add_node")

    class MockTensor1:
        def __init__(self):
            self.data = type("M", (), {"id": "1"})()
            self.shape = (2,)

    assert TracingNodeBuilder.extract_from_tensor(MockTensor1()) == ("1", (2,))

    class MockTensor2:
        def __init__(self):
            self.data = 42
            self.shape = ()

    res = TracingNodeBuilder.extract_from_tensor(MockTensor2())
    assert res[1] == ()


def test_extract_proxy_inputs(mocker):
    from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig

    mocker.patch("ml_switcheroo_compiler.tracing.builder.global_tracing_state.is_tracing", True)
    mocker.patch("ml_switcheroo_compiler.tracing.builder.global_tracing_state.active_graph", True)
    mocker.patch("ml_switcheroo_compiler.tracing.builder.global_tracing_state.add_node")

    class M:
        id = "id"
        shape = ()

    t = Tensor(M(), TensorConfig((), "float32", "cpu"))
    (ids, shapes, ft) = TracingNodeBuilder.extract_proxy_inputs((t, M(), 1))
    assert ids[0] == "id"
    assert ids[1] == "id"
    assert ft is t


def test_create_tracing_logical_node(mocker):
    mocker.patch("ml_switcheroo_compiler.tracing.builder.global_tracing_state.is_tracing", True)
    mocker.patch("ml_switcheroo_compiler.tracing.builder.global_tracing_state.active_graph", True)
    mock_add = mocker.patch("ml_switcheroo_compiler.tracing.builder.global_tracing_state.add_node")
    out_id = TracingNodeBuilder.create_tracing_logical_node("Op", ["1"], {"a": 1}, (2, 3))
    assert type(out_id) is str
    mock_add.assert_called_once()


def test_emit_tracing_node(mocker):
    from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig

    mocker.patch("ml_switcheroo_compiler.tracing.builder.global_tracing_state.is_tracing", True)
    mocker.patch("ml_switcheroo_compiler.tracing.builder.global_tracing_state.active_graph", True)
    mock_add = mocker.patch("ml_switcheroo_compiler.tracing.builder.global_tracing_state.add_node")
    mocker.patch("ml_switcheroo_compiler.tracing.builder.infer_shape", return_value=(2, 3))

    class M:
        id = "1"
        shape = ()
        dtype = "float32"
        device = "cpu"

    t = Tensor(M(), TensorConfig((), "float32", "cpu"))
    res = TracingNodeBuilder.emit_tracing_node("Op", t)
    assert res.config.shape == (2, 3)
