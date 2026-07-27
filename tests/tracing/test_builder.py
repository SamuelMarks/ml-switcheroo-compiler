# ruff: noqa: E501


def test_builder_extract_proxy():
    import ml_switcheroo_compiler.tracing.builder as bmod
    from ml_switcheroo_compiler.tracing.builder import TracingNodeBuilder

    class FakeData:
        id = "my_id"

    class FakeTensor:
        data = FakeData()
        shape = (2, 2)

    (id1, shape1) = TracingNodeBuilder.extract_from_tensor(FakeTensor())
    assert id1 == "my_id"
    assert shape1 == (2, 2)

    class FakeData2:
        def tolist(self):
            return [[1, 2], [3, 4]]

    class FakeTensor2:
        data = FakeData2()
        shape = (2, 2)

    class FakeTracingState:
        constant_cache = {}
        is_tracing = True
        active_graph = True

        def add_node(self, node):
            pass

    old_state = bmod.global_tracing_state
    bmod.global_tracing_state = FakeTracingState()
    (id2, shape2) = TracingNodeBuilder.extract_from_tensor(FakeTensor2())
    assert id2 in FakeTracingState.constant_cache.values()
    (id3, shape3) = TracingNodeBuilder.extract_from_tensor(FakeTensor2())
    assert id2 == id3
    bmod.global_tracing_state = old_state


def test_builder_extract_constant():
    import ml_switcheroo_compiler.tracing.builder as bmod
    from ml_switcheroo_compiler.tracing.builder import TracingNodeBuilder

    class FakeTracingState:
        constant_cache = {}
        is_tracing = True
        active_graph = True

        def add_node(self, node):
            pass

    old_state = bmod.global_tracing_state
    bmod.global_tracing_state = FakeTracingState()

    class ProxyT:
        def __init__(self, id, shape):
            self.id = id
            self.shape = shape

    ProxyT.__name__ = "ProxyTensor"
    proxy_inst = ProxyT("proxy_id", (2,))
    proxy_inst.__class__.__name__ = "ProxyTensor"

    class RealTensor:
        def __init__(self, data):
            self.data = data

        @property
        def shape(self):
            return getattr(self.data, "shape", ())

    val = [ProxyT("id_1", (1,)), RealTensor(123)]
    (id1, shape1) = TracingNodeBuilder.extract_from_constant(val)
    assert id1[0] == "id_1"
    bmod.global_tracing_state = old_state


def test_builder_extract_proxy_inputs_proxy():
    from ml_switcheroo_compiler.tracing.builder import TracingNodeBuilder

    class FakeTensorProxy:
        id = "proxy_id"
        shape = (1, 2)

    (ids, shapes, first) = TracingNodeBuilder.extract_proxy_inputs((FakeTensorProxy(),))
    assert ids == ["proxy_id"]
    assert shapes == [(1, 2)]
    assert first is None


def test_builder_tensor_cache_missing():
    import ml_switcheroo_compiler.tracing.builder as bmod
    from ml_switcheroo_compiler.core.tensor import Tensor
    from ml_switcheroo_compiler.tracing.builder import TracingNodeBuilder

    class FakeData:
        def tolist(self):
            return [1]

        shape = (1,)
        dtype = "float32"
        device = "cpu"

    class FakeTracingState:
        is_tracing = True
        active_graph = True

        def add_node(self, node):
            pass

    old_state = bmod.global_tracing_state
    bmod.global_tracing_state = FakeTracingState()

    class FakeConfig:
        shape = (1,)
        dtype = None
        device = None
        requires_grad = False

    fake_t = Tensor(FakeData(), FakeConfig())
    fake_t._data = FakeData()
    (id1, shape1) = TracingNodeBuilder.extract_from_tensor(fake_t)
    assert shape1 == (1,)
    (ids, shapes, first) = TracingNodeBuilder.extract_proxy_inputs((fake_t, fake_t))
    assert len(ids) == 2
    assert first is not None
    bmod.global_tracing_state = old_state


def test_builder_extract_proxy_inputs_tensor_missing():
    import ml_switcheroo_compiler.tracing.builder as bmod
    from ml_switcheroo_compiler.tracing.builder import TracingNodeBuilder

    class FakeTracingState:
        is_tracing = True
        active_graph = True

        def add_node(self, node):
            pass

    old_state = bmod.global_tracing_state
    bmod.global_tracing_state = FakeTracingState()

    class DummyNotTensor:
        def __init__(self):
            pass

        def tolist(self):
            return [42]

    (ids, shapes, first) = TracingNodeBuilder.extract_proxy_inputs((DummyNotTensor(),))
    assert len(ids) == 1
    assert shapes[0] == ()
    bmod.global_tracing_state = old_state
