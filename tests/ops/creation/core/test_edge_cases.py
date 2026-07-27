import numpy as np

import ml_switcheroo_compiler.ops.creation.frontend_basic as fb
from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.tensor import Device, DType, Tensor, TensorConfig


def test_creation_coverage():
    config.eager_mode = True

    # array
    t = fb.array([1, 2])
    assert isinstance(t, Tensor)
    t2 = fb.array([1.0, 2.0], dtype=DType("float32"))
    assert t2.dtype == DType("float32")

    # asarray
    assert fb.asarray([1, 2]) is not None
    assert fb.asarray(t).dtype == t.dtype
    assert fb.asarray(t, dtype=DType("float32")).dtype == DType("float32")

    # convert_to_tensor
    assert fb.convert_to_tensor([1, 2]) is not None

    # zeros
    z = fb.zeros((2, 2))
    assert z.shape == (2, 2)
    assert (z.numpy() == 0).all()
    z = fb.zeros(2)
    assert z.shape == (2,)

    # ones
    o = fb.ones((2, 2))
    assert o.shape == (2, 2)
    assert (o.numpy() == 1).all()

    # full
    f = fb.full((2, 2), 3.0)
    assert f.shape == (2, 2)
    assert (f.numpy() == 3.0).all()

    class DummyFill:
        data = 4.0

    f2 = fb.full((2, 2), DummyFill())
    assert (f2.numpy() == 4.0).all()

    # zeros_like
    zl = fb.zeros_like(t)
    assert zl.shape == t.shape
    assert (zl.numpy() == 0).all()

    # ones_like
    ol = fb.ones_like(t)
    assert ol.shape == t.shape
    assert (ol.numpy() == 1).all()

    # full_like
    fl = fb.full_like(t, 5.0)
    assert fl.shape == t.shape
    assert (fl.numpy() == 5.0).all()

    # empty
    e = fb.empty((2, 2))
    assert e.shape == (2, 2)

    # empty_like
    el = fb.empty_like(t)
    assert el.shape == t.shape
    assert fb.empty_like(t, dtype=DType("float32")).dtype == DType("float32")

    # convert_to_numpy
    assert isinstance(fb.convert_to_numpy(t), np.ndarray)

    class DummyNoNumpy:
        data = [1, 2]

    assert isinstance(fb.convert_to_numpy(DummyNoNumpy()), np.ndarray)

    # frombuffer
    buf = b"hello"
    fb_buf = fb.frombuffer(buf, dtype=DType("int8"), count=5)
    assert fb_buf.shape == (5,)

    # tracing mode
    original_eager = config.eager_mode
    try:
        config.eager_mode = False
        from ml_switcheroo_compiler.tracing.state import global_tracing_state

        global_tracing_state.is_tracing = True

        class DummyGraph:
            name = "dummy"
            nodes = {}

            def add_node(self, node):
                pass

        global_tracing_state.active_graph = DummyGraph()
        import ml_switcheroo_compiler.ops.registry as registry

        def dummy_emit_creation_node(*args, **kwargs):
            return Tensor(None, TensorConfig(shape=(), dtype=DType("float32"), device=Device("cpu")))

        registry.register_util("_emit_creation_node")(dummy_emit_creation_node)

        # Tracing node creation
        fb.array([1, 2])
        fb.zeros((2, 2))
        fb.ones((2, 2))
        fb.full((2, 2), 3.0)
        fb.zeros_like(t)
        fb.ones_like(t)
        fb.full_like(t, 5.0)
        fb.empty((2, 2))
        fb.frombuffer(buf, dtype=DType("int8"), count=5)

        class DummyNodeShape:
            data = 2

        fb._unpack_shape((DummyNodeShape(),))

    finally:
        config.eager_mode = original_eager
        global_tracing_state.is_tracing = False
