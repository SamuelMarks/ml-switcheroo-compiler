import pytest


def test_signal_missing():
    import ml_switcheroo_compiler.ops.signal as sig

    class FakeShape:
        shape = None

    with pytest.raises(ValueError):
        sig._validate_conv2d_args(FakeShape(), FakeShape())

    class FakeShape2:
        shape = (1, 1)

    with pytest.raises(ValueError):
        sig._validate_conv2d_args(FakeShape2(), FakeShape())

    with pytest.raises(ValueError):
        sig._validate_conv2d_args(FakeShape(), FakeShape2())


def test_signal_classes_infer_shape():
    import ml_switcheroo_compiler.ops.signal as sig

    class DummyData:
        shape = (1, 1)

    assert sig.Rfft().infer_shape(DummyData()) == (1, 1)
    assert sig.Fft2().infer_shape(DummyData()) == (1, 1)
    assert sig.Fftfreq().infer_shape(5) == (5,)
    assert sig.Fftfreq().infer_shape(DummyData()) == ()
    assert sig.Irfft().infer_shape(DummyData()) == (1, 1)
    assert sig.Ihfft().infer_shape(DummyData()) == (1, 1)


def test_signal_tracing_missing():
    from unittest.mock import patch

    import numpy as np

    import ml_switcheroo_compiler.ops.signal as sig

    class DummyData:
        shape = (2, 2)
        dtype = "float32"
        data = np.array([[1.0, 2.0], [3.0, 4.0]])

        def __init__(self, data=None):
            self.data = data if data is not None else np.array([[1.0, 2.0], [3.0, 4.0]])

    from ml_switcheroo_compiler.core.config import config

    config.eager_mode = False

    with patch.object(sig, "_emit_signal_node", return_value="emitted"):
        assert sig.fftn(DummyData()) == "emitted"
        assert sig.ifftn(DummyData()) == "emitted"
        assert sig.irfftn(DummyData()) == "emitted"
        assert sig.ifft2(DummyData()) == "emitted"
        assert sig.irfft2(DummyData()) == "emitted"

    config.eager_mode = True


def test_signal_eager_missing():
    import numpy as np

    import ml_switcheroo_compiler.ops.signal as sig

    class DummyData:
        shape = (2, 2)
        dtype = "float32"
        device = "cpu"
        data = np.array([[1.0, 2.0], [3.0, 4.0]])

        def __init__(self, data=None):
            self.data = data if data is not None else np.array([[1.0, 2.0], [3.0, 4.0]])

    from ml_switcheroo_compiler.core.config import config

    config.eager_mode = True

    import ml_switcheroo_compiler.backends.registry as reg

    class DummyBackend:
        def execute_op(self, name, *a, **k):
            class R:
                shape = (1,)
                dtype = "float32"
                device = "cpu"

                def tolist(self):
                    return [1]

            return R()

    old = reg.get_active_backend
    reg.get_active_backend = lambda: DummyBackend()

    sig.fftn(DummyData())
    sig.ifftn(DummyData())
    sig.irfftn(DummyData())
    sig.ifft2(DummyData())
    sig.irfft2(DummyData())

    reg.get_active_backend = old


def test_signal_missing_lines_in_eager():
    from unittest.mock import patch

    import ml_switcheroo_compiler.ops.signal as s
    from ml_switcheroo_compiler.ops.signal import _emit_signal_node

    with patch.object(s, "_emit_linalg_node", return_value="ok"):
        assert _emit_signal_node("test", [], {}, (1,), "float32") == "ok"
