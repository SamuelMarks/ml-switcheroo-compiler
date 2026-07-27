import sys

import pytest

from ml_switcheroo_compiler.core.device import Device
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig


def make_tensor(data):
    cfg = TensorConfig(device=Device("cpu"), dtype=DType("float32"), shape=())
    return Tensor(data, cfg)


def test_numpy_conversion_fallback():
    t = make_tensor(1.0)
    original_numpy = sys.modules.get("numpy")
    sys.modules["numpy"] = None
    try:
        t.numpy()
    except Exception:
        pass
    finally:
        sys.modules["numpy"] = original_numpy


def test_exception_in_asarray():
    class BadData:
        def tolist(self):
            return [1.0]

        def __array__(self, *args, **kwargs):
            raise RuntimeError("boom")

    t = make_tensor(BadData())
    res = t.__array__()
    assert res.shape == (1,)


def test_item_eval_fallback(monkeypatch):
    class FakeBackend:
        def item(self, x):
            return int(x)

    class FakeTensor(Tensor):
        def eval(self):
            return 5

    t = FakeTensor(1.0, TensorConfig(device=Device("cpu"), dtype=DType("float32"), shape=()))
    import ml_switcheroo_compiler.backends.registry as reg

    monkeypatch.setattr(reg, "get_active_backend", lambda: FakeBackend())
    assert t.item() == 5


def test_getitem_indexerror():
    t = make_tensor([1, 2])
    with pytest.raises(IndexError):
        t[1, 2]

    from ml_switcheroo_compiler.core import config

    orig = config.eager_mode
    config.eager_mode = False

    class FakeProxy:
        def __index__(self):
            raise IndexError("only integers")

    import ml_switcheroo_compiler.tracing.state as state

    orig_tracing = state.global_tracing_state.is_tracing
    state.global_tracing_state.is_tracing = True
    orig_add_node = state.global_tracing_state.add_node
    state.global_tracing_state.add_node = lambda node: None
    try:
        res = t[FakeProxy()]
        assert res is not None
    finally:
        config.eager_mode = orig
        state.global_tracing_state.is_tracing = orig_tracing
        state.global_tracing_state.add_node = orig_add_node
