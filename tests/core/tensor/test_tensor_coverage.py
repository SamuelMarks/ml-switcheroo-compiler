import numpy as np

from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.tracing.state import global_tracing_state


def test_tensor_eval_tracing():
    class DummyGraph:
        def __init__(self):
            self.outputs = []

    class DummyData:
        def __init__(self):
            self.id = "dummy_id"
            self.shape = (2, 2)
            self.dtype = float

    tc = TensorConfig(shape=(2, 2), dtype=float, device=None)
    t = Tensor(DummyData(), tc)
    t2 = Tensor(DummyData(), tc)
    t3 = Tensor(DummyData(), tc)
    orig_eager = config.eager_mode
    orig_tracing = global_tracing_state.is_tracing
    orig_graph = global_tracing_state.active_graph

    try:
        config.eager_mode = False
        global_tracing_state.is_tracing = True
        global_tracing_state.active_graph = DummyGraph()

        t.eval()
        t.eval()
        global_tracing_state.active_graph.outputs.append("other")
        t2.eval()
        global_tracing_state.active_graph = None
        t3.eval()
    finally:
        config.eager_mode = orig_eager
        global_tracing_state.is_tracing = orig_tracing
        global_tracing_state.active_graph = orig_graph


def test_tensor_numpy_import_error():
    import sys

    from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig

    class DummyData:
        def __init__(self):
            self.shape = (2, 2)
            self.dtype = float
            self._data = np.ones((2, 2))

    tc = TensorConfig(shape=(2, 2), dtype=float, device=None)
    t = Tensor(DummyData(), tc)

    # Force ImportError
    original_np = sys.modules.get("numpy")
    sys.modules["numpy"] = None
    try:
        t.numpy()
    except ImportError:
        pass
    finally:
        sys.modules["numpy"] = original_np

    original_importlib = sys.modules.get("importlib.util")

    import types

    dummy_importlib = types.ModuleType("importlib")
    dummy_importlib.util = types.ModuleType("importlib.util")

    def find_spec_none(name):
        return None

    dummy_importlib.util.find_spec = find_spec_none

    sys.modules["importlib.util"] = dummy_importlib.util
    try:
        t.numpy()
    finally:
        sys.modules["importlib.util"] = original_importlib


def test_tensor_getitem_index_error_eager():
    from ml_switcheroo_compiler.core.config import config
    from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig

    class DummyData:
        def __init__(self):
            self.shape = (2, 2)
            self.dtype = float
            self._data = np.ones((2, 2))

    tc = TensorConfig(shape=(2, 2), dtype=float, device=None)
    t = Tensor(DummyData(), tc)

    orig_eager = config.eager_mode
    try:
        config.eager_mode = True
        try:
            t[100]
        except IndexError:
            pass
    finally:
        config.eager_mode = orig_eager
