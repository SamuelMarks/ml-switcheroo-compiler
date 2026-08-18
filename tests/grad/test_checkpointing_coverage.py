def test_checkpointing():
    import numpy as np

    from ml_switcheroo_compiler.core.config import config
    from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
    from ml_switcheroo_compiler.grad.checkpointing import checkpoint
    from ml_switcheroo_compiler.tracing.state import global_tracing_state

    # Eager mode
    def dummy(x):
        return x + 1

    cp = checkpoint(dummy)
    assert cp(1) == 2

    # Traced mode
    config.eager_mode = False
    global_tracing_state.start_tracing()

    class FakeData:
        def __init__(self, id):
            self.id = id

    t1 = Tensor(np.array([1, 2]), TensorConfig(shape=(2,), dtype="float32", device="cpu"))
    t1._data = FakeData("n1")

    t2 = Tensor(np.array([1, 2]), TensorConfig(shape=(2,), dtype="float32", device="cpu"))
    t2._data = FakeData("n2")

    res = cp(t1)

    # with dtype in attr
    def dummy_dtype(x):
        return x

    # Test fallback
    from unittest.mock import patch

    class FakeBlock:
        outputs = ["out"]
        nodes = {"out": type("Node", (), {"inputs": ["real"], "id": "out"})(), "real": type("Node", (), {"shape_metadata": (2,), "attributes": {"dtype": "int32"}, "id": "real"})()}

    with patch("ml_switcheroo_compiler.ops.control_flow_utils._trace_function", return_value=FakeBlock()):
        res2 = cp(t1)
        assert res2.shape == (2,)

    class FakeBlock2:
        outputs = ["out"]
        nodes = {"out": type("Node", (), {"inputs": ["real"], "id": "out"})(), "real": type("Node", (), {"shape_metadata": (2,), "attributes": {}, "id": "real"})()}

    with patch("ml_switcheroo_compiler.ops.control_flow_utils._trace_function", return_value=FakeBlock2()):
        res3 = cp()
        assert res3.shape == (2,)

    config.eager_mode = True
    global_tracing_state.stop_tracing()


def test_remat_and_recompute_grad():
    from ml_switcheroo_compiler.grad.checkpointing import recompute_grad, remat

    def dummy(x):
        return x

    # 106, 118
    assert remat(dummy)(1) == 1
    assert recompute_grad(dummy)(2) == 2
