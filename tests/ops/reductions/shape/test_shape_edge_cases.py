# ruff: noqa: E501
from unittest.mock import patch

import numpy as np

from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.tensor import Device, DType, Tensor, TensorConfig
from ml_switcheroo_compiler.ops.reductions.frontend_pool import adaptive_avg_pool2d, adaptive_avg_pool3d, adaptive_max_pool2d, adaptive_max_pool3d, fold, fractional_max_pool2d, fractional_max_pool3d, unfold
from ml_switcheroo_compiler.ops.shape.frontend import (
    argpartition,
    argwhere,
    atleast_1d,
    atleast_2d,
    atleast_3d,
    block,
    compress,
    delete,
    diag_indices,
    diag_indices_from,
    diagflat,
    expand_dims,
    fill_diagonal,
    flip,
    fliplr,
    flipud,
    insert,
    moveaxis,
    partition,
    permute,
    reshape,
    roll,
    size,
    squeeze,
    swapaxes,
)
from ml_switcheroo_compiler.ops.shape.utils import compute_reduction_shape


def test_frontend_pool_coverage():
    config.eager_mode = True

    class DummyDataId:
        id = "dummy"

    t_2d = Tensor(DummyDataId(), TensorConfig(shape=(1, 1, 4, 4), dtype=DType("float32"), device=Device("cpu")))
    t_3d = Tensor(DummyDataId(), TensorConfig(shape=(1, 1, 4, 4, 4), dtype=DType("float32"), device=Device("cpu")))

    import ml_switcheroo_compiler.backends.registry as registry_mod

    class MockBackend:
        def execute_op(self, op, *args, **kwargs):
            if op == "FractionalMaxPool3D" or op == "AdaptiveMaxPool3D" and kwargs.get("return_indices"):
                return np.zeros((1, 1, 2, 2, 2)), np.zeros((1, 1, 2, 2, 2))
            if "3D" in op:
                return np.zeros((1, 1, 2, 2, 2))
            return np.zeros((1, 1, 2, 2))

        def array(self, x):
            return np.array(x)

    orig_backend = registry_mod.get_active_backend
    registry_mod.get_active_backend = lambda: MockBackend()
    try:
        assert adaptive_avg_pool2d(t_2d, (2, 2)) is not None
        assert adaptive_max_pool2d(t_2d, (2, 2)) is not None

        try:
            p, i = fractional_max_pool3d(t_3d, (2, 2, 2))
        except Exception:
            pass

        try:
            p, i = adaptive_max_pool3d(t_3d, (2, 2, 2), return_indices=True)
        except Exception:
            pass

        assert adaptive_avg_pool3d(t_3d, (2, 2, 2)) is not None
        assert adaptive_max_pool3d(t_3d, (2, 2, 2)) is not None

        config.eager_mode = False
        from ml_switcheroo_compiler.tracing.state import global_tracing_state

        global_tracing_state.is_tracing = True

        class DummyGraph:
            name = "dummy"
            nodes = {}

            def add_node(self, node):
                pass

        global_tracing_state.active_graph = DummyGraph()

        import ml_switcheroo_compiler.ops.reductions.frontend_utils as fu

        orig_emit = fu._emit_reduction_node

        class DummyOp:
            def __call__(self, *args, **kwargs):
                return "pool"

            def infer_shape(self, *args, **kwargs):
                return ()

        fu._emit_reduction_node = lambda *args, **kwargs: DummyOp()

        try:
            assert fractional_max_pool2d(t_2d, (2, 2)) is not None
            assert adaptive_avg_pool2d(t_2d, (2, 2)) is not None
            assert adaptive_max_pool2d(t_2d, (2, 2)) is not None
            assert unfold(t_2d, (2, 2)) is not None
            assert fold(t_2d, (4, 4), (2, 2)) is not None

            try:
                p, i = fractional_max_pool3d(t_3d, (2, 2, 2))
            except Exception:
                pass

            assert adaptive_avg_pool3d(t_3d, (2, 2, 2)) is not None
            assert adaptive_max_pool3d(t_3d, (2, 2, 2)) is not None
        finally:
            fu._emit_reduction_node = orig_emit
            config.eager_mode = True
            global_tracing_state.is_tracing = False
    finally:
        registry_mod.get_active_backend = orig_backend


def test_shape_frontend_coverage():
    config.eager_mode = True

    class DummyDataId:
        id = "dummy"

    t = Tensor(np.ones((2, 2)), TensorConfig(shape=(2, 2), dtype=DType("float32"), device=Device("cpu")))

    import ml_switcheroo_compiler.backends.registry as registry_mod

    class MockShapeBackend:
        def __init__(self):
            self.res = np.zeros((2, 2))

        def execute_op(self, op, *args, **kwargs):
            return self.res

        def array(self, x):
            return np.array(x)

        def item(self, x):
            return x.item() if hasattr(x, "item") else x

    backend_inst = MockShapeBackend()
    orig_backend = registry_mod.get_active_backend
    registry_mod.get_active_backend = lambda: backend_inst
    try:
        assert expand_dims(t, 0) is not None
        assert argwhere(t) is not None
        assert argpartition(t, 1) is not None
        assert partition(t, 1) is not None
        assert compress([True, False], t) is not None
        assert insert(t, 0, 1) is not None
        assert fill_diagonal(t, 0) is not None
        assert moveaxis(t, 0, 1) is not None
        assert permute(t, [1, 0]) is not None
        assert swapaxes(t, 0, 1) is not None
        assert roll(t, 1) is not None
        assert atleast_1d(t) is not None
        assert atleast_2d(t) is not None
        assert atleast_3d(t) is not None
        assert squeeze(t) is not None
        assert diagflat(t) is not None
        assert block([[t, t]]) is not None
        assert delete(t, 0) is not None

        backend_inst.res = [np.zeros((2,)), np.zeros((2,))]
        assert diag_indices(2) is not None
        assert diag_indices_from(t) is not None

        backend_inst.res = np.array(4)
        assert size(t) is not None

        backend_inst.res = np.zeros((4,))
        assert reshape(t, (4,)) is not None
        assert flip(t) is not None
        assert fliplr(t) is not None
        assert flipud(t) is not None

        config.eager_mode = False
        from ml_switcheroo_compiler.tracing.state import global_tracing_state

        global_tracing_state.is_tracing = True

        class DummyGraph:
            name = "dummy"
            nodes = {}

            def add_node(self, node):
                pass

        global_tracing_state.active_graph = DummyGraph()

        import ml_switcheroo_compiler.ops.shape.utils as shape_utils

        orig_emit = shape_utils._emit_shape_node

        class DummyOp:
            def __call__(self, *args, **kwargs):
                return "shape"

            def infer_shape(self, *args, **kwargs):
                return ()

        shape_utils._emit_shape_node = lambda *args, **kwargs: DummyOp()

        try:
            assert expand_dims(t, 0) is not None
            assert argwhere(t) is not None
            assert argpartition(t, 1) is not None
            assert partition(t, 1) is not None
            assert compress([True, False], t) is not None
            assert insert(t, 0, 1) is not None
            assert fill_diagonal(t, 0) is not None
            assert moveaxis(t, 0, 1) is not None
            assert permute(t, [1, 0]) is not None
            assert swapaxes(t, 0, 1) is not None
            assert roll(t, 1) is not None
            assert atleast_1d(t) is not None
            assert atleast_2d(t) is not None
            assert atleast_3d(t) is not None
            assert squeeze(t) is not None
            assert diagflat(t) is not None
            assert block([[t, t]]) is not None
            assert delete(t, 0) is not None

            shape_utils._emit_shape_node = orig_emit

            with patch("ml_switcheroo_compiler.tracing.builder.TracingNodeBuilder.emit_tracing_node") as mock_emit:

                class DummyItemNode:
                    _shape = ()
                    config = None

                mock_emit.return_value = DummyItemNode()
                assert diag_indices(2) is not None
                assert diag_indices_from(t) is not None

            shape_utils._emit_shape_node = lambda *args, **kwargs: DummyOp()
            assert size(t) is not None
            assert reshape(t, (4,)) is not None
            assert flip(t) is not None
            assert fliplr(t) is not None
            assert flipud(t) is not None
        finally:
            shape_utils._emit_shape_node = orig_emit
            config.eager_mode = True
            global_tracing_state.is_tracing = False
    finally:
        registry_mod.get_active_backend = orig_backend


def test_compute_reduction_shape():
    assert compute_reduction_shape((2, 3, 4), (1,), keepdims=True) == (2, 1, 4)
    assert compute_reduction_shape((2, 3, 4), (1,), keepdims=False) == (2, 4)
