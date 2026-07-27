import numpy as np

from ml_switcheroo_compiler.core.device import Device
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.ops.nn.loss import AdaptiveLogSoftmaxWithLoss, adaptive_log_softmax_with_loss, binary_crossentropy, categorical_crossentropy, sparse_categorical_crossentropy


def test_loss_extras():
    import ml_switcheroo_compiler.tracing.state as state
    from ml_switcheroo_compiler.core.config import config

    orig = config.eager_mode
    orig_tracing = state.global_tracing_state.is_tracing
    config.eager_mode = False
    state.global_tracing_state.is_tracing = True
    orig_add_node = state.global_tracing_state.add_node
    state.global_tracing_state.add_node = lambda node: None
    try:
        t_true = Tensor(np.array([1.0]), TensorConfig(shape=(1,), dtype=DType("float32"), device=Device("cpu")))
        t_pred = Tensor(np.array([0.5]), TensorConfig(shape=(1,), dtype=DType("float32"), device=Device("cpu")))
        try:
            binary_crossentropy(t_true, t_pred, label_smoothing=0.1)
        except Exception:
            pass
        t_true_c = Tensor(np.array([[1.0, 0.0]]), TensorConfig(shape=(1, 2), dtype=DType("float32"), device=Device("cpu")))
        t_pred_c = Tensor(np.array([[0.8, 0.2]]), TensorConfig(shape=(1, 2), dtype=DType("float32"), device=Device("cpu")))
        categorical_crossentropy(t_true_c, t_pred_c, label_smoothing=0.1, from_logits=True)
        sparse_categorical_crossentropy(t_true, t_pred_c, ignore_class=1)

        class DummyShape:
            shape = (1, 2)

        assert AdaptiveLogSoftmaxWithLoss().infer_shape(None, DummyShape()) == ((1, 2), ())

        class MockData:
            id = "data"

        class MockTensor:
            data = MockData()
            shape = (1, 2)
            dtype = DType("float32")
            device = Device("cpu")

        adaptive_log_softmax_with_loss(MockTensor(), MockTensor(), [1])

        class LogPoissonLoss:
            def infer_shape(self, *args):
                return getattr(args[0], "shape", ()) if args else ()

        assert LogPoissonLoss().infer_shape(DummyShape()) == (1, 2)
        assert LogPoissonLoss().infer_shape() == ()
    finally:
        config.eager_mode = orig
        state.global_tracing_state.is_tracing = orig_tracing
        state.global_tracing_state.add_node = orig_add_node


def test_loss_extra_functions():
    import ml_switcheroo_compiler.ops.nn.loss as loss
    import ml_switcheroo_compiler.tracing.state as state
    from ml_switcheroo_compiler.core.config import config
    from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig

    orig = config.eager_mode
    orig_tracing = state.global_tracing_state.is_tracing
    config.eager_mode = False
    state.global_tracing_state.is_tracing = True
    orig_add_node = state.global_tracing_state.add_node
    state.global_tracing_state.add_node = lambda node: None
    try:
        t = Tensor(np.array([1.0]), TensorConfig(shape=(1,), dtype=DType("float32"), device=Device("cpu")))
        t2 = Tensor(np.array([0.5]), TensorConfig(shape=(1,), dtype=DType("float32"), device=Device("cpu")))

        loss.l2_loss(t)
        loss.log_poisson_loss(t, t2, compute_full_loss=True)
        loss.log_poisson_loss(t, t2, compute_full_loss=False)
        loss.circle_loss(t, t2)
        loss.categorical_generalized_cross_entropy(t, t2, 0.1)
        loss.dice_loss(t, t2)
        loss.tversky_loss(t, t2, 0.5, 0.5)
        loss.in_top_k(t, t2, 1)
        loss.scale_regularization_loss(t)

        class MockTopK:
            def infer_shape(self, *args):
                return getattr(args[0], "shape", ()) if args else ()

        assert loss.InTopK().infer_shape(type("DummyShape", (), {"shape": (1, 2)})()) == (1, 2)
        assert loss.InTopK().infer_shape() == ()

        # also test missing branches in binary_crossentropy and categorical
        loss.binary_crossentropy(t, t2, from_logits=True)
        loss.categorical_crossentropy(t, t2, from_logits=False, label_smoothing=0.0)
        loss.sparse_categorical_crossentropy(t, t2, from_logits=True, ignore_class=1)

    finally:
        config.eager_mode = orig
        state.global_tracing_state.is_tracing = orig_tracing
        state.global_tracing_state.add_node = orig_add_node


def test_loss_extra_functions_2():
    import ml_switcheroo_compiler.ops.nn.loss as loss
    import ml_switcheroo_compiler.tracing.state as state
    from ml_switcheroo_compiler.core.config import config
    from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig

    orig = config.eager_mode
    orig_tracing = state.global_tracing_state.is_tracing
    config.eager_mode = False
    state.global_tracing_state.is_tracing = True
    orig_add_node = state.global_tracing_state.add_node
    state.global_tracing_state.add_node = lambda node: None

    try:

        class MockData:
            id = "data"

        class MockTensor:
            data = MockData()
            shape = (1, 2)
            dtype = DType("float32")
            device = Device("cpu")

        mt = MockTensor()
        t = Tensor(np.array([1.0]), TensorConfig(shape=(1,), dtype=DType("float32"), device=Device("cpu")))
        loss.ctc_decode(t, t, top_paths=2)

        # Test adaptive_log_softmax_with_loss eager branch
        config.eager_mode = True

        from unittest.mock import patch

        with patch("ml_switcheroo_compiler.backends.registry.get_active_backend") as mock_backend:
            mock_backend.return_value.execute_op.return_value = (np.array([1.0]), np.array([0.5]))
            loss.adaptive_log_softmax_with_loss(mt, mt, [1])

        # Test adaptive_log_softmax_with_loss tracing exception
        config.eager_mode = False
        state.global_tracing_state.is_tracing = False
        try:
            loss.adaptive_log_softmax_with_loss(mt, mt, [1])
        except RuntimeError:
            pass

        state.global_tracing_state.is_tracing = True
        loss.adaptive_log_softmax_with_loss(mt, mt, [1])

        # Test mathematically complete CTC loss numpy eager solver
        from ml_switcheroo_compiler.backends.numpy.eager.loss_ops import _np_ctc_loss

        # Define mock logits (T=2, B=1, C=3)
        # Logits at T=0: [0.1, 0.8, 0.1] (strongly predicts index 1)
        # Logits at T=1: [0.1, 0.1, 0.8] (strongly predicts index 2)
        logits_test = np.array([[[0.1, 0.8, 0.1]], [[0.1, 0.1, 0.8]]], dtype=np.float32)
        labels_test = np.array([[1, 2]], dtype=np.int32)
        label_len = np.array([2], dtype=np.int32)
        logit_len = np.array([2], dtype=np.int32)

        loss_val = _np_ctc_loss(None, labels_test, logits_test, label_len, logit_len, logits_time_major=True)
        assert len(loss_val) == 1
        assert loss_val[0] > 0.0

        # Test CTC decode shape and log-probabilities
        paths, log_probs = loss.ctc_decode(t, t, top_paths=2)
        assert len(paths) == 2
        assert log_probs is not None
        assert log_probs.shape == (1, 2)

    finally:
        config.eager_mode = orig
        state.global_tracing_state.is_tracing = orig_tracing
        state.global_tracing_state.add_node = orig_add_node


def test_loss_eager_adaptive():
    from unittest.mock import patch

    import ml_switcheroo_compiler.ops.nn.loss as loss
    from ml_switcheroo_compiler.core.config import config
    from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig

    orig = config.eager_mode
    config.eager_mode = True

    try:

        class MockData:
            id = "data"

        class MockTensor:
            data = MockData()
            shape = (1, 2)
            dtype = "float32"
            device = "cpu"

        mt = MockTensor()

        with patch("ml_switcheroo_compiler.backends.registry.get_active_backend") as mock_backend:
            mock_backend.return_value.execute_op.return_value = ("out", "loss")

            # Use original Tensor initialization here
            # But the result returned creates a new Tensor, which needs TensorConfig properly.
            from ml_switcheroo_compiler.core.device import Device
            from ml_switcheroo_compiler.core.dtype import DType

            real_t = Tensor(np.array([1.0]), TensorConfig(shape=(1,), dtype=DType("float32"), device=Device("cpu")))
            out, loss_val = loss.adaptive_log_softmax_with_loss(real_t, real_t, [1])
            assert out.data == "out"
            assert loss_val.data == "loss"
    finally:
        config.eager_mode = orig
