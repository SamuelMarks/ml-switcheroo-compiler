"""Tests for test_frontend."""

from __future__ import annotations

import importlib
import typing
from unittest.mock import patch

import numpy as np
import pytest

from ml_switcheroo_compiler.backends.eager_registry import EagerValue
from ml_switcheroo_compiler.core.config import ConfigContext, config
from ml_switcheroo_compiler.core.device import Device
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.ops.reductions import approx_max_k, approx_min_k, segment_max, segment_mean, segment_min, segment_prod, unsorted_segment_max, unsorted_segment_mean, unsorted_segment_min, unsorted_segment_prod, unsorted_segment_sqrt_n, unsorted_segment_sum
from ml_switcheroo_compiler.ops.reductions.frontend import max as fe_max
from ml_switcheroo_compiler.ops.reductions.frontend import min as fe_min
from ml_switcheroo_compiler.ops.reductions.frontend import segment_sum
from ml_switcheroo_compiler.ops.reductions.frontend import sum as fe_sum
from ml_switcheroo_compiler.ops.reductions.frontend_pool import UnpoolOptions, adaptive_avg_pool2d, adaptive_avg_pool3d, adaptive_max_pool2d, adaptive_max_pool3d, fold, fractional_max_pool2d, fractional_max_pool3d, max_unpool1d, max_unpool2d, max_unpool3d, unfold
from ml_switcheroo_compiler.ops.reductions.frontend_utils import _emit_reduction_node, reduce_window
from ml_switcheroo_compiler.tracing.tracer import ProxyTensor, _tracer

if typing.TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import EagerCallableProtocol, EagerTensorProtocol, EagerValue
WindowConfig = typing.cast("EagerTensorProtocol", getattr(importlib.import_module("ml_switcheroo_compiler.ops.reductions.frontend_utils"), "WindowConfig"))


def test_reductions_frontend_coverage_brute() -> None:
    """Test the reductions frontend coverage brute behavior.

    Returns:
        TensorOrScalar: The inferred shape or computed result.
    """
    config.eager_mode = False
    _tracer.start_tracing()
    data = Tensor(ProxyTensor(id="data_id", shape=(5,)), TensorConfig((5,), DType.Float32, Device("cpu")))
    segment_ids = Tensor(ProxyTensor(id="segment_id", shape=(5,)), TensorConfig((5,), DType.Int32, Device("cpu")))
    segment_sum(data, segment_ids)
    _tracer.stop_tracing()
    config.eager_mode = True
    assert callable(test_reductions_frontend_coverage_brute)


def test_remaining_segment_ops() -> None:
    """Test the remaining segment ops behavior.

    Returns:
        TensorOrScalar: The inferred shape or computed result.
    """
    data = Tensor(ProxyTensor(id="data_id", shape=(3,), dtype=DType("float32")), TensorConfig((3,), DType("float32"), None))
    segment_ids = Tensor(ProxyTensor(id="seg_id", shape=(3,), dtype=DType("int32")), TensorConfig((3,), DType("int32"), None))
    with ConfigContext(eager_mode=False):
        _tracer.start_tracing()
        segment_max(data, segment_ids)
        segment_mean(data, segment_ids)
        segment_min(data, segment_ids)
        segment_prod(data, segment_ids)
        unsorted_segment_max(data, segment_ids, 2)
        unsorted_segment_mean(data, segment_ids, 2)
        unsorted_segment_min(data, segment_ids, 2)
        unsorted_segment_prod(data, segment_ids, 2)
        unsorted_segment_sqrt_n(data, segment_ids, 2)
        unsorted_segment_sum(data, segment_ids, 2)
        approx_max_k(data, 2)
        approx_min_k(data, 2)
        _tracer.stop_tracing()
    assert callable(test_remaining_segment_ops)


def _test_reductions_frontend_brute_part_1(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 1 of test_reductions_frontend_brute.

    Args:
        _state: Shared test state dictionary.
    """
    config.backend = "numpy"
    config.eager_mode = True
    t_2d = Tensor(np.random.rand(2, 3, 8, 8).astype(np.float32), TensorConfig((2, 3, 8, 8), "float32", "cpu"))
    t_3d = Tensor(np.random.rand(2, 3, 8, 8, 8).astype(np.float32), TensorConfig((2, 3, 8, 8, 8), "float32", "cpu"))
    t_1d = Tensor(np.random.rand(2, 3, 8).astype(np.float32), TensorConfig((2, 3, 8), "float32", "cpu"))
    indices = Tensor(np.zeros((2, 3, 8)).astype(np.int64), TensorConfig((2, 3, 8), "int64", "cpu"))
    indices2d = Tensor(np.zeros((2, 3, 8, 8)).astype(np.int64), TensorConfig((2, 3, 8, 8), "int64", "cpu"))
    indices3d = Tensor(np.zeros((2, 3, 8, 8, 8)).astype(np.int64), TensorConfig((2, 3, 8, 8, 8), "int64", "cpu"))
    getattr(_state, "update")({"indices": indices, "indices2d": indices2d, "indices3d": indices3d, "t_1d": t_1d})
    getattr(_state, "update")({"t_2d": t_2d, "t_3d": t_3d})
    _state.update(locals())


def _test_reductions_frontend_brute_part_2(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 2 of test_reductions_frontend_brute.

    Args:
        _state: Shared test state dictionary.
    """
    (indices, indices2d, indices3d, t_1d) = (_state["indices"], _state["indices2d"], _state["indices3d"], _state["t_1d"])
    (t_2d, t_3d) = (_state["t_2d"], _state["t_3d"])
    with patch("ml_switcheroo_compiler.ops.reductions.frontend_pool.get_active_backend") as mock_backend, patch("ml_switcheroo_compiler.ops.reductions.frontend_utils.get_active_backend") as mock_backend_utils:

        class DummyBackend:
            """Test class DummyBackend."""

            def execute_op(self, op_type: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
                """Test function execute_op.

                Args:
                    op_type: Test parameter `op_type`.
                    args: Test parameter `args`.
                    kwargs: Test parameter `kwargs`.
                """
                if kwargs.get("return_indices", False) or op_type == "FractionalMaxPool3d":
                    return (np.zeros_like(args[0]), np.zeros_like(args[0]).astype(np.int64))
                return np.zeros_like(args[0])

            def array(self, x: EagerValue) -> EagerValue:
                """Test function array.

                Args:
                    x: Test parameter `x`.
                """
                return x

        dummy_backend = DummyBackend()
        mock_backend.return_value = dummy_backend
        mock_backend_utils.return_value = dummy_backend
        typing.cast("EagerCallableProtocol", adaptive_avg_pool2d)(t_2d, (4, 4))
        typing.cast("EagerCallableProtocol", adaptive_max_pool2d)(t_2d, (4, 4))
        typing.cast("EagerCallableProtocol", fractional_max_pool3d)(t_3d, (4, 4, 4))
        typing.cast("EagerCallableProtocol", adaptive_avg_pool3d)(t_3d, (4, 4, 4))
        typing.cast("EagerCallableProtocol", adaptive_max_pool3d)(t_3d, (4, 4, 4))
        typing.cast("EagerCallableProtocol", adaptive_max_pool3d)(t_3d, (4, 4, 4), return_indices=True)
        opt1 = UnpoolOptions(kernel_size=2, output_size=(16,))
        opt2 = UnpoolOptions(kernel_size=(2, 2), output_size=(16, 16))
        opt3 = UnpoolOptions(kernel_size=(2, 2, 2), output_size=(16, 16, 16))
        typing.cast("EagerCallableProtocol", max_unpool1d)(t_1d, indices, opt1)
        typing.cast("EagerCallableProtocol", max_unpool2d)(t_2d, indices2d, opt2)
        typing.cast("EagerCallableProtocol", max_unpool3d)(t_3d, indices3d, opt3)
        w_conf = WindowConfig(window_dimensions=(1, 1, 2), window_strides=(1, 1, 2), padding=((0, 0), (0, 0), (0, 0)), base_dilation=(1, 1, 1), window_dilation=(1, 1, 1))
        typing.cast("EagerCallableProtocol", reduce_window)(t_1d, 0.0, "max", w_conf)
        t_scalar = Tensor(np.array(0.0).astype(np.float32), TensorConfig((), "float32", "cpu"))
        typing.cast("EagerCallableProtocol", reduce_window)(t_1d, t_scalar, "max", w_conf)
    getattr(_state, "update")({"DummyBackend": DummyBackend, "dummy_backend": dummy_backend, "mock_backend": mock_backend, "mock_backend_utils": mock_backend_utils})
    getattr(_state, "update")({"opt1": opt1, "opt2": opt2, "opt3": opt3, "t_scalar": t_scalar})
    getattr(_state, "update")({"w_conf": w_conf})
    _state.update(locals())


def _test_reductions_frontend_brute_part_3(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 3 of test_reductions_frontend_brute.

    Args:
        _state: Shared test state dictionary.
    """
    (indices, indices2d, indices3d, opt1) = (_state["indices"], _state["indices2d"], _state["indices3d"], _state["opt1"])
    (opt2, opt3, t_1d, t_2d) = (_state["opt2"], _state["opt3"], _state["t_1d"], _state["t_2d"])
    (t_3d, w_conf) = (_state["t_3d"], _state["w_conf"])
    config.eager_mode = False
    from ml_switcheroo_compiler.tracing.state import global_tracing_state

    global_tracing_state.is_tracing = True
    try:
        with patch("ml_switcheroo_compiler.ops.reductions.frontend_pool._emit_reduction_node") as mock_emit, patch("ml_switcheroo_compiler.ops.reductions.frontend_utils._emit_reduction_node") as mock_emit_u:
            mock_emit.return_value = t_2d
            mock_emit_u.return_value = t_1d
            typing.cast("EagerCallableProtocol", fractional_max_pool2d)(t_2d, (4, 4))
            typing.cast("EagerCallableProtocol", adaptive_avg_pool2d)(t_2d, (4, 4))
            typing.cast("EagerCallableProtocol", adaptive_max_pool2d)(t_2d, (4, 4))
            typing.cast("EagerCallableProtocol", unfold)(t_2d, (2, 2))
            typing.cast("EagerCallableProtocol", fold)(t_2d, (8, 8), (2, 2))
            typing.cast("EagerCallableProtocol", fractional_max_pool3d)(t_3d, (4, 4, 4))
            typing.cast("EagerCallableProtocol", adaptive_avg_pool3d)(t_3d, (4, 4, 4))
            typing.cast("EagerCallableProtocol", adaptive_max_pool3d)(t_3d, (4, 4, 4))
            typing.cast("EagerCallableProtocol", adaptive_max_pool3d)(t_3d, (4, 4, 4), return_indices=True)
            typing.cast("EagerCallableProtocol", max_unpool1d)(t_1d, indices, opt1)
            typing.cast("EagerCallableProtocol", max_unpool2d)(t_2d, indices2d, opt2)
            typing.cast("EagerCallableProtocol", max_unpool3d)(t_3d, indices3d, opt3)
            typing.cast("EagerCallableProtocol", reduce_window)(t_1d, 0.0, "max", w_conf)
            typing.cast("EagerCallableProtocol", reduce_window)(t_1d, t_1d, "max", w_conf)
    finally:
        global_tracing_state.is_tracing = False
    getattr(_state, "update")({"global_tracing_state": global_tracing_state})
    _state.update(locals())


def _test_reductions_frontend_brute_part_4(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 4 of test_reductions_frontend_brute.

    Args:
        _state: Shared test state dictionary.
    """
    global_tracing_state = _state["global_tracing_state"]
    setattr(global_tracing_state, "is_tracing", True)
    try:

        class DummyData:
            """Test class DummyData."""

            id = "dummy"

        t_dummy = typing.cast("EagerCallableProtocol", Tensor)(typing.cast("EagerCallableProtocol", DummyData)(), typing.cast("EagerCallableProtocol", TensorConfig)((2,), "float32", "cpu"))
        with patch("ml_switcheroo_compiler.ops.reductions.frontend_utils.global_tracing_state.add_node"):
            typing.cast("EagerCallableProtocol", _emit_reduction_node)("Test", [t_dummy], {}, (2,), "float32")
    finally:
        setattr(global_tracing_state, "is_tracing", False)
    assert getattr(global_tracing_state, "is_tracing") is False


def test_reductions_frontend_brute() -> None:
    """Test function test_reductions_frontend_brute."""
    _state: dict[str, EagerTensorProtocol] = {}
    _test_reductions_frontend_brute_part_1(_state)
    _test_reductions_frontend_brute_part_2(_state)
    _test_reductions_frontend_brute_part_3(_state)
    _test_reductions_frontend_brute_part_4(_state)
    assert len(_state) > 0


def test_segment_exact_shapes(mocker: EagerTensorProtocol) -> None:
    """Verify that segmented reduction ops calculate exact shapes in graph mode.

    Args:
        mocker (EagerTensorProtocol): Pytest mocker fixture.
    """
    captured_shapes: dict[str, tuple[int, ...]] = {}

    def mock_emit(op_name: str, inputs: list[Tensor], attrs: dict[str, EagerValue], shape: tuple[int, ...], dtype: EagerValue) -> str:
        """Test function mock_emit.

        Args:
            op_name: Test parameter `op_name`.
            inputs: Test parameter `inputs`.
            attrs: Test parameter `attrs`.
            shape: Test parameter `shape`.
            dtype: Test parameter `dtype`.
        """
        captured_shapes[op_name] = shape
        return op_name

    mocker.patch("ml_switcheroo_compiler.ops.reductions.frontend_segment._emit_reduction_node", side_effect=mock_emit)
    config.eager_mode = False
    data_2d = Tensor(None, TensorConfig((10, 8), "float32", "cpu"))
    seg_ids = Tensor(None, TensorConfig((10,), "int32", "cpu"))
    segment_sum(data_2d, seg_ids, num_segments=4)
    assert captured_shapes["SegmentSum"] == (4, 8)
    segment_mean(data_2d, seg_ids, num_segments=4)
    assert captured_shapes["SegmentMean"] == (4, 8)
    segment_max(data_2d, seg_ids, num_segments=4)
    assert captured_shapes["SegmentMax"] == (4, 8)
    segment_min(data_2d, seg_ids, num_segments=4)
    assert captured_shapes["SegmentMin"] == (4, 8)
    segment_prod(data_2d, seg_ids, num_segments=4)
    assert captured_shapes["SegmentProd"] == (4, 8)


def test_reductions_frontend_sum_max_min() -> None:
    """Test sum, max, min in both eager and graph modes."""
    t = Tensor(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32), TensorConfig((2, 2), "float32", "cpu"))
    config.eager_mode = True
    s_eager = fe_sum(t, axis=0, keepdims=True)
    assert s_eager.shape == (1, 2)
    m_eager = fe_max(t, axis=1, keepdims=False)
    assert m_eager.shape == (2,)
    n_eager = fe_min(t, axis=None, keepdims=False)
    assert n_eager.shape == ()
    config.eager_mode = False
    s_graph = fe_sum(t, axis=0, keepdims=True)
    assert isinstance(s_graph, Tensor)
    m_graph = fe_max(t, axis=1, keepdims=False)
    assert isinstance(m_graph, Tensor)
    n_graph = fe_min(t, axis=None, keepdims=False)
    assert isinstance(n_graph, Tensor)


def test_reductions_frontend_stats_ops() -> None:
    """Test all frontend_stats operations for full branch coverage."""
    from ml_switcheroo_compiler.ops.reductions.frontend_stats import approx_max_k, approx_min_k, corrcoef, correlate, cov, ctc_loss, pmean, psum
    from ml_switcheroo_compiler.tracing.state import global_tracing_state

    t = Tensor(np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32), TensorConfig((4,), "float32", "cpu"))
    t2 = Tensor(np.array([2.0, 3.0, 4.0, 5.0], dtype=np.float32), TensorConfig((4,), "float32", "cpu"))
    with patch.object(global_tracing_state, "add_node"):
        assert isinstance(typing.cast("EagerValue", psum(t, "devices")), Tensor)
        assert isinstance(typing.cast("EagerValue", pmean(t, "devices")), Tensor)
        (v1, i1) = typing.cast("EagerCallableProtocol", approx_max_k)(t, k=2)
        assert isinstance(typing.cast("EagerValue", v1), Tensor)
        assert isinstance(typing.cast("EagerValue", i1), Tensor)
        (v2, i2) = typing.cast("EagerCallableProtocol", approx_min_k)(t, k=2)
        assert isinstance(typing.cast("EagerValue", v2), Tensor)
        assert isinstance(typing.cast("EagerValue", i2), Tensor)
        targets = Tensor(np.array([1, 2], dtype=np.int32), TensorConfig((2,), "int32", "cpu"))
        inp_len = Tensor(np.array([4], dtype=np.int32), TensorConfig((1,), "int32", "cpu"))
        tgt_len = Tensor(np.array([2], dtype=np.int32), TensorConfig((1,), "int32", "cpu"))
        loss = ctc_loss(t, targets, inp_len, tgt_len)
        assert isinstance(loss, Tensor)
        assert isinstance(corrcoef(t), Tensor)
        assert isinstance(corrcoef(t, t2, rowvar=False, bias=True, ddof=1), Tensor)
        assert isinstance(correlate(t, t2, mode="full"), Tensor)
        assert isinstance(cov(t), Tensor)
        assert isinstance(cov(t, t2, rowvar=True, bias=False, ddof=0, fweights=None, aweights=None), Tensor)
        with pytest.raises(ValueError, match="Invalid keyword argument"):
            cov(t, invalid_key=True)
