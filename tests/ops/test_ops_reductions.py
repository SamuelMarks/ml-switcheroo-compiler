"""Unit tests for basic reduction operations.

This module verifies the correctness of reduction operations such as Sum, Mean, Max, and
Min by comparing their shape inference and evaluation results against their equivalent
NumPy implementations.
"""

import numpy as np

from ml_switcheroo_compiler.ops.reductions.basic import (
    Max,
    Mean,
    Min,
    Sum,
)


def test_reduction_ops() -> None:
    """Tests the correctness of basic reduction operations against NumPy equivalents.

    This test validates that the custom reduction operations (Sum, Mean, Max, Min)
    correctly infer output shapes and produce identical numerical results to their
    corresponding NumPy functions (np.sum, np.mean, np.max, np.min) under
    various configurations of axis and keepdims

    Returns:
    None
    """
    x = np.array([[1.0, 2.0], [3.0, 4.0]])

    ops = [
        (Sum(), np.sum),
        (Mean(), np.mean),
        (Max(), np.max),
        (Min(), np.min),
    ]

    for op, np_func in ops:
        assert op.infer_shape(x.shape) == ()
        assert np.allclose(op.numpy_eval(x), np_func(x))
        assert np.allclose(op.numpy_eval(x, axis=0), np_func(x, axis=0))
        assert np.allclose(
            op.numpy_eval(x, axis=1, keepdims=True),
            np_func(x, axis=1, keepdims=True),
        )


def test_segment_sum_opdef() -> None:
    """Test segment_sum_opdef."""
    import numpy as np

    from ml_switcheroo_compiler.ops.reductions.basic import SegmentSum

    ss = SegmentSum()
    assert ss.infer_shape(None, None, None) == ()
    data = np.array([1, 2, 3, 4])
    segment_ids = np.array([0, 0, 1, 1])
    out = ss.numpy_eval(data, segment_ids)
    assert np.array_equal(out, np.array([3, 7]))

    out2 = ss.numpy_eval(data, segment_ids, num_segments=3)
    assert np.array_equal(out2, np.array([3, 7, 0]))

    assert ss.emit_jax() == "Not implemented SegmentSum"
    assert ss.emit_keras() == "Not implemented SegmentSum"
    assert ss.emit_mlx() == "Not implemented SegmentSum"
    assert ss.emit_pytorch() == "Not implemented SegmentSum"
    assert ss.emit_tensorflow() == "Not implemented SegmentSum"


def test_segment_sum_frontend() -> None:
    """Test segment sum frontend."""
    from ml_switcheroo_compiler.ops.reductions import segment_sum

    assert callable(segment_sum)
    import numpy as np
    import pytest

    from ml_switcheroo_compiler.core.config import ConfigContext
    from ml_switcheroo_compiler.core.device import Device, DeviceType
    from ml_switcheroo_compiler.core.dtype import DType
    from ml_switcheroo_compiler.core.errors import UnimplementedMathError
    from ml_switcheroo_compiler.core.tensor import Tensor
    from ml_switcheroo_compiler.tracing.tracer import ProxyTensor, _tracer

    device = Device(DeviceType.CPU)
    x = Tensor(np.ones((4,)), shape=(4,), dtype=DType.Int32, device=device)
    s = Tensor(np.array([0, 0, 1, 1]), shape=(4,), dtype=DType.Int32, device=device)

    with ConfigContext(eager_mode=True), pytest.raises(UnimplementedMathError):
        segment_sum(x, s)

    graph = _tracer.start_tracing("test_ss")
    try:
        x_p = Tensor(
            ProxyTensor(id="x", shape=(4,), dtype="int32"),
            shape=(4,),
            dtype=DType.Int32,
            device=device,
        )
        s_p = Tensor(
            ProxyTensor(id="s", shape=(4,), dtype="int32"),
            shape=(4,),
            dtype=DType.Int32,
            device=device,
        )
        out = segment_sum(x_p, s_p, 2)
        assert out.shape == ()
        node = graph.nodes[out.data.id]
        assert node.op_type == "SegmentSum"
        assert node.attributes["num_segments"] == 2
    finally:
        _tracer.stop_tracing()


def test_reduce_window_opdef() -> None:
    """Test reduce_window_opdef."""
    import numpy as np

    from ml_switcheroo_compiler.ops.reductions.basic import ReduceWindow

    op = ReduceWindow()

    class DummyShape:
        """Dummy Shape."""

        def __init__(self, shape: tuple) -> None:
            """Init.

            Args:
                shape (tuple): The shape.
            """
            self.shape = shape

    assert op.infer_shape(None, 0, "max", [2, 2]) == ()
    assert op.infer_shape(DummyShape((1, 4, 4, 1)), 0, "max", [1, 2, 2, 1], [1, 2, 2, 1]) == (
        1,
        2,
        2,
        1,
    )

    # Test numpy evaluation mock
    x = np.ones((4, 4))
    out = op.numpy_eval(x, 0, "max", [2, 2], [2, 2])
    assert out.shape == (2, 2)
    assert np.all(out == 0)

    assert op.emit_jax() == "Not implemented ReduceWindow"
    assert op.emit_keras() == "Not implemented ReduceWindow"
    assert op.emit_mlx() == "Not implemented ReduceWindow"
    assert op.emit_pytorch() == "Not implemented ReduceWindow"
    assert op.emit_tensorflow() == "Not implemented ReduceWindow"


def test_reduce_window_frontend() -> None:
    """Test reduce_window_frontend."""
    import numpy as np
    import pytest

    from ml_switcheroo_compiler.core.config import ConfigContext
    from ml_switcheroo_compiler.core.device import Device, DeviceType
    from ml_switcheroo_compiler.core.dtype import DType
    from ml_switcheroo_compiler.core.errors import UnimplementedMathError
    from ml_switcheroo_compiler.core.tensor import Tensor
    from ml_switcheroo_compiler.ops.reductions.frontend import reduce_window
    from ml_switcheroo_compiler.tracing.tracer import ProxyTensor, _tracer

    device = Device(DeviceType.CPU)
    x = Tensor(np.ones((4, 4)), shape=(4, 4), dtype=DType.Int32, device=device)

    with ConfigContext(eager_mode=True), pytest.raises(UnimplementedMathError):
        reduce_window(x, 0, "max", [2, 2], [2, 2])

    graph = _tracer.start_tracing("test_reduce_window")
    try:
        x_proxy = Tensor(
            ProxyTensor(id="x", shape=(4, 4), dtype="int32"),
            shape=(4, 4),
            dtype=DType.Int32,
            device=device,
        )
        out = reduce_window(x_proxy, 0, "max", [2, 2], [2, 2])
        assert out.shape == (2, 2)
        node = graph.nodes[out.data.id]
        assert node.op_type == "ReduceWindow"
        assert node.attributes["computation"] == "max"
        assert node.attributes["window_dimensions"] == [2, 2]

        # Test tensor init_value
        init_val_proxy = Tensor(
            ProxyTensor(id="init", shape=(), dtype="int32"),
            shape=(),
            dtype=DType.Int32,
            device=device,
        )
        out2 = reduce_window(x_proxy, init_val_proxy, "max", [2, 2], [2, 2])
        assert graph.nodes[out2.data.id].inputs == ["x", "init"]
    finally:
        _tracer.stop_tracing()


def test_psum_pmean_opdef() -> None:
    """Test psum_pmean_opdef."""
    import numpy as np

    from ml_switcheroo_compiler.ops.reductions.basic import Pmean, Psum

    op_sum = Psum()
    op_mean = Pmean()

    class DummyShape:
        """Docstring."""

        shape = (4, 4)

    assert op_sum.infer_shape(DummyShape(), "batch") == (4, 4)
    assert op_sum.infer_shape(None, "batch") == ()
    assert op_mean.infer_shape(DummyShape(), "batch") == (4, 4)
    assert op_mean.infer_shape(None, "batch") == ()

    x = np.ones((4, 4))
    assert np.array_equal(op_sum.numpy_eval(x, "batch"), x)
    assert np.array_equal(op_mean.numpy_eval(x, "batch"), x)

    for op in (op_sum, op_mean):
        assert op.emit_jax() == f"Not implemented {op.op_name}"
        assert op.emit_keras() == f"Not implemented {op.op_name}"
        assert op.emit_mlx() == f"Not implemented {op.op_name}"
        assert op.emit_pytorch() == f"Not implemented {op.op_name}"
        assert op.emit_tensorflow() == f"Not implemented {op.op_name}"


def test_psum_pmean_frontend() -> None:
    """Test psum_pmean_frontend."""
    import numpy as np
    import pytest

    from ml_switcheroo_compiler.core.config import ConfigContext
    from ml_switcheroo_compiler.core.device import Device, DeviceType
    from ml_switcheroo_compiler.core.dtype import DType
    from ml_switcheroo_compiler.core.errors import UnimplementedMathError
    from ml_switcheroo_compiler.core.tensor import Tensor
    from ml_switcheroo_compiler.ops.reductions.frontend import pmean, psum
    from ml_switcheroo_compiler.tracing.tracer import ProxyTensor, _tracer

    device = Device(DeviceType.CPU)
    x = Tensor(np.ones((4, 4)), shape=(4, 4), dtype=DType.Int32, device=device)

    with ConfigContext(eager_mode=True):
        with pytest.raises(UnimplementedMathError):
            psum(x, "batch")
        with pytest.raises(UnimplementedMathError):
            pmean(x, "batch")

    graph = _tracer.start_tracing("test_pm")
    try:
        x_proxy = Tensor(
            ProxyTensor(id="x", shape=(4, 4), dtype="int32"),
            shape=(4, 4),
            dtype=DType.Int32,
            device=device,
        )
        out_sum = psum(x_proxy, "batch")
        assert out_sum.shape == (4, 4)
        node_sum = graph.nodes[out_sum.data.id]
        assert node_sum.op_type == "Psum"
        assert node_sum.attributes["axis_name"] == "batch"

        out_mean = pmean(x_proxy, "batch")
        assert out_mean.shape == (4, 4)
        node_mean = graph.nodes[out_mean.data.id]
        assert node_mean.op_type == "Pmean"
        assert node_mean.attributes["axis_name"] == "batch"
    finally:
        _tracer.stop_tracing()


def test_reduction_numpy_eval_coverage() -> None:
    """Test numpy eval coverage."""
    import numpy as np

    from ml_switcheroo_compiler.ops.reductions.basic import CountNonzero, Cumsum, Logsumexp, Norm

    x = np.array([1.0, 2.0, 0.0])
    lse = Logsumexp()
    assert lse.numpy_eval(x) is not None

    cnz = CountNonzero()
    assert cnz.numpy_eval(x) is not None

    norm = Norm()
    assert norm.numpy_eval(x) is not None

    cumsum = Cumsum()
    assert cumsum.numpy_eval(x) is not None


def test_reduction_infer_shape_coverage() -> None:
    """Test infer shape coverage."""
    from ml_switcheroo_compiler.ops.reductions.basic import CountNonzero, Cumsum, Logsumexp, Norm

    class DummyShape:
        """Docstring."""

        shape = (2, 2)

    ds = DummyShape()

    lse = Logsumexp()
    assert lse.infer_shape(ds, axis=0) == ()

    cnz = CountNonzero()
    assert cnz.infer_shape(ds, axis=0) == ()

    norm = Norm()
    assert norm.infer_shape(ds, axis=0) == ()

    cumsum = Cumsum()
    assert cumsum.infer_shape(ds, axis=0) == ()


def test_reduction_args_coverage() -> None:
    """Test args coverage."""
    from ml_switcheroo_compiler.ops.reductions.basic import ReductionOp

    class MockReduction(ReductionOp):
        """Docstring."""

        op_name = "MockReduction"
        np_op_name = "mock"

    op = MockReduction()
    assert op._format_args("x", axis=0) == "x, axis=0"
    assert op._format_args("x", keepdims=True) == "x, keepdims=True"


def test_reduce_window_coverage() -> None:
    """Test reduce window coverage."""
    from ml_switcheroo_compiler.ops.reductions.basic import ReduceWindow

    class DummyShape:
        """Docstring."""

        shape = (1, 4, 4, 1)

    rw = ReduceWindow()
    # Test missing defaults
    assert rw.infer_shape(DummyShape(), 0, "max", [1, 2]) == (1, 3, 4, 1)

    # Test with padding, base_dilation, window_dilation
    assert rw.infer_shape(
        DummyShape(), 0, "max", [2, 2], [1, 1], [(1, 1), (0, 0)], [2, 2], [2, 2]
    ) == (1, 5, 4, 1)


def test_reduce_window_coverage_zero_dim() -> None:
    """Test reduce window zero dim."""
    from ml_switcheroo_compiler.ops.reductions.basic import ReduceWindow

    class DummyShape:
        """Docstring."""

        shape = (1, 1)

    rw = ReduceWindow()
    assert rw.infer_shape(DummyShape(), 0, "max", [2, 2]) == (0, 0)
