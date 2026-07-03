"""Unit tests for basic linear algebra operations including matrix multiplication, dot.

product, and Einstein summation.
"""

from unittest.mock import MagicMock, patch

import numpy as np

# Check 52-53: _emit_linalg_node outside tracing
import pytest

from ml_switcheroo_compiler.core.config import ConfigContext, config
from ml_switcheroo_compiler.core.device import Device, DeviceType
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.ops.configs import ConvConfig
from ml_switcheroo_compiler.ops.linalg import (
    cholesky,
    conv_general_dilated,
    cross,
    det,
    dot,
    dot_general,
    eigh,
    eigvalsh,
    fft,
    fft2d,
    fft3d,
    ifft,
    ifft2d,
    ifft3d,
    inv,
    irfft,
    irfft2d,
    irfft3d,
    matmul,
    matrix_power,
    qr,
    rfft,
    rfft2d,
    rfft3d,
    slogdet,
    solve,
    svd,
)
from ml_switcheroo_compiler.ops.linalg.conv_ops import ConvGeneralDilated
from ml_switcheroo_compiler.ops.linalg.decompositions import pinv
from ml_switcheroo_compiler.ops.linalg.dot import Dot, DotGeneral
from ml_switcheroo_compiler.ops.linalg.einsum import Einsum
from ml_switcheroo_compiler.ops.linalg.fft_ops import (
    Fft,
    Fft2d,
    Fft3d,
    Ifft,
    Ifft2d,
    Ifft3d,
    Irfft,
    Irfft2d,
    Irfft3d,
    Rfft,
    Rfft2d,
    Rfft3d,
)
from ml_switcheroo_compiler.ops.linalg.products import Matmul
from ml_switcheroo_compiler.ops.linalg.utils import _emit_linalg_node
from ml_switcheroo_compiler.tracing.state import global_tracing_state
from ml_switcheroo_compiler.tracing.tracer import ProxyTensor


def test_matmul_op() -> None:
    """Tests the matrix multiplication operator.

    This test verifies that the Matmul operator correctly infers the output shape
    of two matrices and evaluates the matrix multiplication using NumPy's matmul
    implementation

    Returns:
    None
    """
    op = Matmul()
    a = np.random.randn(2, 3)
    b = np.random.randn(3, 4)

    assert op.infer_shape(a.shape, b.shape) == (2, 4)
    assert op.infer_shape(None, None) is None

    res = op.eager_eval(a, b)
    assert np.allclose(res, np.matmul(a, b))


def test_dot_op() -> None:
    """Tests the dot product operator.

    This test verifies that the Dot operator correctly handles shape inference
    and evaluates the dot product of two 1D arrays using NumPy's dot
    implementation

    Returns:
    None
    """
    op = Dot()
    a = np.random.randn(3)
    b = np.random.randn(3)

    assert op.infer_shape(a.shape, b.shape) is None

    res = op.eager_eval(a, b)
    assert np.allclose(res, np.dot(a, b))


def test_einsum_op() -> None:
    """Tests the Einstein summation operator.

    This test verifies that the Einsum operator correctly handles shape inference
    and evaluates the Einstein summation of two matrices using NumPy's einsum
    implementation with a specified subscript string

    Returns:
    None
    """
    op = Einsum()
    a = np.random.randn(2, 3)
    b = np.random.randn(3, 4)
    subscripts = "ij,jk->ik"

    assert op.infer_shape(subscripts, a.shape, b.shape) == (2, 4)

    res = op.eager_eval(subscripts, a, b)
    assert np.allclose(res, np.einsum(subscripts, a, b))


def test_dot_general_opdef() -> None:
    """Test dot_general_opdef."""
    op = DotGeneral()

    class DummyShape:
        """Dummy Shape."""

        def __init__(self, shape: tuple) -> None:
            """Init.

            Args:
                shape (tuple): The shape.
            """
            self.shape = shape

    assert op.infer_shape(None, None, (((0,), (0,)), ((), ()))) == ()
    assert op.infer_shape(DummyShape((2, 3)), DummyShape((3, 4)), (((1,), (0,)), ((), ()))) == (
        2,
        4,
    )
    assert op.infer_shape(
        DummyShape((5, 2, 3)),
        DummyShape((5, 3, 4)),
        (((2,), (1,)), ((0,), (0,))),
    ) == (5, 2, 4)

    x = np.ones((2, 3))
    y = np.ones((3, 4))
    out = op.eager_eval(x, y, (((1,), (0,)), ((), ())))
    assert out.shape == (2, 4)

    xb = np.ones((5, 2, 3))
    yb = np.ones((5, 3, 4))
    outb = op.eager_eval(xb, yb, (((2,), (1,)), ((0,), (0,))))
    assert outb.shape == (5, 2, 4)

    assert op.emit_jax() == "Not implemented DotGeneral"
    assert op.emit_keras() == "Not implemented DotGeneral"
    assert op.emit_mlx() == "Not implemented DotGeneral"
    assert op.emit_pytorch() == "Not implemented DotGeneral"
    assert op.emit_tensorflow() == "Not implemented DotGeneral"


def test_dot_general_frontend() -> None:
    """Test dot_general_frontend."""
    device = Device(DeviceType.CPU)
    x = Tensor(np.ones((2, 3)), TensorConfig((2, 3), DType.Int32, device))
    y = Tensor(np.ones((3, 4)), TensorConfig((3, 4), DType.Int32, device))

    with ConfigContext(eager_mode=True):
        out = dot_general(x, y, (((1,), (0,)), ((), ())))
        assert isinstance(out, Tensor)

    graph = global_tracing_state.start_tracing("test_dot_general")
    try:
        x_proxy = Tensor(
            ProxyTensor(id="x", shape=(2, 3), dtype="int32"),
            TensorConfig((2, 3), DType.Int32, device),
        )
        y_proxy = Tensor(
            ProxyTensor(id="y", shape=(3, 4), dtype="int32"),
            TensorConfig((3, 4), DType.Int32, device),
        )
        out = dot_general(x_proxy, y_proxy, (((1,), (0,)), ((), ())))
        assert out.shape == (2, 4)
        node = graph.nodes[out.data.id]
        assert node.op_type == "DotGeneral"
        assert node.attributes["dimension_numbers"] == (((1,), (0,)), ((), ()))
    finally:
        global_tracing_state.stop_tracing()


def test_conv_general_dilated_opdef() -> None:
    """Test conv_general_dilated_opdef."""
    op = ConvGeneralDilated()

    cfg = ConvConfig([1], "SAME")
    assert op.infer_shape(None, None, cfg) == ()

    class Dummy:
        """Docstring."""

        shape = (1, 3, 32, 32)

    assert op.infer_shape(Dummy(), Dummy(), ConvConfig([1], "SAME")) == ()

    x = np.ones((1, 3, 32, 32))
    w = np.ones((16, 3, 3, 3))
    out = op.eager_eval(x, w, ConvConfig(window_strides=[1, 1], padding="SAME"))
    assert out.shape == (1, 16, 32, 32)

    assert op.emit_jax() == "Not implemented ConvGeneralDilated"
    assert op.emit_keras() == "Not implemented ConvGeneralDilated"
    assert op.emit_mlx() == "Not implemented ConvGeneralDilated"
    assert op.emit_pytorch() == "Not implemented ConvGeneralDilated"
    assert op.emit_tensorflow() == "Not implemented ConvGeneralDilated"


def test_conv_general_dilated_frontend() -> None:
    """Test conv_general_dilated_frontend."""
    device = Device(DeviceType.CPU)
    x = Tensor(np.ones((1, 3, 32, 32)), TensorConfig((1, 3, 32, 32), DType.Float32, device))
    w = Tensor(np.ones((16, 3, 3, 3)), TensorConfig((16, 3, 3, 3), DType.Float32, device))

    with ConfigContext(eager_mode=True):
        assert conv_general_dilated(x, w, ConvConfig(window_strides=[1, 1], padding="SAME")).shape == (1, 16, 32, 32)

    graph = global_tracing_state.start_tracing("test_conv")
    try:
        x_proxy = Tensor(
            ProxyTensor(id="x", shape=(1, 3, 32, 32), dtype="float32"),
            TensorConfig((1, 3, 32, 32), DType.Float32, device),
        )
        w_proxy = Tensor(
            ProxyTensor(id="w", shape=(16, 3, 3, 3), dtype="float32"),
            TensorConfig((16, 3, 3, 3), DType.Float32, device),
        )
        out = conv_general_dilated(x_proxy, w_proxy, ConvConfig(window_strides=[1, 1], padding="SAME"))
        assert out.shape == ()
        node = graph.nodes[out.data.id]
        assert node.op_type == "ConvGeneralDilated"
        assert node.attributes["config"].window_strides == [1, 1]
    finally:
        global_tracing_state.stop_tracing()


def test_fft_rfft_opdef() -> None:
    """Test fft_rfft_opdef."""
    op_fft = Fft()
    op_rfft = Rfft()

    class DummyShape:
        """Docstring."""

        shape = (10,)

    assert op_fft.infer_shape(None, 5) == ()
    assert op_fft.infer_shape(DummyShape(), 5) == (5,)

    assert op_rfft.infer_shape(None, 10) == ()
    assert op_rfft.infer_shape(DummyShape(), None) == (6,)

    x = np.random.rand(10)
    out_fft = op_fft.eager_eval(x)
    assert out_fft.shape == (10,)

    out_rfft = op_rfft.eager_eval(x)
    assert out_rfft.shape == (6,)

    assert op_fft.emit_jax() == "Not implemented Fft"
    assert op_fft.emit_keras() == "Not implemented Fft"
    assert op_fft.emit_mlx() == "Not implemented Fft"
    assert op_fft.emit_pytorch() == "Not implemented Fft"
    assert op_fft.emit_tensorflow() == "Not implemented Fft"

    assert op_rfft.emit_jax() == "Not implemented Rfft"
    assert op_rfft.emit_keras() == "Not implemented Rfft"
    assert op_rfft.emit_mlx() == "Not implemented Rfft"
    assert op_rfft.emit_pytorch() == "Not implemented Rfft"
    assert op_rfft.emit_tensorflow() == "Not implemented Rfft"


def test_fft_rfft_frontend() -> None:
    """Test fft_rfft_frontend."""
    device = Device(DeviceType.CPU)
    x = Tensor(np.random.rand(10), TensorConfig((10,), DType.Float32, device))

    with ConfigContext(eager_mode=True):
        out_f = fft(x)
        assert out_f.shape == (10,)
        out_r = rfft(x)
        assert out_r.shape == (6,)

    graph = global_tracing_state.start_tracing("test_fft")
    try:
        x_proxy = Tensor(
            ProxyTensor(id="x", shape=(10,), dtype="float32"),
            TensorConfig((10,), DType.Float32, device),
        )
        out_f2 = fft(x_proxy)
        assert out_f2.shape == (10,)
        node = graph.nodes[out_f2.data.id]
        assert node.op_type == "Fft"

        out_r2 = rfft(x_proxy)
        assert out_r2.shape == (6,)
        node2 = graph.nodes[out_r2.data.id]
        assert node2.op_type == "Rfft"
    finally:
        global_tracing_state.stop_tracing()


def test_linalg_eager_missing() -> None:
    """Test missing linalg eager ops."""
    device = "cpu"
    with ConfigContext(eager_mode=True):
        a = Tensor(np.ones((2, 3)), TensorConfig((2, 3), DType.Float32, device))
        b = Tensor(np.ones((3, 4)), TensorConfig((3, 4), DType.Float32, device))
        c = Tensor(np.eye(3), TensorConfig((3, 3), DType.Float32, device))
        d = Tensor(np.ones((3,)), TensorConfig((3,), DType.Float32, device))
        d2 = Tensor(np.ones((3,)), TensorConfig((3,), DType.Float32, device))

        matmul(a, b)
        dot(d, d)
        cholesky(c)
        svd(c)
        qr(c)
        eigh(c)
        eigvalsh(c)
        inv(c)
        solve(c, d)
        det(c)
        slogdet(c)
        cross(d, d2)
        fft(d)
        rfft(d)
        matrix_power(c, 2)
        try:
            dot_general(c, d, (((1,), (0,)), ((), ())))
        except Exception:
            pass

    # dot_general tracing with scalar
    with ConfigContext(eager_mode=False):
        global_tracing_state.start_tracing("dot_general_scalar")
        s = Tensor(ProxyTensor("s", (), "float32"), TensorConfig((), DType.Float32, device))
        dot_general(s, s, (((), ()), ((), ())))

        # Test 595: lhs_batch non-empty
        b = Tensor(ProxyTensor("b", (2, 3), "float32"), TensorConfig((2, 3), DType.Float32, device))
        dot_general(b, b, (((1,), (1,)), ((0,), (0,))))

        global_tracing_state.stop_tracing()

    with ConfigContext(eager_mode=False), pytest.raises(RuntimeError):
        _emit_linalg_node("Test", [], {}, [()], [DType.Float32])


def test_decompositions_eager_pinv() -> None:
    """Test eager pinv."""
    config.eager_mode = True
    data = np.array([[1.0, 2.0], [3.0, 4.0]])
    inp = Tensor(data, TensorConfig((2, 2), "float32", Device("cpu")))

    mock_backend = MagicMock()
    mock_backend.execute_op.return_value = np.array([[1.0]])
    with patch("ml_switcheroo_compiler.backends.registry.get_active_backend", return_value=mock_backend):
        pinv(inp)
    config.eager_mode = False


def test_all_ffts() -> None:
    """Test all ffts."""
    assert Ifft().infer_shape(TensorConfig((2,), "float32", "cpu")) == (2,)
    assert Fft2d().infer_shape(TensorConfig((2,), "float32", "cpu")) == (2,)
    assert Ifft2d().infer_shape(TensorConfig((2,), "float32", "cpu")) == (2,)
    assert Fft3d().infer_shape(TensorConfig((2,), "float32", "cpu")) == (2,)
    assert Ifft3d().infer_shape(TensorConfig((2,), "float32", "cpu")) == (2,)
    assert Rfft2d().infer_shape(TensorConfig((2,), "float32", "cpu")) == (2,)
    assert Rfft3d().infer_shape(TensorConfig((2,), "float32", "cpu")) == (2,)
    assert Irfft().infer_shape(TensorConfig((2,), "float32", "cpu")) == (2,)
    assert Irfft2d().infer_shape(TensorConfig((2,), "float32", "cpu")) == (2,)
    assert Irfft3d().infer_shape(TensorConfig((2,), "float32", "cpu")) == (2,)

    device = Device("cpu")
    with ConfigContext(eager_mode=False):
        global_tracing_state.start_tracing()
        try:
            a = Tensor("dummy", TensorConfig((4, 4, 4), DType.Float32, device))
            fft(a)
            fft2d(a)
            fft3d(a)
            ifft(a)
            ifft2d(a)
            ifft3d(a)
            rfft(a)
            rfft2d(a)
            rfft3d(a)
            irfft(a)
            irfft2d(a)
            irfft3d(a)

            # with shapes specified
            fft2d(a, s=(2, 2))
            fft3d(a, s=(2, 2, 2))
            ifft2d(a, s=(2, 2))
            ifft3d(a, s=(2, 2, 2))
            rfft2d(a, s=(2, 2))
            rfft3d(a, s=(2, 2, 2))
            irfft(a, n=2)
            irfft2d(a, s=(2, 2))
            irfft3d(a, s=(2, 2, 2))
        finally:
            global_tracing_state.stop_tracing()


def test_all_ffts_eager_extra() -> None:
    """Test all ffts eager."""
    device = Device("cpu")
    with ConfigContext(eager_mode=True):
        with patch("ml_switcheroo_compiler.backends.registry.get_active_backend") as mock_backend:
            mock_backend.return_value.execute_op.return_value = np.zeros((1,))
            mock_backend.return_value.array.return_value = np.zeros((1,))
            a = Tensor(
                np.zeros((4, 4, 4), dtype=np.float32),
                TensorConfig((4, 4, 4), DType.Float32, device),
            )

            fft(a, n=2)
            fft2d(a, s=(2, 2))
            fft3d(a, s=(2, 2, 2))
            ifft(a, n=2)
            ifft2d(a, s=(2, 2))
            ifft3d(a, s=(2, 2, 2))
            rfft(a, n=2)
            rfft2d(a, s=(2, 2))
            rfft3d(a, s=(2, 2, 2))
            irfft(a, n=2)
            irfft2d(a, s=(2, 2))
            irfft3d(a, s=(2, 2, 2))

            fft(a)
            fft2d(a)
            fft3d(a)
            ifft(a)
            ifft2d(a)
            ifft3d(a)
            rfft(a)
            rfft2d(a)
            rfft3d(a)
            irfft(a)
            irfft2d(a)
            irfft3d(a)
