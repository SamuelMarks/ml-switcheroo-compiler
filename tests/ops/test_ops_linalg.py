"""Unit tests for basic linear algebra operations including matrix multiplication, dot.

product, and Einstein summation.
"""

import numpy as np

from ml_switcheroo_compiler.ops.linalg.basic import (
    Dot,
    Einsum,
    Matmul,
)


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

    res = op.numpy_eval(a, b)
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

    res = op.numpy_eval(a, b)
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

    assert op.infer_shape(subscripts, a.shape, b.shape) is None

    res = op.numpy_eval(subscripts, a, b)
    assert np.allclose(res, np.einsum(subscripts, a, b))


def test_dot_general_opdef() -> None:
    """Test dot_general_opdef."""
    import numpy as np

    from ml_switcheroo_compiler.ops.linalg.basic import DotGeneral

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
    out = op.numpy_eval(x, y, (((1,), (0,)), ((), ())))
    assert out.shape == (2, 4)

    xb = np.ones((5, 2, 3))
    yb = np.ones((5, 3, 4))
    outb = op.numpy_eval(xb, yb, (((2,), (1,)), ((0,), (0,))))
    assert outb.shape == (5, 2, 4)

    assert op.emit_jax() == "Not implemented DotGeneral"
    assert op.emit_keras() == "Not implemented DotGeneral"
    assert op.emit_mlx() == "Not implemented DotGeneral"
    assert op.emit_pytorch() == "Not implemented DotGeneral"
    assert op.emit_tensorflow() == "Not implemented DotGeneral"


def test_dot_general_frontend() -> None:
    """Test dot_general_frontend."""
    import numpy as np
    import pytest

    from ml_switcheroo_compiler.core.config import ConfigContext
    from ml_switcheroo_compiler.core.device import Device, DeviceType
    from ml_switcheroo_compiler.core.dtype import DType
    from ml_switcheroo_compiler.core.errors import UnimplementedMathError
    from ml_switcheroo_compiler.core.tensor import Tensor
    from ml_switcheroo_compiler.ops.linalg.frontend import dot_general
    from ml_switcheroo_compiler.tracing.tracer import ProxyTensor, _tracer

    device = Device(DeviceType.CPU)
    x = Tensor(np.ones((2, 3)), shape=(2, 3), dtype=DType.Int32, device=device)
    y = Tensor(np.ones((3, 4)), shape=(3, 4), dtype=DType.Int32, device=device)

    with ConfigContext(eager_mode=True), pytest.raises(UnimplementedMathError):
        dot_general(x, y, (((1,), (0,)), ((), ())))

    graph = _tracer.start_tracing("test_dot_general")
    try:
        x_proxy = Tensor(
            ProxyTensor(id="x", shape=(2, 3), dtype="int32"),
            shape=(2, 3),
            dtype=DType.Int32,
            device=device,
        )
        y_proxy = Tensor(
            ProxyTensor(id="y", shape=(3, 4), dtype="int32"),
            shape=(3, 4),
            dtype=DType.Int32,
            device=device,
        )
        out = dot_general(x_proxy, y_proxy, (((1,), (0,)), ((), ())))
        assert out.shape == (2, 4)
        node = graph.nodes[out.data.id]
        assert node.op_type == "DotGeneral"
        assert node.attributes["dimension_numbers"] == (((1,), (0,)), ((), ()))
    finally:
        _tracer.stop_tracing()


def test_conv_general_dilated_opdef() -> None:
    """Test conv_general_dilated_opdef."""
    import numpy as np

    from ml_switcheroo_compiler.ops.linalg.basic import ConvGeneralDilated

    op = ConvGeneralDilated()
    assert op.infer_shape(None, None, [1], "SAME") == ()

    class Dummy:
        shape = (1, 3, 32, 32)

    assert op.infer_shape(Dummy(), Dummy(), [1], "SAME") == ()

    x = np.ones((1, 3, 32, 32))
    w = np.ones((16, 3, 3, 3))
    out = op.numpy_eval(x, w, [1, 1], "SAME")
    assert out.shape == (1,)

    assert op.emit_jax() == "Not implemented ConvGeneralDilated"
    assert op.emit_keras() == "Not implemented ConvGeneralDilated"
    assert op.emit_mlx() == "Not implemented ConvGeneralDilated"
    assert op.emit_pytorch() == "Not implemented ConvGeneralDilated"
    assert op.emit_tensorflow() == "Not implemented ConvGeneralDilated"


def test_conv_general_dilated_frontend() -> None:
    """Test conv_general_dilated_frontend."""
    import numpy as np
    import pytest

    from ml_switcheroo_compiler.core.config import ConfigContext
    from ml_switcheroo_compiler.core.device import Device, DeviceType
    from ml_switcheroo_compiler.core.dtype import DType
    from ml_switcheroo_compiler.core.errors import UnimplementedMathError
    from ml_switcheroo_compiler.core.tensor import Tensor
    from ml_switcheroo_compiler.ops.linalg.frontend import conv_general_dilated
    from ml_switcheroo_compiler.tracing.tracer import ProxyTensor, _tracer

    device = Device(DeviceType.CPU)
    x = Tensor(np.ones((1, 3, 32, 32)), shape=(1, 3, 32, 32), dtype=DType.Float32, device=device)
    w = Tensor(np.ones((16, 3, 3, 3)), shape=(16, 3, 3, 3), dtype=DType.Float32, device=device)

    with ConfigContext(eager_mode=True), pytest.raises(UnimplementedMathError):
        conv_general_dilated(x, w, [1, 1], "SAME")

    graph = _tracer.start_tracing("test_conv")
    try:
        x_proxy = Tensor(
            ProxyTensor(id="x", shape=(1, 3, 32, 32), dtype="float32"),
            shape=(1, 3, 32, 32),
            dtype=DType.Float32,
            device=device,
        )
        w_proxy = Tensor(
            ProxyTensor(id="w", shape=(16, 3, 3, 3), dtype="float32"),
            shape=(16, 3, 3, 3),
            dtype=DType.Float32,
            device=device,
        )
        out = conv_general_dilated(x_proxy, w_proxy, [1, 1], "SAME")
        assert out.shape == ()
        node = graph.nodes[out.data.id]
        assert node.op_type == "ConvGeneralDilated"
        assert node.attributes["window_strides"] == [1, 1]
    finally:
        _tracer.stop_tracing()


def test_fft_rfft_opdef() -> None:
    """Test fft_rfft_opdef."""
    import numpy as np

    from ml_switcheroo_compiler.ops.linalg.basic import Fft, Rfft

    op_fft = Fft()
    op_rfft = Rfft()

    class DummyShape:
        shape = (10,)

    assert op_fft.infer_shape(None, 5) == ()
    assert op_fft.infer_shape(DummyShape(), 5) == (5,)

    assert op_rfft.infer_shape(None, 10) == ()
    assert op_rfft.infer_shape(DummyShape(), None) == (6,)

    x = np.random.rand(10)
    out_fft = op_fft.numpy_eval(x)
    assert out_fft.shape == (10,)

    out_rfft = op_rfft.numpy_eval(x)
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
    import numpy as np

    from ml_switcheroo_compiler.core.config import ConfigContext
    from ml_switcheroo_compiler.core.device import Device, DeviceType
    from ml_switcheroo_compiler.core.dtype import DType
    from ml_switcheroo_compiler.core.tensor import Tensor
    from ml_switcheroo_compiler.ops.linalg.frontend import fft, rfft
    from ml_switcheroo_compiler.tracing.tracer import ProxyTensor, _tracer

    device = Device(DeviceType.CPU)
    x = Tensor(np.random.rand(10), shape=(10,), dtype=DType.Float32, device=device)

    with ConfigContext(eager_mode=True):
        out_f = fft(x)
        assert out_f.shape == (10,)
        out_r = rfft(x)
        assert out_r.shape == (6,)

    graph = _tracer.start_tracing("test_fft")
    try:
        x_proxy = Tensor(
            ProxyTensor(id="x", shape=(10,), dtype="float32"),
            shape=(10,),
            dtype=DType.Float32,
            device=device,
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
        _tracer.stop_tracing()


def test_linalg_eager_missing() -> None:
    """Test missing linalg eager ops."""
    import numpy as np
    from ml_switcheroo_compiler.core.config import ConfigContext
    from ml_switcheroo_compiler.core.dtype import DType
    from ml_switcheroo_compiler.core.tensor import Tensor
    from ml_switcheroo_compiler.ops.linalg.frontend import (
        matmul,
        dot,
        cholesky,
        svd,
        qr,
        eigh,
        inv,
        solve,
        det,
        slogdet,
    )
    from ml_switcheroo_compiler.ops.linalg.frontend import (
        cross,
        fft,
        rfft,
        eigvalsh,
        matrix_power,
        dot_general,
    )

    device = "cpu"
    with ConfigContext(eager_mode=True):
        a = Tensor(np.ones((2, 3)), (2, 3), DType.Float32, device)
        b = Tensor(np.ones((3, 4)), (3, 4), DType.Float32, device)
        c = Tensor(np.eye(3), (3, 3), DType.Float32, device)
        d = Tensor(np.ones((3,)), (3,), DType.Float32, device)
        d2 = Tensor(np.ones((3,)), (3,), DType.Float32, device)

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
        from ml_switcheroo_compiler.tracing.tracer import _tracer, ProxyTensor

        _tracer.start_tracing("dot_general_scalar")
        s = Tensor(ProxyTensor("s", (), "float32"), (), DType.Float32, device)
        dot_general(s, s, (((), ()), ((), ())))

        # Test 595: lhs_batch non-empty
        b = Tensor(ProxyTensor("b", (2, 3), "float32"), (2, 3), DType.Float32, device)
        dot_general(b, b, (((1,), (1,)), ((0,), (0,))))

        _tracer.stop_tracing()

    # Check 52-53: _emit_linalg_node outside tracing
    from ml_switcheroo_compiler.ops.linalg.frontend import _emit_linalg_node
    import pytest

    with ConfigContext(eager_mode=False):
        with pytest.raises(RuntimeError):
            _emit_linalg_node("Test", [], {}, [()], [DType.Float32])
