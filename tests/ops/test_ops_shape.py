"""Unit tests for shape manipulation and state operations in the ml_switcheroo_compiler library."""

from ml_switcheroo_compiler.core.tensor import TensorConfig

import numpy as np

from ml_switcheroo_compiler.ops.shape import BroadcastTo, Reshape, Transpose


def test_reshape_op() -> None:
    """Tests the Reshape operation's shape inference and NumPy evaluation.

    Verifies that the Reshape operation correctly infers the target shape
    and matches the behavior of np.reshape during evaluation

    Returns:
    None
    """
    op = Reshape()
    x = np.array([1, 2, 3, 4])
    newshape = (2, 2)

    assert op.infer_shape(x.shape, newshape) == newshape
    assert np.array_equal(op.eager_eval(x, newshape), np.reshape(x, newshape))


def test_transpose_op() -> None:
    """Tests the Transpose operation's shape inference and NumPy evaluation.

    Verifies that the Transpose operation correctly infers the transposed shape
    with and without specified axes, and matches np.transpose during evaluation

    Returns:
    None
    """
    op = Transpose()
    x = np.random.randn(2, 3)

    assert op.infer_shape(x.shape) is None
    assert op.infer_shape(x.shape, (1, 0)) == (3, 2)
    assert np.array_equal(op.eager_eval(x), np.transpose(x))
    assert np.array_equal(op.eager_eval(x, axes=(1, 0)), np.transpose(x, axes=(1, 0)))


def test_broadcast_to_op() -> None:
    """Tests the BroadcastTo operation's shape inference and NumPy evaluation.

    Verifies that the BroadcastTo operation correctly infers the broadcasted shape
    and matches np.broadcast_to during evaluation

    Returns:
    None
    """
    op = BroadcastTo()
    x = np.array([1, 2])
    shape = (2, 2)

    assert op.infer_shape(x.shape, shape) == shape
    assert np.array_equal(op.eager_eval(x, shape), np.broadcast_to(x, shape))


def test_state_ops() -> None:
    """Tests the shape inference and error handling of state operations.

    Verifies that ReadVariable and AssignVariable correctly infer shapes,
    and raise CompilationError when attempting NumPy evaluation directly

    Returns:
    None
    """
    import pytest

    from ml_switcheroo_compiler.core.errors import CompilationError
    from ml_switcheroo_compiler.ops.base import get_op

    r = get_op("ReadVariable")()
    a = get_op("AssignVariable")()

    assert r.infer_shape(shape=(2,)) == (2,)
    assert a.infer_shape((2,)) == (2,)

    with pytest.raises(CompilationError):
        r.eager_eval()

    with pytest.raises(CompilationError):
        a.eager_eval(1)


def test_dynamic_update_slice() -> None:
    """Test dynamic_update_slice."""
    import numpy as np

    from ml_switcheroo_compiler.core.config import ConfigContext
    from ml_switcheroo_compiler.core.device import Device, DeviceType
    from ml_switcheroo_compiler.core.dtype import DType
    from ml_switcheroo_compiler.core.tensor import Tensor
    from ml_switcheroo_compiler.ops.shape.frontend import dynamic_update_slice
    from ml_switcheroo_compiler.tracing.tracer import ProxyTensor, _tracer

    device = Device(DeviceType.CPU)
    t = Tensor(np.zeros((5,)), TensorConfig((5,), DType.Float32, device))
    u = Tensor(np.ones((2,)), TensorConfig((2,), DType.Float32, device))

    with ConfigContext(eager_mode=True):
        assert dynamic_update_slice(t, u, [0]).shape == (5,)

    graph = _tracer.start_tracing("test")
    try:
        t_proxy = Tensor(
            ProxyTensor(id="t", shape=(5,), dtype="float32"),
            TensorConfig((5,), DType.Float32, device),
        )
        u_proxy = Tensor(
            ProxyTensor(id="u", shape=(2,), dtype="float32"),
            TensorConfig((2,), DType.Float32, device),
        )
        proxy = ProxyTensor(id="idx", shape=(), dtype="int32")
        out = dynamic_update_slice(
            t_proxy, u_proxy, [Tensor(proxy, TensorConfig((), DType.Int32, device))]
        )
        assert out.shape == (5,)
        node = graph.nodes[out.data.id]
        assert node.op_type == "DynamicUpdateSlice"
    finally:
        _tracer.stop_tracing()


def test_dynamic_slice_opdef() -> None:
    """Test dynamic_slice_opdef."""
    import numpy as np

    from ml_switcheroo_compiler.ops.shape import DynamicSlice, DynamicUpdateSlice

    ds = DynamicSlice()
    assert ds.infer_shape(None, None, [2, 3]) == (2, 3)
    x = np.arange(10)
    assert np.array_equal(ds.eager_eval(x, [2], [3]), np.array([2, 3, 4]))
    assert ds.emit_jax() == "Not implemented"
    assert ds.emit_keras() == "Not implemented"
    assert ds.emit_mlx() == "Not implemented"
    assert ds.emit_pytorch() == "Not implemented"
    assert ds.emit_tensorflow() == "Not implemented"

    dus = DynamicUpdateSlice()
    assert dus.infer_shape(x, None, None) == (10,)
    u = np.array([99, 99])
    out = dus.eager_eval(x, u, [2])
    assert out[2] == 99
    assert out[3] == 99
    assert out[4] == 4
    assert dus.emit_jax() == "Not implemented"
    assert dus.emit_keras() == "Not implemented"
    assert dus.emit_mlx() == "Not implemented"
    assert dus.emit_pytorch() == "Not implemented"
    assert dus.emit_tensorflow() == "Not implemented"


def test_select() -> None:
    """Test select."""
    import numpy as np

    from ml_switcheroo_compiler.core.config import ConfigContext
    from ml_switcheroo_compiler.core.device import Device, DeviceType
    from ml_switcheroo_compiler.core.dtype import DType
    from ml_switcheroo_compiler.core.tensor import Tensor
    from ml_switcheroo_compiler.ops.shape.frontend import select

    device = Device(DeviceType.CPU)
    c = Tensor(np.array([True, False]), TensorConfig((2,), DType.Bool, device))
    x = Tensor(np.array([1, 2]), TensorConfig((2,), DType.Int32, device))
    y = Tensor(np.array([3, 4]), TensorConfig((2,), DType.Int32, device))

    with ConfigContext(eager_mode=True):
        out = select(c, x, y)
        assert np.array_equal(out.data, np.array([1, 4]))


def test_top_k_opdef() -> None:
    """Test top_k_opdef."""
    import numpy as np

    from ml_switcheroo_compiler.ops.shape import TopK

    tk = TopK()
    assert tk.infer_shape(None, 2) == ()

    class DummyShape:
        """Docstring."""

        shape = (10,)

    assert tk.infer_shape(DummyShape(), 2) == (2,)

    x = np.array([1, 5, 2, 8, 3])
    vals, idxs = tk.eager_eval(x, 2)
    assert np.array_equal(vals, [8, 5])
    assert np.array_equal(idxs, [3, 1])

    assert tk.emit_jax() == "Not implemented TopK"
    assert tk.emit_keras() == "Not implemented TopK"
    assert tk.emit_mlx() == "Not implemented TopK"
    assert tk.emit_pytorch() == "Not implemented TopK"
    assert tk.emit_tensorflow() == "Not implemented TopK"


def test_top_k_frontend() -> None:
    """Test top_k_frontend."""
    import numpy as np

    from ml_switcheroo_compiler.core.config import ConfigContext
    from ml_switcheroo_compiler.core.device import Device, DeviceType
    from ml_switcheroo_compiler.core.dtype import DType
    from ml_switcheroo_compiler.core.tensor import Tensor
    from ml_switcheroo_compiler.ops.shape.frontend import top_k
    from ml_switcheroo_compiler.tracing.tracer import ProxyTensor, _tracer

    device = Device(DeviceType.CPU)
    x = Tensor(np.array([1, 5, 2, 8, 3]), TensorConfig((5,), DType.Int32, device))

    with ConfigContext(eager_mode=True):
        out_val, out_idx = top_k(x, 2)
        assert out_val.shape == (2,)

    graph = _tracer.start_tracing("test_top_k")
    try:
        x_proxy = Tensor(
            ProxyTensor(id="x", shape=(5,), dtype="int32"), TensorConfig((5,), DType.Int32, device)
        )
        v, i = top_k(x_proxy, 2)
        assert v.shape == (2,)
        assert i.shape == (2,)
        node_v = graph.nodes[v.data.id]
        assert node_v.op_type == "TopK"
        assert node_v.attributes["k"] == 2
    finally:
        _tracer.stop_tracing()


def test_sort_opdef() -> None:
    """Test sort_opdef."""
    import numpy as np

    from ml_switcheroo_compiler.ops.shape import Sort

    s = Sort()
    assert s.infer_shape(None) == ()

    class DummyShape:
        """Docstring."""

        shape = (10,)

    assert s.infer_shape(DummyShape()) == (10,)

    x = np.array([3, 1, 2])
    out = s.eager_eval(x)
    assert np.array_equal(out, [1, 2, 3])

    assert s.emit_jax() == "Not implemented Sort"
    assert s.emit_keras() == "Not implemented Sort"
    assert s.emit_mlx() == "Not implemented Sort"
    assert s.emit_pytorch() == "Not implemented Sort"
    assert s.emit_tensorflow() == "Not implemented Sort"


def test_sort_frontend() -> None:
    """Test sort_frontend."""
    import numpy as np

    from ml_switcheroo_compiler.core.config import ConfigContext
    from ml_switcheroo_compiler.core.device import Device, DeviceType
    from ml_switcheroo_compiler.core.dtype import DType
    from ml_switcheroo_compiler.core.tensor import Tensor
    from ml_switcheroo_compiler.ops.shape.frontend import sort
    from ml_switcheroo_compiler.tracing.tracer import ProxyTensor, _tracer

    device = Device(DeviceType.CPU)
    x = Tensor(np.array([3, 1, 2]), TensorConfig((3,), DType.Int32, device))

    with ConfigContext(eager_mode=True):
        res = sort(x)
        np.testing.assert_array_equal(res.data, np.array([1, 2, 3]))

    graph = _tracer.start_tracing("test_sort")
    try:
        x_proxy = Tensor(
            ProxyTensor(id="x", shape=(3,), dtype="int32"), TensorConfig((3,), DType.Int32, device)
        )
        out = sort(x_proxy)
        assert out.shape == (3,)
        node = graph.nodes[out.data.id]
        assert node.op_type == "Sort"
        assert node.attributes["is_stable"] is True
    finally:
        _tracer.stop_tracing()


def test_broadcast_in_dim_opdef() -> None:
    """Test broadcast_in_dim_opdef."""
    import numpy as np

    from ml_switcheroo_compiler.ops.shape import BroadcastInDim

    op = BroadcastInDim()
    assert op.infer_shape(None, (2, 3), [1]) == (2, 3)

    x = np.array([1, 2, 3])
    out = op.eager_eval(x, (2, 3), [1])
    assert out.shape == (2, 3)
    assert np.array_equal(out[0], [1, 2, 3])
    assert np.array_equal(out[1], [1, 2, 3])

    assert op.emit_jax() == "Not implemented BroadcastInDim"
    assert op.emit_keras() == "Not implemented BroadcastInDim"
    assert op.emit_mlx() == "Not implemented BroadcastInDim"
    assert op.emit_pytorch() == "Not implemented BroadcastInDim"
    assert op.emit_tensorflow() == "Not implemented BroadcastInDim"


def test_broadcast_in_dim_frontend() -> None:
    """Test broadcast_in_dim_frontend."""
    import numpy as np

    from ml_switcheroo_compiler.core.config import ConfigContext
    from ml_switcheroo_compiler.core.device import Device, DeviceType
    from ml_switcheroo_compiler.core.dtype import DType
    from ml_switcheroo_compiler.core.tensor import Tensor
    from ml_switcheroo_compiler.ops.shape.frontend import broadcast_in_dim
    from ml_switcheroo_compiler.tracing.tracer import ProxyTensor, _tracer

    device = Device(DeviceType.CPU)
    x = Tensor(np.array([1, 2, 3]), TensorConfig((3,), DType.Int32, device))

    with ConfigContext(eager_mode=True):
        assert broadcast_in_dim(x, (2, 3), [1]).shape == (2, 3)

    graph = _tracer.start_tracing("test_broadcast")
    try:
        x_proxy = Tensor(
            ProxyTensor(id="x", shape=(3,), dtype="int32"), TensorConfig((3,), DType.Int32, device)
        )
        out = broadcast_in_dim(x_proxy, (2, 3), [1])
        assert out.shape == (2, 3)
        node = graph.nodes[out.data.id]
        assert node.op_type == "BroadcastInDim"
        assert node.attributes["shape"] == (2, 3)
        assert node.attributes["broadcast_dimensions"] == [1]
    finally:
        _tracer.stop_tracing()


def test_image_resize_opdef() -> None:
    """Test image_resize_opdef."""
    import numpy as np

    from ml_switcheroo_compiler.ops.shape import Resize

    op = Resize()
    assert op.infer_shape(None, (10, 10)) == ()

    class DummyShape:
        """Docstring."""

        shape = (1, 5, 5, 3)

    assert op.infer_shape(DummyShape(), (10, 10)) == (1, 10, 10, 3)

    x = np.ones((1, 5, 5, 3))
    out = op.eager_eval(x, (10, 10))
    assert out.shape == (1, 10, 10, 3)

    assert op.emit_jax() == "Not implemented Resize"
    assert op.emit_keras() == "Not implemented Resize"
    assert op.emit_mlx() == "Not implemented Resize"
    assert op.emit_pytorch() == "Not implemented Resize"
    assert op.emit_tensorflow() == "Not implemented Resize"


def test_image_resize_frontend() -> None:
    """Test image_resize_frontend."""
    import numpy as np

    from ml_switcheroo_compiler.core.config import ConfigContext
    from ml_switcheroo_compiler.core.device import Device, DeviceType
    from ml_switcheroo_compiler.core.dtype import DType
    from ml_switcheroo_compiler.core.tensor import Tensor
    from ml_switcheroo_compiler.ops.shape.frontend import image_resize
    from ml_switcheroo_compiler.tracing.tracer import ProxyTensor, _tracer

    device = Device(DeviceType.CPU)
    x = Tensor(np.ones((1, 5, 5, 3)), TensorConfig((1, 5, 5, 3), DType.Float32, device))

    with ConfigContext(eager_mode=True):
        assert image_resize(x, (10, 10)).shape == (1, 10, 10, 3)

    graph = _tracer.start_tracing("test_resize")
    try:
        x_proxy = Tensor(
            ProxyTensor(id="x", shape=(1, 5, 5, 3), dtype="float32"),
            TensorConfig((1, 5, 5, 3), DType.Float32, device),
        )
        out = image_resize(x_proxy, (10, 10))
        assert out.shape == (1, 10, 10, 3)
        node = graph.nodes[out.data.id]
        assert node.op_type == "Resize"
        assert node.attributes["shape"] == (10, 10)
        assert node.attributes["method"] == "bilinear"
    finally:
        _tracer.stop_tracing()


def test_shape_basic_coverage() -> None:
    """Test shape basic coverage."""
    import numpy as np

    from ml_switcheroo_compiler.ops.shape import Resize, TopK, Transpose

    # Test Transpose _format_args
    op1 = Transpose()
    assert op1._format_args("x", None) == "x"

    # Test TopK eager_eval with ndarray k
    op2 = TopK()
    res2_val, res2_idx = op2.eager_eval(np.array([1, 2, 3]), np.array(1))
    assert res2_val.shape == (1,)

    # Test Resize infer_shape with len < 3
    op3 = Resize()

    class DummyShape:
        """Docstring."""

        shape = (10, 10)

    assert op3.infer_shape(DummyShape(), (20, 20)) == (10, 10)


def test_shape_frontend_missing() -> None:
    """Test shape frontend missing coverage."""
    import numpy as np

    from ml_switcheroo_compiler.core.config import ConfigContext
    from ml_switcheroo_compiler.core.dtype import DType
    from ml_switcheroo_compiler.core.tensor import Tensor
    from ml_switcheroo_compiler.ops.shape.frontend import expand, gather, permute, repeat, squeeze
    from ml_switcheroo_compiler.tracing.tracer import ProxyTensor, _tracer

    device = "cpu"
    x = Tensor(np.ones((2, 1, 3)), TensorConfig((2, 1, 3), DType.Float32, device))
    x_proxy = Tensor(
        ProxyTensor("x", (2, 1, 3), "float32"), TensorConfig((2, 1, 3), DType.Float32, device)
    )

    with ConfigContext(eager_mode=False):
        _tracer.start_tracing("test")
        squeeze(x_proxy, dim=1)
        permute(x_proxy, (0, 2, 1))
        repeat(x_proxy, 2, dim=0)

        # top_k tracing with 0D tensor to hit out_shape false branch
        x_proxy_0d = Tensor(
            ProxyTensor("x0", (), "float32"), TensorConfig((), DType.Float32, device)
        )
        from ml_switcheroo_compiler.ops.shape.frontend import top_k

        top_k(x_proxy_0d, 1)
        _tracer.stop_tracing()

    with ConfigContext(eager_mode=True):
        # expand eager
        x_expand = Tensor(np.ones((2, 1)), TensorConfig((2, 1), DType.Float32, device))
        expand(x_expand, (2, 3))

        # repeat eager
        repeat(x_expand, 2, dim=0)

        # gather eager
        idx = Tensor(np.array([[[0]]]), TensorConfig((1, 1, 1), DType.Int32, device))
        gather(x, 2, idx)

        # searchsorted eager
        from ml_switcheroo_compiler.ops.shape.frontend import searchsorted

        x_1d = Tensor(np.array([1.0, 3.0]), TensorConfig((2,), DType.Float32, device))
        v = Tensor(np.array([2.0]), TensorConfig((1,), DType.Float32, device))
        searchsorted(x_1d, v)


def test_slicing_eager_strided() -> None:
    """Test eager slicing strided."""
    import numpy as np

    from ml_switcheroo_compiler.core.config import config
    from ml_switcheroo_compiler.core.device import Device
    from ml_switcheroo_compiler.core.tensor import Tensor
    from ml_switcheroo_compiler.ops.shape.slicing import strided_slice

    config.eager_mode = True
    data = np.array([1.0, 2.0])
    inp = Tensor(data, TensorConfig((2,), "float32", Device("cpu")))
    try:
        strided_slice(inp, [0], [1], [1])
    except Exception:
        pass
    config.eager_mode = False
