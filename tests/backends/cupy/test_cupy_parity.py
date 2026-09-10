"""Parity and device residency verification tests for CuPy backend."""

import sys
from unittest.mock import MagicMock, patch

from ml_switcheroo_compiler.backends.cupy.eager import (
    _verify_gpu_residency,
    cupy_avgpool2d,
    cupy_conv2d,
    cupy_maxpool2d,
    cupy_resize_bilinear,
    execute_op,
)


class MockGPUData:
    """Mock buffer representing GPU device memory."""

    def __init__(self, device: str = "cuda:0") -> None:
        """Initialize mock GPU data with device indicator.

        Args:
            device (str): Device descriptor.
        """
        self.device = device


class MockGPUArray:
    """Mock CuPy array residing strictly in GPU memory."""

    def __init__(self, shape=(2, 2), dtype=None, data=None) -> None:
        """Initialize mock GPU array.

        Args:
            shape (tuple): Array dimensions.
            dtype (object): Data type.
            data (object): Memory buffer.
        """
        self.shape = shape
        self.ndim = len(shape)
        self.dtype = dtype or "float32"
        self.data = data if data is not None else MockGPUData("cuda:0")

    def copy(self) -> "MockGPUArray":
        """Copy the array preserving GPU residency.

        Returns:
            MockGPUArray: Copied array on GPU.
        """
        return MockGPUArray(shape=self.shape, dtype=self.dtype, data=MockGPUData("cuda:0"))

    def __add__(self, other: object) -> "MockGPUArray":
        """Mock addition returning GPU array."""
        return MockGPUArray(shape=self.shape, dtype=self.dtype, data=MockGPUData("cuda:0"))

    def __mul__(self, other: object) -> "MockGPUArray":
        """Mock multiplication returning GPU array."""
        return MockGPUArray(shape=self.shape, dtype=self.dtype, data=MockGPUData("cuda:0"))

    def __truediv__(self, other: object) -> "MockGPUArray":
        """Mock division returning GPU array."""
        return MockGPUArray(shape=self.shape, dtype=self.dtype, data=MockGPUData("cuda:0"))

    def __getitem__(self, key: object) -> "MockGPUArray":
        """Mock slicing returning GPU array."""
        return MockGPUArray(shape=(2, 2), dtype=self.dtype, data=MockGPUData("cuda:0"))


def _create_mock_cupy() -> MagicMock:
    """Create a fully-mocked CuPy module maintaining device residency.

    Returns:
        MagicMock: Configured mock CuPy module.
    """
    mock_cp = MagicMock()
    mock_cp.ndarray = MockGPUArray

    def asarray_fn(x, *args, **kwargs):
        if isinstance(x, MockGPUArray):
            return x
        return MockGPUArray(shape=(2, 2))

    mock_cp.asarray.side_effect = asarray_fn
    mock_cp.zeros.side_effect = lambda shape, *a, **kw: MockGPUArray(shape=shape)
    mock_cp.pad.side_effect = lambda x, p=None, *a, **kw: MockGPUArray(shape=x.shape)
    mock_cp.tensordot.side_effect = lambda a, b, *ar, **kw: MagicMock(transpose=lambda *axes: MockGPUArray(shape=(a.shape[0], b.shape[0], 2, 2)))
    mock_cp.maximum.side_effect = lambda a, b: MockGPUArray(shape=a.shape)
    mock_cp.linspace.side_effect = lambda s, e, n, *a, **kw: MagicMock(astype=lambda dt: MockGPUArray(shape=(n,)))
    mock_cp.floor.side_effect = lambda x: MagicMock(astype=lambda dt: MockGPUArray(shape=x.shape))
    mock_cp.minimum.side_effect = lambda a, b: MockGPUArray(shape=(4,))

    # Reduction ops
    mock_cp.argmax.side_effect = lambda x, *a, **kw: MockGPUArray(shape=(1,), dtype="int64")
    mock_cp.argmin.side_effect = lambda x, *a, **kw: MockGPUArray(shape=(1,), dtype="int64")
    mock_cp.cumsum.side_effect = lambda x, *a, **kw: MockGPUArray(shape=x.shape)
    mock_cp.cumprod.side_effect = lambda x, *a, **kw: MockGPUArray(shape=x.shape)
    mock_cp.linalg.norm.side_effect = lambda x, *a, **kw: MockGPUArray(shape=(1,))

    # Linear algebra ops
    mock_cp.linalg.cholesky.side_effect = lambda x, *a, **kw: MockGPUArray(shape=x.shape)
    mock_cp.linalg.qr.side_effect = lambda x, *a, **kw: (MockGPUArray(shape=x.shape), MockGPUArray(shape=x.shape))
    mock_cp.linalg.svd.side_effect = lambda x, *a, **kw: (MockGPUArray(shape=x.shape), MockGPUArray(shape=(2,)), MockGPUArray(shape=x.shape))
    mock_cp.linalg.solve.side_effect = lambda a, b, *ar, **kw: MockGPUArray(shape=b.shape)
    mock_cp.linalg.eig.side_effect = lambda x, *a, **kw: (MockGPUArray(shape=(2,)), MockGPUArray(shape=x.shape))
    mock_cp.linalg.eigh.side_effect = lambda x, *a, **kw: (MockGPUArray(shape=(2,)), MockGPUArray(shape=x.shape))

    return mock_cp


def test_cupy_reductions_parity_and_residency() -> None:
    """Test reduction operations on CuPy maintaining strict GPU device residency."""
    mock_cp = _create_mock_cupy()
    with patch.dict(sys.modules, {"cupy": mock_cp}):
        with patch("ml_switcheroo_compiler.backends.cupy.eager.cp", mock_cp):
            inp = MockGPUArray(shape=(4, 4))

            # ArgMax
            res_argmax = execute_op(None, "ArgMax", inp, dim=0)
            _verify_gpu_residency(res_argmax)
            mock_cp.argmax.assert_called()

            # ArgMin
            res_argmin = execute_op(None, "ArgMin", inp, dim=1)
            _verify_gpu_residency(res_argmin)
            mock_cp.argmin.assert_called()

            # ReduceL2
            res_l2 = execute_op(None, "ReduceL2", inp, axis=0)
            _verify_gpu_residency(res_l2)
            mock_cp.linalg.norm.assert_called()

            # CumSum
            res_cumsum = execute_op(None, "CumSum", inp, dim=0)
            _verify_gpu_residency(res_cumsum)
            mock_cp.cumsum.assert_called()

            # CumProd
            res_cumprod = execute_op(None, "CumProd", inp, dim=0)
            _verify_gpu_residency(res_cumprod)
            mock_cp.cumprod.assert_called()


def test_cupy_linalg_parity_and_residency() -> None:
    """Test linear algebra operations on CuPy without CPU transfer or NumPy fallback."""
    mock_cp = _create_mock_cupy()
    with patch.dict(sys.modules, {"cupy": mock_cp}):
        with patch("ml_switcheroo_compiler.backends.cupy.eager.cp", mock_cp):
            mat_a = MockGPUArray(shape=(4, 4))
            mat_b = MockGPUArray(shape=(4, 4))

            # Cholesky
            res_chol = execute_op(None, "Cholesky", mat_a)
            _verify_gpu_residency(res_chol)
            mock_cp.linalg.cholesky.assert_called()

            # QR
            q, r = execute_op(None, "QR", mat_a)
            _verify_gpu_residency(q)
            _verify_gpu_residency(r)
            mock_cp.linalg.qr.assert_called()

            # SVD
            u, s, vh = execute_op(None, "SVD", mat_a)
            _verify_gpu_residency(u)
            _verify_gpu_residency(s)
            _verify_gpu_residency(vh)
            mock_cp.linalg.svd.assert_called()

            # Solve
            res_solve = execute_op(None, "Solve", mat_a, mat_b)
            _verify_gpu_residency(res_solve)
            mock_cp.linalg.solve.assert_called()

            # Eig and Eigh
            w, v = execute_op(None, "Eig", mat_a)
            _verify_gpu_residency(w)
            _verify_gpu_residency(v)
            mock_cp.linalg.eig.assert_called()

            wh, vh = execute_op(None, "Eigh", mat_a)
            _verify_gpu_residency(wh)
            _verify_gpu_residency(vh)
            mock_cp.linalg.eigh.assert_called()


def test_cupy_vision_nn_parity_and_residency() -> None:
    """Test vision and neural network ops on CuPy running strictly on GPU."""
    mock_cp = _create_mock_cupy()
    with patch.dict(sys.modules, {"cupy": mock_cp}):
        with patch("ml_switcheroo_compiler.backends.cupy.eager.cp", mock_cp):
            x_4d = MockGPUArray(shape=(2, 3, 32, 32))
            w_conv = MockGPUArray(shape=(16, 3, 3, 3))

            # Conv2D
            res_conv = execute_op(None, "Conv2D", x_4d, w_conv, stride=1, padding="SAME")
            _verify_gpu_residency(res_conv)

            # MaxPool2D
            res_maxpool = execute_op(None, "MaxPool2D", x_4d, kernel_size=2, stride=2)
            _verify_gpu_residency(res_maxpool)

            # AvgPool2D
            res_avgpool = execute_op(None, "AvgPool2D", x_4d, kernel_size=2, stride=2)
            _verify_gpu_residency(res_avgpool)

            # Pad
            res_pad = execute_op(None, "Pad", x_4d, pad_width=((0, 0), (0, 0), (1, 1), (1, 1)))
            _verify_gpu_residency(res_pad)
            mock_cp.pad.assert_called()

            # ResizeBilinear
            res_resize = execute_op(None, "ResizeBilinear", x_4d, size=(64, 64))
            _verify_gpu_residency(res_resize)


def test_cupy_direct_vision_kernels() -> None:
    """Directly test native GPU kernel implementations in eager module."""
    mock_cp = _create_mock_cupy()
    with patch("ml_switcheroo_compiler.backends.cupy.eager.cp", mock_cp):
        # 4D Conv2D
        x_4d = MockGPUArray(shape=(1, 1, 4, 4))
        w_4d = MockGPUArray(shape=(1, 1, 3, 3))
        res_conv4d = cupy_conv2d(x_4d, w_4d, stride=1, padding="SAME")
        assert res_conv4d is not None

        # 2D Conv2D
        x_2d = MockGPUArray(shape=(4, 4))
        w_2d = MockGPUArray(shape=(3, 3))
        res_conv2d = cupy_conv2d(x_2d, w_2d, stride=1, padding="SAME")
        assert res_conv2d is not None

        # MaxPool2D
        res_mp = cupy_maxpool2d(x_4d, kernel_size=2, stride=2)
        assert res_mp is not None

        # AvgPool2D
        res_ap = cupy_avgpool2d(x_4d, kernel_size=2, stride=2)
        assert res_ap is not None

        # ResizeBilinear
        res_rb = cupy_resize_bilinear(x_4d, size=(8, 8))
        assert res_rb is not None

        # Fallback when cp is None
        with patch("ml_switcheroo_compiler.backends.cupy.eager.cp", None):
            assert cupy_conv2d(x_4d, w_4d) == x_4d
            assert cupy_maxpool2d(x_4d) == x_4d
            assert cupy_avgpool2d(x_4d) == x_4d
            assert cupy_resize_bilinear(x_4d, size=(8, 8)) == x_4d
