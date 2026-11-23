"""Behavioral unit tests for utils extended."""

from __future__ import annotations

import typing

import numpy as np

import ml_switcheroo_compiler.backends.eager.utils as utils_mod
from ml_switcheroo_compiler.backends.numpy.eager import ensure_ndarray
from ml_switcheroo_compiler.core.shape import TypedVariadicCallable
from tests.testing_utils import GenericNDArray, MockTestStub, coerce_shape

if typing.TYPE_CHECKING:
    pass


class MockCpuResult:
    """Mock cpu container with numpy conversion."""

    def __init__(
        self,
        arr: GenericNDArray,
    ) -> None:
        """Initialize container.

        Args:
            arr (GenericNDArray): Underlying array.
        """
        self._arr: GenericNDArray = arr

    def numpy(
        self,
    ) -> GenericNDArray:
        """Convert to numpy array.

        Returns:
            np.ndarray: Array.
        """
        return self._arr


class MockDetachResult:
    """Mock detach result providing cpu method."""

    def __init__(
        self,
        arr: GenericNDArray,
    ) -> None:
        """Initialize mock detach.

        Args:
            arr (GenericNDArray): Underlying array.
        """
        self._arr: GenericNDArray = arr

    def cpu(self) -> MockCpuResult:
        """Move to cpu.

        Returns:
            MockCpuResult: Cpu container.
        """
        return MockCpuResult(self._arr)


class MockTorchDetachOnly(MockTestStub):
    """Mock object with detach method but no direct numpy attribute."""

    def __init__(
        self,
        arr: GenericNDArray,
    ) -> None:
        """Initialize mock.

        Args:
            arr (GenericNDArray): Underlying array.
        """
        self._arr: GenericNDArray = arr

    def detach(self) -> MockDetachResult:
        """Detach tensor.

        Returns:
            MockDetachResult: Detached mock.
        """
        return MockDetachResult(self._arr)


class MockHasNumpyOnly(MockTestStub):
    """Mock object with only a numpy method."""

    def __init__(
        self,
        arr: GenericNDArray,
    ) -> None:
        """Initialize mock.

        Args:
            arr (GenericNDArray): Underlying numpy array.
        """
        self._arr: GenericNDArray = arr

    def numpy(
        self,
    ) -> GenericNDArray:
        """Convert to numpy array.

        Returns:
            np.ndarray: Converted array.
        """
        return self._arr


class MockBackendModule(MockTestStub):
    """Mock backend module supporting array operations."""

    def array(
        self,
        val: list[int] | GenericNDArray,
        dtype: str | None = None,
    ) -> GenericNDArray:
        """Convert input to array.

        Args:
            val (list[int] | GenericNDArray): Input value.
            dtype (str | None): Optional data type.

        Returns:
            np.ndarray: Converted array.
        """
        if dtype is not None:
            return np.array(val, dtype=dtype)
        return np.array(val)


class MockOriginalTensor(MockTestStub):
    """Mock original tensor with a dtype attribute."""

    def __init__(self, dtype: str) -> None:
        """Initialize mock original tensor.

        Args:
            dtype (str): Target dtype.
        """
        self.dtype: str = dtype


class MockTorchStyleBackend:
    """Mock backend using dim and keepdim instead of axis and keepdims."""

    def mean(
        self, x: np.ndarray[tuple[int, ...], np.dtype[np.float64]], dim: tuple[int, ...], keepdim: bool = True
    ) -> GenericNDArray:
        """Compute mean across dimensions.

        Args:
            x (np.ndarray[tuple[int, ...], np.dtype[np.float64]]): Input array.
            dim (tuple[int, ...]): Target dimensions.
            keepdim (bool): Whether to keep dimensions.

        Returns:
            np.ndarray: Computed mean.
        """
        return ensure_ndarray(np.mean(x, axis=dim, keepdims=True) if keepdim else np.mean(x, axis=dim, keepdims=False))

    def var(
        self,
        x: np.ndarray[tuple[int, ...], np.dtype[np.float64]],
        dim: tuple[int, ...],
        keepdim: bool = True,
        unbiased: bool = False,
    ) -> GenericNDArray:
        """Compute variance across dimensions.

        Args:
            x (np.ndarray[tuple[int, ...], np.dtype[np.float64]]): Input array.
            dim (tuple[int, ...]): Target dimensions.
            keepdim (bool): Whether to keep dimensions.
            unbiased (bool): Whether to use unbiased estimator.

        Returns:
            np.ndarray: Computed variance.
        """
        ddof: int = 1 if unbiased else 0
        return ensure_ndarray(
            np.var(x, axis=dim, keepdims=True, ddof=ddof) if keepdim else np.var(x, axis=dim, keepdims=False, ddof=ddof)
        )


class MockFailingAxisBackend:
    """Mock backend raising TypeError when axis is passed to mean or var."""

    def __init__(self, fallback: MockTorchStyleBackend) -> None:
        """Initialize mock.

        Args:
            fallback (MockTorchStyleBackend): Fallback implementation.
        """
        self.fallback: MockTorchStyleBackend = fallback

    def mean(
        self,
        x: np.ndarray[tuple[int, ...], np.dtype[np.float64]],
        *args: int,
        **kwargs: tuple[int, ...] | bool | int | str,
    ) -> GenericNDArray:
        """Compute mean or raise TypeError if axis in kwargs.

        Args:
            x (np.ndarray[tuple[int, ...], np.dtype[np.float64]]): Input array.
            *args (int): Additional positional args.
            **kwargs (tuple[int, ...] | bool | int | str): Keyword arguments.

        Returns:
            np.ndarray: Computed mean.

        Raises:
            TypeError: When axis kwarg is present.
        """
        if "axis" in kwargs:
            raise TypeError("Use dim instead of axis")
        dim_val: tuple[int, ...] = coerce_shape(kwargs["dim"])
        keepdim_val: bool = bool(kwargs.get("keepdim", True))
        return self.fallback.mean(x, dim=dim_val, keepdim=keepdim_val)

    def var(
        self,
        x: np.ndarray[tuple[int, ...], np.dtype[np.float64]],
        *args: int,
        **kwargs: tuple[int, ...] | bool | int | str,
    ) -> GenericNDArray:
        """Compute variance or raise TypeError if axis in kwargs.

        Args:
            x (np.ndarray[tuple[int, ...], np.dtype[np.float64]]): Input array.
            *args (int): Additional positional args.
            **kwargs (tuple[int, ...] | bool | int | str): Keyword arguments.

        Returns:
            np.ndarray: Computed variance.

        Raises:
            TypeError: When axis kwarg is present.
        """
        if "axis" in kwargs:
            raise TypeError("Use dim instead of axis")
        dim_val: tuple[int, ...] = coerce_shape(kwargs["dim"])
        keepdim_val: bool = bool(kwargs.get("keepdim", True))
        unbiased_val: bool = bool(kwargs.get("unbiased", False))
        return self.fallback.var(x, dim=dim_val, keepdim=keepdim_val, unbiased=unbiased_val)


class MockMathBackend:
    """Mock backend providing elementwise mathematical operations."""

    def add(
        self,
        a: GenericNDArray,
        b: float | GenericNDArray,
    ) -> GenericNDArray:
        """Add two arrays or array and scalar.

        Args:
            a (GenericNDArray): First operand.
            b (float | GenericNDArray): Second operand.

        Returns:
            np.ndarray: Elementwise addition.
        """
        return np.add(a, b)

    def subtract(
        self,
        a: GenericNDArray,
        b: float | GenericNDArray,
    ) -> GenericNDArray:
        """Subtract two arrays or array and scalar.

        Args:
            a (GenericNDArray): First operand.
            b (float | GenericNDArray): Second operand.

        Returns:
            np.ndarray: Elementwise subtraction.
        """
        return np.subtract(a, b)

    def multiply(
        self,
        a: GenericNDArray,
        b: float | GenericNDArray,
    ) -> GenericNDArray:
        """Multiply two arrays or array and scalar.

        Args:
            a (GenericNDArray): First operand.
            b (float | GenericNDArray): Second operand.

        Returns:
            np.ndarray: Elementwise product.
        """
        return np.multiply(a, b)

    def divide(
        self,
        a: GenericNDArray,
        b: float | GenericNDArray,
    ) -> GenericNDArray:
        """Divide two arrays or array and scalar.

        Args:
            a (GenericNDArray): First operand.
            b (float | GenericNDArray): Second operand.

        Returns:
            np.ndarray: Elementwise division.
        """
        return np.divide(a, b)

    def sqrt(
        self,
        a: GenericNDArray,
    ) -> GenericNDArray:
        """Compute square root.

        Args:
            a (GenericNDArray): Input array.

        Returns:
            np.ndarray: Elementwise square root.
        """
        return np.sqrt(a)


class SignalBackendWrapper:
    """Signal execution backend ensuring reliable array padding."""

    arange = staticmethod(np.arange)
    meshgrid = staticmethod(np.meshgrid)
    exp = staticmethod(np.exp)
    sum = staticmethod(np.sum)
    zeros_like = staticmethod(np.zeros_like)
    expand_dims = staticmethod(np.expand_dims)
    reshape = staticmethod(np.reshape)
    median = staticmethod(np.median)

    def pad(
        self,
        arr: GenericNDArray,
        pad_width: tuple[tuple[int, int], tuple[int, int], tuple[int, int], tuple[int, int]],
        mode: str = "reflect",
    ) -> GenericNDArray:
        """Pad 4D array without relying on ufunc initial sentinel.

        Args:
            arr (GenericNDArray): 4D array to pad.
            pad_width (tuple[tuple[int, int], tuple[int, int], tuple[int, int], tuple[int, int]]): Padding dimensions.
            mode (str): Padding mode. Defaults to "reflect".

        Returns:
            np.ndarray: Padded array.
        """
        _ = mode
        (b0, b1), (c0, c1), (h0, h1), (w0, w1) = pad_width
        b, c, h, w = arr.shape
        out_shape = (b + b0 + b1, c + c0 + c1, h + h0 + h1, w + w0 + w1)
        padded: GenericNDArray = (
            np.zeros(out_shape, dtype=np.float32)
            if arr.dtype == np.dtype(np.float32)
            else np.zeros(out_shape, dtype=np.float64)
        )
        padded[b0 : b0 + b, c0 : c0 + c, h0 : h0 + h, w0 : w0 + w] = arr
        return padded


class MockRandModuleWithUniformAndRand:
    """Mock random module having uniform and rand."""

    def uniform(self, size: tuple[int, ...] = ()) -> GenericNDArray:
        """Generate uniform random values.

        Args:
            size (tuple[int, ...]): Target size.

        Returns:
            np.ndarray: Result array.
        """
        return np.ones(size)

    def rand(self, *shape: int) -> GenericNDArray:
        """Generate random values.

        Args:
            *shape (int): Dimensions.

        Returns:
            np.ndarray: Result array.
        """
        return np.ones(shape)


class MockRandFallbackModule:
    """Mock random module triggering fallback at line 93, 122, and 143."""

    def __init__(self) -> None:
        """Initialize mock module."""
        self.calls: dict[str, int] = {}

    def _get_fn(self, name: str) -> TypedVariadicCallable[GenericNDArray]:
        """Return callable only on second check.

        Args:
            name (str): Input `name` value (name) for `_get_fn`.

        Returns:
            TypedVariadicCallable[GenericNDArray]: Computed output value from
                `_get_fn`.

        Raises:
            AttributeError: If execution or validation fails in `_get_fn`.
        """
        count: int = self.calls.get(name, 0)
        self.calls[name] = count + 1
        if count == 0:
            raise AttributeError(f"Initial check fails for {name}")
        return lambda *args, **kwargs: np.ones((2, 2))

    @property
    def uniform(
        self,
    ) -> TypedVariadicCallable[GenericNDArray]:
        """Dynamic uniform attribute.

        Returns:
            TypedVariadicCallable[GenericNDArray]: Computed output value from
                `uniform`.
        """
        return self._get_fn("uniform")

    @property
    def normal(
        self,
    ) -> TypedVariadicCallable[GenericNDArray]:
        """Dynamic normal attribute.

        Returns:
            TypedVariadicCallable[GenericNDArray]: Computed output value from
                `normal`.
        """
        return self._get_fn("normal")

    @property
    def randint(
        self,
    ) -> TypedVariadicCallable[GenericNDArray]:
        """Dynamic randint attribute.

        Returns:
            TypedVariadicCallable[GenericNDArray]: Computed output value from
                `randint`.
        """
        return self._get_fn("randint")


class MockBackendWithFallback:
    """Mock backend wrapping fallback random module."""

    def __init__(self) -> None:
        """Initialize backend."""
        self.random: MockRandFallbackModule = MockRandFallbackModule()


class MockSubRand:
    """Sub-module offering rand operation."""

    def rand(self, *shape: int) -> GenericNDArray:
        """Generate random values.

        Args:
            *shape (int): Dimensions.

        Returns:
            np.ndarray: Result array.
        """
        return np.zeros(shape)


class MockBackendWithRandOnly:
    """Mock backend having only rand on random attribute."""

    def __init__(self) -> None:
        """Initialize backend with sub-module."""
        self.random: MockSubRand = MockSubRand()


class MockBackendDirectRand:
    """Mock backend having rand directly."""

    def rand(self, *shape: int) -> GenericNDArray:
        """Generate random array.

        Args:
            *shape (int): Dimensions.

        Returns:
            np.ndarray: Result array.
        """
        return np.ones(shape)


class MockBackendDirectUniform:
    """Mock backend having uniform directly."""

    def uniform(self, size: tuple[int, ...] = ()) -> GenericNDArray:
        """Generate uniform array.

        Args:
            size (tuple[int, ...]): Dimensions.

        Returns:
            np.ndarray: Result array.
        """
        return np.ones(size)


class MockSubRandn:
    """Sub-module offering randn operation."""

    def randn(self, *shape: int) -> GenericNDArray:
        """Generate standard normal array.

        Args:
            *shape (int): Dimensions.

        Returns:
            np.ndarray: Result array.
        """
        return np.zeros(shape)


class MockBackendWithRandnOnly:
    """Mock backend having only randn on random attribute."""

    def __init__(self) -> None:
        """Initialize backend with sub-module."""
        self.random: MockSubRandn = MockSubRandn()


class MockBackendDirectRandn:
    """Mock backend having randn directly."""

    def randn(self, *shape: int) -> GenericNDArray:
        """Generate standard normal array.

        Args:
            *shape (int): Dimensions.

        Returns:
            np.ndarray: Result array.
        """
        return np.ones(shape)


class MockBackendDirectNormal:
    """Mock backend having normal directly."""

    def normal(self, size: tuple[int, ...] = ()) -> GenericNDArray:
        """Generate normal array.

        Args:
            size (tuple[int, ...]): Dimensions.

        Returns:
            np.ndarray: Result array.
        """
        return np.ones(size)


class MockStringConvertibleGroup:
    """Mock group operand convertible via str without __int__."""

    def __init__(self, val: str) -> None:
        """Initialize mock.

        Args:
            val (str): String representation.
        """
        self._val: str = val

    def __str__(self) -> str:
        """Return string representation.

        Returns:
            str: Value.
        """
        return self._val


def test_utils_to_numpy_array() -> None:
    """Verify all branches of _to_numpy_array."""
    arr = np.array([1.0, 2.0, 3.0])

    # 1. Hasattr numpy
    obj_numpy = MockHasNumpyOnly(arr)
    res1 = utils_mod._to_numpy_array(np, obj_numpy, "other")
    np.testing.assert_array_equal(ensure_ndarray(res1), arr)

    # 2. name == "torch" and hasattr detach
    obj_torch = MockTorchDetachOnly(arr)
    res2 = utils_mod._to_numpy_array(np, obj_torch, "torch")
    np.testing.assert_array_equal(ensure_ndarray(res2), arr)

    # 3. name == "torch" but not hasattr detach
    res3 = utils_mod._to_numpy_array(np, [4.0, 5.0], "torch")
    np.testing.assert_array_equal(ensure_ndarray(res3), np.array([4.0, 5.0]))

    # 4. Standard fallback to np_mod.asarray
    res4 = utils_mod._to_numpy_array(np, [7.0, 8.0], "numpy")
    np.testing.assert_array_equal(ensure_ndarray(res4), np.array([7.0, 8.0]))


def test_utils_from_numpy_array() -> None:
    """Verify all branches of _from_numpy_array."""
    arr = np.array([1, 2, 3])
    orig = MockOriginalTensor("float32")
    backend = MockBackendModule()

    # 1. name == "torch"
    assert id(utils_mod._from_numpy_array(backend, arr, "torch", orig)) == id(arr)

    # 2. name == "mlx.core"
    assert id(utils_mod._from_numpy_array(backend, arr, "mlx.core", orig)) == id(arr)

    # 3. name == "jax.numpy"
    assert id(utils_mod._from_numpy_array(backend, arr, "jax.numpy", orig)) == id(arr)

    # 4. original_tensor is not None
    res_with_orig = utils_mod._from_numpy_array(backend, arr, "other", orig)
    assert isinstance(res_with_orig, np.ndarray)
    assert res_with_orig.dtype == np.float32

    # 5. original_tensor is None
    res_no_orig = utils_mod._from_numpy_array(backend, arr, "other", None)
    assert isinstance(res_no_orig, np.ndarray)
    assert res_no_orig.dtype == np.int64

    # 6. backend_module without 'array' attribute
    assert utils_mod._from_numpy_array(None, arr, "other", None) is arr


def test_utils_channels_conversion() -> None:
    """Verify helper passthrough channel functions."""
    data = np.zeros((1, 3, 4, 4))
    assert utils_mod._to_channels_last(np, data, "NCHW") is data
    assert utils_mod._from_channels_last(np, data, "NCHW") is data
