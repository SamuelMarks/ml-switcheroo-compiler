"""Tests for test_eager_utils_coverage."""

from __future__ import annotations

import typing

import numpy as np

import ml_switcheroo_compiler.backends.eager.utils as utils_mod
from ml_switcheroo_compiler.core.shape import TypedVariadicCallable
from tests.testing_utils import as_callable, as_ndarray, as_shape, as_tensor

if typing.TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import EagerCallableProtocol


class MockCpuResult:
    """Mock cpu container with numpy conversion."""

    def __init__(self, arr: np.ndarray[tuple[int, ...], np.dtype[np.generic]]) -> None:
        """Initialize container.

        Args:
            arr (np.ndarray): Underlying array.
        """
        self._arr: np.ndarray[tuple[int, ...], np.dtype[np.generic]] = arr

    def numpy(self) -> np.ndarray[tuple[int, ...], np.dtype[np.generic]]:
        """Convert to numpy array.

        Returns:
            np.ndarray: Array.
        """
        return self._arr


class MockDetachResult:
    """Mock detach result providing cpu method."""

    def __init__(self, arr: np.ndarray[tuple[int, ...], np.dtype[np.generic]]) -> None:
        """Initialize mock detach.

        Args:
            arr (np.ndarray): Underlying array.
        """
        self._arr: np.ndarray[tuple[int, ...], np.dtype[np.generic]] = arr

    def cpu(self) -> MockCpuResult:
        """Move to cpu.

        Returns:
            MockCpuResult: Cpu container.
        """
        return MockCpuResult(self._arr)


class MockTorchDetachOnly:
    """Mock object with detach method but no direct numpy attribute."""

    def __init__(self, arr: np.ndarray[tuple[int, ...], np.dtype[np.generic]]) -> None:
        """Initialize mock.

        Args:
            arr (np.ndarray): Underlying array.
        """
        self._arr: np.ndarray[tuple[int, ...], np.dtype[np.generic]] = arr

    def detach(self) -> MockDetachResult:
        """Detach tensor.

        Returns:
            MockDetachResult: Detached mock.
        """
        return MockDetachResult(self._arr)


class MockHasNumpyOnly:
    """Mock object with only a numpy method."""

    def __init__(self, arr: np.ndarray[tuple[int, ...], np.dtype[np.generic]]) -> None:
        """Initialize mock.

        Args:
            arr (np.ndarray): Underlying numpy array.
        """
        self._arr: np.ndarray[tuple[int, ...], np.dtype[np.generic]] = arr

    def numpy(self) -> np.ndarray[tuple[int, ...], np.dtype[np.generic]]:
        """Convert to numpy array.

        Returns:
            np.ndarray: Converted array.
        """
        return self._arr


class MockBackendModule:
    """Mock backend module supporting array operations."""

    def array(self, val: list[int] | np.ndarray[tuple[int, ...], np.dtype[np.generic]], dtype: str | None = None) -> np.ndarray[tuple[int, ...], np.dtype[np.generic]]:
        """Convert input to array.

        Args:
            val (list[int] | np.ndarray): Input value.
            dtype (str | None): Optional data type.

        Returns:
            np.ndarray: Converted array.
        """
        if dtype is not None:
            return np.array(val, dtype=dtype)
        return np.array(val)


class MockOriginalTensor:
    """Mock original tensor with a dtype attribute."""

    def __init__(self, dtype: str) -> None:
        """Initialize mock original tensor.

        Args:
            dtype (str): Target dtype.
        """
        self.dtype: str = dtype


class MockTorchStyleBackend:
    """Mock backend using dim and keepdim instead of axis and keepdims."""

    def mean(self, x: np.ndarray[tuple[int, ...], np.dtype[np.generic]], dim: tuple[int, ...], keepdim: bool = True) -> np.ndarray[tuple[int, ...], np.dtype[np.generic]]:
        """Compute mean across dimensions.

        Args:
            x (np.ndarray): Input array.
            dim (tuple[int, ...]): Target dimensions.
            keepdim (bool): Whether to keep dimensions.

        Returns:
            np.ndarray: Computed mean.
        """
        return as_ndarray(as_callable(np.mean)(x, axis=dim, keepdims=keepdim))

    def var(self, x: np.ndarray[tuple[int, ...], np.dtype[np.generic]], dim: tuple[int, ...], keepdim: bool = True, unbiased: bool = False) -> np.ndarray[tuple[int, ...], np.dtype[np.generic]]:
        """Compute variance across dimensions.

        Args:
            x (np.ndarray): Input array.
            dim (tuple[int, ...]): Target dimensions.
            keepdim (bool): Whether to keep dimensions.
            unbiased (bool): Whether to use unbiased estimator.

        Returns:
            np.ndarray: Computed variance.
        """
        ddof: int = 1 if unbiased else 0
        return as_ndarray(as_callable(np.var)(x, axis=dim, keepdims=keepdim, ddof=ddof))


class MockFailingAxisBackend:
    """Mock backend raising TypeError when axis is passed to mean or var."""

    def __init__(self, fallback: MockTorchStyleBackend) -> None:
        """Initialize mock.

        Args:
            fallback (MockTorchStyleBackend): Fallback implementation.
        """
        self.fallback: MockTorchStyleBackend = fallback

    def mean(self, x: np.ndarray[tuple[int, ...], np.dtype[np.generic]], *args: int, **kwargs: tuple[int, ...] | bool | int | str) -> np.ndarray[tuple[int, ...], np.dtype[np.generic]]:
        """Compute mean or raise TypeError if axis in kwargs.

        Args:
            x (np.ndarray): Input array.
            *args (int): Additional positional args.
            **kwargs (tuple[int, ...] | bool | int | str): Keyword arguments.

        Returns:
            np.ndarray: Computed mean.

        Raises:
            TypeError: When axis kwarg is present.
        """
        if "axis" in kwargs:
            raise TypeError("Use dim instead of axis")
        dim_val: tuple[int, ...] = as_shape(kwargs["dim"])
        keepdim_val: bool = bool(kwargs.get("keepdim", True))
        return self.fallback.mean(x, dim=dim_val, keepdim=keepdim_val)

    def var(self, x: np.ndarray[tuple[int, ...], np.dtype[np.generic]], *args: int, **kwargs: tuple[int, ...] | bool | int | str) -> np.ndarray[tuple[int, ...], np.dtype[np.generic]]:
        """Compute variance or raise TypeError if axis in kwargs.

        Args:
            x (np.ndarray): Input array.
            *args (int): Additional positional args.
            **kwargs (tuple[int, ...] | bool | int | str): Keyword arguments.

        Returns:
            np.ndarray: Computed variance.

        Raises:
            TypeError: When axis kwarg is present.
        """
        if "axis" in kwargs:
            raise TypeError("Use dim instead of axis")
        dim_val: tuple[int, ...] = as_shape(kwargs["dim"])
        keepdim_val: bool = bool(kwargs.get("keepdim", True))
        unbiased_val: bool = bool(kwargs.get("unbiased", False))
        return self.fallback.var(x, dim=dim_val, keepdim=keepdim_val, unbiased=unbiased_val)


class MockMathBackend:
    """Mock backend providing elementwise mathematical operations."""

    def add(self, a: np.ndarray[tuple[int, ...], np.dtype[np.generic]], b: float | np.ndarray[tuple[int, ...], np.dtype[np.generic]]) -> np.ndarray[tuple[int, ...], np.dtype[np.generic]]:
        """Add two arrays or array and scalar.

        Args:
            a (np.ndarray): First operand.
            b (float | np.ndarray): Second operand.

        Returns:
            np.ndarray: Elementwise addition.
        """
        return np.add(a, b)

    def subtract(self, a: np.ndarray[tuple[int, ...], np.dtype[np.generic]], b: float | np.ndarray[tuple[int, ...], np.dtype[np.generic]]) -> np.ndarray[tuple[int, ...], np.dtype[np.generic]]:
        """Subtract two arrays or array and scalar.

        Args:
            a (np.ndarray): First operand.
            b (float | np.ndarray): Second operand.

        Returns:
            np.ndarray: Elementwise subtraction.
        """
        return np.subtract(a, b)

    def multiply(self, a: np.ndarray[tuple[int, ...], np.dtype[np.generic]], b: float | np.ndarray[tuple[int, ...], np.dtype[np.generic]]) -> np.ndarray[tuple[int, ...], np.dtype[np.generic]]:
        """Multiply two arrays or array and scalar.

        Args:
            a (np.ndarray): First operand.
            b (float | np.ndarray): Second operand.

        Returns:
            np.ndarray: Elementwise product.
        """
        return np.multiply(a, b)

    def divide(self, a: np.ndarray[tuple[int, ...], np.dtype[np.generic]], b: float | np.ndarray[tuple[int, ...], np.dtype[np.generic]]) -> np.ndarray[tuple[int, ...], np.dtype[np.generic]]:
        """Divide two arrays or array and scalar.

        Args:
            a (np.ndarray): First operand.
            b (float | np.ndarray): Second operand.

        Returns:
            np.ndarray: Elementwise division.
        """
        return np.divide(a, b)

    def sqrt(self, a: np.ndarray[tuple[int, ...], np.dtype[np.generic]]) -> np.ndarray[tuple[int, ...], np.dtype[np.generic]]:
        """Compute square root.

        Args:
            a (np.ndarray): Input array.

        Returns:
            np.ndarray: Elementwise square root.
        """
        return np.sqrt(a)


class SignalBackendWrapper:
    """Signal execution backend ensuring reliable array padding."""

    def __getattr__(self, name: str) -> EagerCallableProtocol:
        """Delegate to numpy module.

        Args:
            name (str): Attribute name.

        Returns:
            EagerCallableProtocol: Resolved callable attribute.
        """
        return as_callable(getattr(np, name))

    def pad(
        self,
        arr: np.ndarray[tuple[int, ...], np.dtype[np.generic]],
        pad_width: tuple[tuple[int, int], tuple[int, int], tuple[int, int], tuple[int, int]],
        mode: str = "reflect",
    ) -> np.ndarray[tuple[int, ...], np.dtype[np.generic]]:
        """Pad 4D array without relying on ufunc initial sentinel.

        Args:
            arr (np.ndarray): 4D array to pad.
            pad_width (tuple[tuple[int, int], ...]): Padding dimensions.
            mode (str): Padding mode. Defaults to "reflect".

        Returns:
            np.ndarray: Padded array.
        """
        (b0, b1), (c0, c1), (h0, h1), (w0, w1) = pad_width
        b, c, h, w = arr.shape
        padded = np.zeros((b + b0 + b1, c + c0 + c1, h + h0 + h1, w + w0 + w1), dtype=arr.dtype)
        padded[b0 : b0 + b, c0 : c0 + c, h0 : h0 + h, w0 : w0 + w] = arr
        return padded


class MockRandModuleWithUniformAndRand:
    """Mock random module having uniform and rand."""

    def uniform(self, size: tuple[int, ...] = ()) -> np.ndarray[tuple[int, ...], np.dtype[np.generic]]:
        """Generate uniform random values.

        Args:
            size (tuple[int, ...]): Target size.

        Returns:
            np.ndarray: Result array.
        """
        return np.ones(size)

    def rand(self, *shape: int) -> np.ndarray[tuple[int, ...], np.dtype[np.generic]]:
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

    def __getattr__(self, name: str) -> TypedVariadicCallable[np.ndarray[tuple[int, ...], np.dtype[np.generic]]]:
        """Dynamically return callable only on second check.

        Args:
            name (str): Attribute name.

        Returns:
            TypedVariadicCallable[np.ndarray]: Function result.

        Raises:
            AttributeError: On first check.
        """
        count: int = self.calls.get(name, 0)
        self.calls[name] = count + 1
        if count == 0:
            raise AttributeError(f"Initial check fails for {name}")
        return lambda *args, **kwargs: np.ones((2, 2))


class MockBackendWithFallback:
    """Mock backend wrapping fallback random module."""

    def __init__(self) -> None:
        """Initialize backend."""
        self.random: MockRandFallbackModule = MockRandFallbackModule()


class MockSubRand:
    """Sub-module offering rand operation."""

    def rand(self, *shape: int) -> np.ndarray[tuple[int, ...], np.dtype[np.generic]]:
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

    def rand(self, *shape: int) -> np.ndarray[tuple[int, ...], np.dtype[np.generic]]:
        """Generate random array.

        Args:
            *shape (int): Dimensions.

        Returns:
            np.ndarray: Result array.
        """
        return np.ones(shape)


class MockBackendDirectUniform:
    """Mock backend having uniform directly."""

    def uniform(self, size: tuple[int, ...] = ()) -> np.ndarray[tuple[int, ...], np.dtype[np.generic]]:
        """Generate uniform array.

        Args:
            size (tuple[int, ...]): Dimensions.

        Returns:
            np.ndarray: Result array.
        """
        return np.ones(size)


class MockSubRandn:
    """Sub-module offering randn operation."""

    def randn(self, *shape: int) -> np.ndarray[tuple[int, ...], np.dtype[np.generic]]:
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

    def randn(self, *shape: int) -> np.ndarray[tuple[int, ...], np.dtype[np.generic]]:
        """Generate standard normal array.

        Args:
            *shape (int): Dimensions.

        Returns:
            np.ndarray: Result array.
        """
        return np.ones(shape)


class MockBackendDirectNormal:
    """Mock backend having normal directly."""

    def normal(self, size: tuple[int, ...] = ()) -> np.ndarray[tuple[int, ...], np.dtype[np.generic]]:
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
    res1 = as_callable(utils_mod._to_numpy_array)(np, obj_numpy, "other")
    as_callable(np.testing.assert_array_equal)(res1, arr)

    # 2. name == "torch" and hasattr detach
    obj_torch = MockTorchDetachOnly(arr)
    res2 = as_callable(utils_mod._to_numpy_array)(np, obj_torch, "torch")
    as_callable(np.testing.assert_array_equal)(res2, arr)

    # 3. name == "torch" but not hasattr detach
    res3 = utils_mod._to_numpy_array(np, [4.0, 5.0], "torch")
    as_callable(np.testing.assert_array_equal)(res3, as_callable(np.array)([4.0, 5.0]))

    # 4. Standard fallback to np_mod.asarray
    res4 = utils_mod._to_numpy_array(np, [7.0, 8.0], "numpy")
    as_callable(np.testing.assert_array_equal)(res4, as_callable(np.array)([7.0, 8.0]))


def test_utils_from_numpy_array() -> None:
    """Verify all branches of _from_numpy_array."""
    arr = np.array([1, 2, 3])
    orig = MockOriginalTensor("float32")
    backend = MockBackendModule()

    # 1. name == "torch"
    assert as_tensor(id(as_callable(utils_mod._from_numpy_array)(backend, arr, "torch", orig))) == id(arr)

    # 2. name == "mlx.core"
    assert as_tensor(id(as_callable(utils_mod._from_numpy_array)(backend, arr, "mlx.core", orig))) == id(arr)

    # 3. name == "jax.numpy"
    assert as_tensor(id(as_callable(utils_mod._from_numpy_array)(backend, arr, "jax.numpy", orig))) == id(arr)

    # 4. original_tensor is not None
    res_with_orig = as_callable(utils_mod._from_numpy_array)(backend, arr, "other", orig)
    assert isinstance(as_tensor(res_with_orig), np.ndarray)
    assert as_tensor(res_with_orig.dtype) == np.float32

    # 5. original_tensor is None
    res_no_orig = as_callable(utils_mod._from_numpy_array)(backend, arr, "other", None)
    assert isinstance(as_tensor(res_no_orig), np.ndarray)
    assert res_no_orig.dtype == np.int64

    # 6. backend_module without 'array' attribute
    assert utils_mod._from_numpy_array(None, arr, "other", None) is arr


def test_utils_channels_conversion() -> None:
    """Verify helper passthrough channel functions."""
    data = np.zeros((1, 3, 4, 4))
    assert utils_mod._to_channels_last(np, data, "NCHW") is data
    assert utils_mod._from_channels_last(np, data, "NCHW") is data
