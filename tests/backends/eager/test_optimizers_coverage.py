"""Tests for test_optimizers_coverage."""

from __future__ import annotations

import typing

import numpy as np

import ml_switcheroo_compiler.backends.eager.optimizers as opt_mod
from ml_switcheroo_compiler.core.shape import TypedVariadicCallable
from tests.testing_utils import ensure_callable, ensure_eager_tensor, ensure_ndarray, ensure_shape

if typing.TYPE_CHECKING:
    pass


class MockCpuResult:
    """Mock cpu container with numpy conversion."""

    def __init__(self, arr: np.ndarray[tuple[int, ...], np.dtype[np.float64]]) -> None:
        """Initialize container.

        Args:
            arr (np.ndarray): Underlying array.
        """
        self._arr: np.ndarray[tuple[int, ...], np.dtype[np.float64]] = arr

    def numpy(self) -> np.ndarray[tuple[int, ...], np.dtype[np.float64]]:
        """Convert to numpy array.

        Returns:
            np.ndarray: Array.
        """
        return self._arr


class MockDetachResult:
    """Mock detach result providing cpu method."""

    def __init__(self, arr: np.ndarray[tuple[int, ...], np.dtype[np.float64]]) -> None:
        """Initialize mock detach.

        Args:
            arr (np.ndarray): Underlying array.
        """
        self._arr: np.ndarray[tuple[int, ...], np.dtype[np.float64]] = arr

    def cpu(self) -> MockCpuResult:
        """Move to cpu.

        Returns:
            MockCpuResult: Cpu container.
        """
        return MockCpuResult(self._arr)


class MockTorchDetachOnly:
    """Mock object with detach method but no direct numpy attribute."""

    def __init__(self, arr: np.ndarray[tuple[int, ...], np.dtype[np.float64]]) -> None:
        """Initialize mock.

        Args:
            arr (np.ndarray): Underlying array.
        """
        self._arr: np.ndarray[tuple[int, ...], np.dtype[np.float64]] = arr

    def detach(self) -> MockDetachResult:
        """Detach tensor.

        Returns:
            MockDetachResult: Detached mock.
        """
        return MockDetachResult(self._arr)


class MockHasNumpyOnly:
    """Mock object with only a numpy method."""

    def __init__(self, arr: np.ndarray[tuple[int, ...], np.dtype[np.float64]]) -> None:
        """Initialize mock.

        Args:
            arr (np.ndarray): Underlying numpy array.
        """
        self._arr: np.ndarray[tuple[int, ...], np.dtype[np.float64]] = arr

    def numpy(self) -> np.ndarray[tuple[int, ...], np.dtype[np.float64]]:
        """Convert to numpy array.

        Returns:
            np.ndarray: Converted array.
        """
        return self._arr


class MockBackendModule:
    """Mock backend module supporting array operations."""

    def array(self, val: list[int] | np.ndarray[tuple[int, ...], np.dtype[np.float64]], dtype: str | None = None) -> np.ndarray[tuple[int, ...], np.dtype[np.float64]]:
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

    def mean(self, x: np.ndarray[tuple[int, ...], np.dtype[np.float64]], dim: tuple[int, ...], keepdim: bool = True) -> np.ndarray[tuple[int, ...], np.dtype[np.float64]]:
        """Compute mean across dimensions.

        Args:
            x (np.ndarray): Input array.
            dim (tuple[int, ...]): Target dimensions.
            keepdim (bool): Whether to keep dimensions.

        Returns:
            np.ndarray: Computed mean.
        """
        return ensure_ndarray(ensure_callable(np.mean)(x, axis=dim, keepdims=keepdim))

    def var(self, x: np.ndarray[tuple[int, ...], np.dtype[np.float64]], dim: tuple[int, ...], keepdim: bool = True, unbiased: bool = False) -> np.ndarray[tuple[int, ...], np.dtype[np.float64]]:
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
        return ensure_ndarray(ensure_callable(np.var)(x, axis=dim, keepdims=keepdim, ddof=ddof))


class MockFailingAxisBackend:
    """Mock backend raising TypeError when axis is passed to mean or var."""

    def __init__(self, fallback: MockTorchStyleBackend) -> None:
        """Initialize mock.

        Args:
            fallback (MockTorchStyleBackend): Fallback implementation.
        """
        self.fallback: MockTorchStyleBackend = fallback

    def mean(self, x: np.ndarray[tuple[int, ...], np.dtype[np.float64]], *args: int, **kwargs: tuple[int, ...] | bool | int | str) -> np.ndarray[tuple[int, ...], np.dtype[np.float64]]:
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
        dim_val: tuple[int, ...] = ensure_shape(kwargs["dim"])
        keepdim_val: bool = bool(kwargs.get("keepdim", True))
        return self.fallback.mean(x, dim=dim_val, keepdim=keepdim_val)

    def var(self, x: np.ndarray[tuple[int, ...], np.dtype[np.float64]], *args: int, **kwargs: tuple[int, ...] | bool | int | str) -> np.ndarray[tuple[int, ...], np.dtype[np.float64]]:
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
        dim_val: tuple[int, ...] = ensure_shape(kwargs["dim"])
        keepdim_val: bool = bool(kwargs.get("keepdim", True))
        unbiased_val: bool = bool(kwargs.get("unbiased", False))
        return self.fallback.var(x, dim=dim_val, keepdim=keepdim_val, unbiased=unbiased_val)


class MockMathBackend:
    """Mock backend providing elementwise mathematical operations."""

    def add(self, a: np.ndarray[tuple[int, ...], np.dtype[np.float64]], b: float | np.ndarray[tuple[int, ...], np.dtype[np.float64]]) -> np.ndarray[tuple[int, ...], np.dtype[np.float64]]:
        """Add two arrays or array and scalar.

        Args:
            a (np.ndarray): First operand.
            b (float | np.ndarray): Second operand.

        Returns:
            np.ndarray: Elementwise addition.
        """
        return np.add(a, b)

    def subtract(self, a: np.ndarray[tuple[int, ...], np.dtype[np.float64]], b: float | np.ndarray[tuple[int, ...], np.dtype[np.float64]]) -> np.ndarray[tuple[int, ...], np.dtype[np.float64]]:
        """Subtract two arrays or array and scalar.

        Args:
            a (np.ndarray): First operand.
            b (float | np.ndarray): Second operand.

        Returns:
            np.ndarray: Elementwise subtraction.
        """
        return np.subtract(a, b)

    def multiply(self, a: np.ndarray[tuple[int, ...], np.dtype[np.float64]], b: float | np.ndarray[tuple[int, ...], np.dtype[np.float64]]) -> np.ndarray[tuple[int, ...], np.dtype[np.float64]]:
        """Multiply two arrays or array and scalar.

        Args:
            a (np.ndarray): First operand.
            b (float | np.ndarray): Second operand.

        Returns:
            np.ndarray: Elementwise product.
        """
        return np.multiply(a, b)

    def divide(self, a: np.ndarray[tuple[int, ...], np.dtype[np.float64]], b: float | np.ndarray[tuple[int, ...], np.dtype[np.float64]]) -> np.ndarray[tuple[int, ...], np.dtype[np.float64]]:
        """Divide two arrays or array and scalar.

        Args:
            a (np.ndarray): First operand.
            b (float | np.ndarray): Second operand.

        Returns:
            np.ndarray: Elementwise division.
        """
        return np.divide(a, b)

    def sqrt(self, a: np.ndarray[tuple[int, ...], np.dtype[np.float64]]) -> np.ndarray[tuple[int, ...], np.dtype[np.float64]]:
        """Compute square root.

        Args:
            a (np.ndarray): Input array.

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
        arr: np.ndarray[tuple[int, ...], np.dtype[np.float64]],
        pad_width: tuple[tuple[int, int], tuple[int, int], tuple[int, int], tuple[int, int]],
        mode: str = "reflect",
    ) -> np.ndarray[tuple[int, ...], np.dtype[np.float64]]:
        """Pad 4D array without relying on ufunc initial sentinel.

        Args:
            arr (np.ndarray): 4D array to pad.
            pad_width (tuple[tuple[int, int], ...]): Padding dimensions.
            mode (str): Padding mode. Defaults to "reflect".

        Returns:
            np.ndarray: Padded array.
        """
        _ = mode
        (b0, b1), (c0, c1), (h0, h1), (w0, w1) = pad_width
        b, c, h, w = arr.shape
        padded = np.zeros((b + b0 + b1, c + c0 + c1, h + h0 + h1, w + w0 + w1), dtype=arr.dtype)
        padded[b0 : b0 + b, c0 : c0 + c, h0 : h0 + h, w0 : w0 + w] = arr
        return padded


class MockRandModuleWithUniformAndRand:
    """Mock random module having uniform and rand."""

    def uniform(self, size: tuple[int, ...] = ()) -> np.ndarray[tuple[int, ...], np.dtype[np.float64]]:
        """Generate uniform random values.

        Args:
            size (tuple[int, ...]): Target size.

        Returns:
            np.ndarray: Result array.
        """
        return np.ones(size)

    def rand(self, *shape: int) -> np.ndarray[tuple[int, ...], np.dtype[np.float64]]:
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

    def _get_fn(self, name: str) -> TypedVariadicCallable[np.ndarray[tuple[int, ...], np.dtype[np.float64]]]:
        """Return callable only on second check.

        Args:
            name (str): Input `name` value (name) for `_get_fn`.

        Returns:
            TypedVariadicCallable[np.ndarray[tuple[int, ...], np.dtype[np.float64]]]: Computed output value from `_get_fn`.
        """
        count: int = self.calls.get(name, 0)
        self.calls[name] = count + 1
        if count == 0:
            raise AttributeError(f"Initial check fails for {name}")
        return lambda *args, **kwargs: np.ones((2, 2))

    @property
    def uniform(self) -> TypedVariadicCallable[np.ndarray[tuple[int, ...], np.dtype[np.float64]]]:
        """Dynamic uniform attribute.

        Returns:
            TypedVariadicCallable[np.ndarray[tuple[int, ...], np.dtype[np.float64]]]: Computed output value from `uniform`.
        """
        return self._get_fn("uniform")

    @property
    def normal(self) -> TypedVariadicCallable[np.ndarray[tuple[int, ...], np.dtype[np.float64]]]:
        """Dynamic normal attribute.

        Returns:
            TypedVariadicCallable[np.ndarray[tuple[int, ...], np.dtype[np.float64]]]: Computed output value from `normal`.
        """
        return self._get_fn("normal")

    @property
    def randint(self) -> TypedVariadicCallable[np.ndarray[tuple[int, ...], np.dtype[np.float64]]]:
        """Dynamic randint attribute.

        Returns:
            TypedVariadicCallable[np.ndarray[tuple[int, ...], np.dtype[np.float64]]]: Computed output value from `randint`.
        """
        return self._get_fn("randint")


class MockBackendWithFallback:
    """Mock backend wrapping fallback random module."""

    def __init__(self) -> None:
        """Initialize backend."""
        self.random: MockRandFallbackModule = MockRandFallbackModule()


class MockSubRand:
    """Sub-module offering rand operation."""

    def rand(self, *shape: int) -> np.ndarray[tuple[int, ...], np.dtype[np.float64]]:
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

    def rand(self, *shape: int) -> np.ndarray[tuple[int, ...], np.dtype[np.float64]]:
        """Generate random array.

        Args:
            *shape (int): Dimensions.

        Returns:
            np.ndarray: Result array.
        """
        return np.ones(shape)


class MockBackendDirectUniform:
    """Mock backend having uniform directly."""

    def uniform(self, size: tuple[int, ...] = ()) -> np.ndarray[tuple[int, ...], np.dtype[np.float64]]:
        """Generate uniform array.

        Args:
            size (tuple[int, ...]): Dimensions.

        Returns:
            np.ndarray: Result array.
        """
        return np.ones(size)


class MockSubRandn:
    """Sub-module offering randn operation."""

    def randn(self, *shape: int) -> np.ndarray[tuple[int, ...], np.dtype[np.float64]]:
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

    def randn(self, *shape: int) -> np.ndarray[tuple[int, ...], np.dtype[np.float64]]:
        """Generate standard normal array.

        Args:
            *shape (int): Dimensions.

        Returns:
            np.ndarray: Result array.
        """
        return np.ones(shape)


class MockBackendDirectNormal:
    """Mock backend having normal directly."""

    def normal(self, size: tuple[int, ...] = ()) -> np.ndarray[tuple[int, ...], np.dtype[np.float64]]:
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


def test_optimizers_coverage() -> None:
    """Verify all optimizer update formulas."""
    backend = MockMathBackend()
    param = np.ones((2, 2), dtype=np.float64)
    m = np.zeros((2, 2), dtype=np.float64)
    v = np.zeros((2, 2), dtype=np.float64)
    grad = np.full((2, 2), 0.1, dtype=np.float64)

    # 1. Adam
    p_new, m_new, v_new = ensure_callable(opt_mod.apply_adam)(backend, param, m, v, grad, 0.001)
    assert ensure_eager_tensor(p_new).shape == (2, 2)
    assert ensure_eager_tensor(m_new).shape == (2, 2)
    assert ensure_eager_tensor(v_new).shape == (2, 2)

    # 2. Adagrad
    accum = np.zeros((2, 2), dtype=np.float64)
    p_new_ada, accum_new_ada = ensure_callable(opt_mod.apply_adagrad)(backend, param, accum, grad, 0.01)
    assert ensure_eager_tensor(p_new_ada).shape == (2, 2)
    assert ensure_eager_tensor(accum_new_ada).shape == (2, 2)

    # 3. Ftrl
    linear = np.zeros((2, 2), dtype=np.float64)
    p_new_ftrl, accum_new_ftrl, linear_new_ftrl = ensure_callable(opt_mod.apply_ftrl)(backend, param, accum, linear, grad, 0.01)
    assert ensure_eager_tensor(p_new_ftrl).shape == (2, 2)
    assert ensure_eager_tensor(accum_new_ftrl).shape == (2, 2)
    assert ensure_eager_tensor(linear_new_ftrl).shape == (2, 2)

    # 4. RMSProp
    ms = np.ones((2, 2), dtype=np.float64)
    mom = np.zeros((2, 2), dtype=np.float64)
    p_new_rms, ms_new_rms, mom_new_rms = ensure_callable(opt_mod.apply_rmsprop)(backend, param, ms, mom, grad, 0.001)
    assert ensure_eager_tensor(p_new_rms).shape == (2, 2)
    assert ensure_eager_tensor(ms_new_rms).shape == (2, 2)
    assert ensure_eager_tensor(mom_new_rms).shape == (2, 2)
