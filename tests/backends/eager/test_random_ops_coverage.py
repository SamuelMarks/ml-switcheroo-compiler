"""Tests for test_random_ops_coverage."""

from __future__ import annotations

import typing

import numpy as np

import ml_switcheroo_compiler.backends.eager.random_ops as rand_ops_mod
from ml_switcheroo_compiler.backends.eager_registry import EagerValue
from ml_switcheroo_compiler.core.shape import TypedVariadicCallable

if typing.TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import EagerCallableProtocol, EagerTensorProtocol


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
        return typing.cast("np.ndarray[tuple[int, ...], np.dtype[np.generic]]", typing.cast("EagerCallableProtocol", np.mean)(x, axis=dim, keepdims=keepdim))

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
        return typing.cast("np.ndarray[tuple[int, ...], np.dtype[np.generic]]", typing.cast("EagerCallableProtocol", np.var)(x, axis=dim, keepdims=keepdim, ddof=ddof))


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
        dim_val: tuple[int, ...] = typing.cast("tuple[int, ...]", kwargs["dim"])
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
        dim_val: tuple[int, ...] = typing.cast("tuple[int, ...]", kwargs["dim"])
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

    def __getattr__(self, name: str) -> TypedVariadicCallable[np.ndarray[tuple[int, ...], np.dtype[np.generic]]]:
        """Delegate to numpy module.

        Args:
            name (str): Attribute name.

        Returns:
            TypedVariadicCallable[np.ndarray]: Resolved callable attribute.
        """
        return typing.cast("TypedVariadicCallable[np.ndarray[tuple[int, ...], np.dtype[np.generic]]]", getattr(np, name))

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


def test_random_ops_prng_and_split() -> None:
    """Verify PRNGKey, RandomSplit, and RandomFoldIn branches."""
    backend_with_array = MockBackendModule()
    backend_without_array = 123  # does not have array attribute

    # 1. prng_key
    k1 = rand_ops_mod.prng_key(typing.cast("EagerValue", backend_with_array), 42)
    assert isinstance(k1, np.ndarray)
    np.testing.assert_array_equal(k1, np.array([0, 42], dtype=np.uint32))

    k2 = rand_ops_mod.prng_key(backend_without_array, 42)
    assert k2 == [0, 42]

    # 2. random_split
    s1 = rand_ops_mod.random_split(typing.cast("EagerValue", backend_with_array), k1, 3)
    assert isinstance(s1, np.ndarray)
    assert s1.shape == (3, 2)

    s2 = rand_ops_mod.random_split(backend_without_array, k2, 2)
    assert isinstance(s2, list)
    assert len(s2) == 2

    # 3. random_fold_in with indexed key
    f1 = rand_ops_mod.random_fold_in(typing.cast("EagerValue", backend_with_array), [10, 20], 5)
    assert isinstance(f1, np.ndarray)
    np.testing.assert_array_equal(f1, np.array([15, 20], dtype=np.uint32))

    # random_fold_in with key lacking __getitem__
    f2 = rand_ops_mod.random_fold_in(backend_without_array, 999, 7)
    assert f2 == [7, 0]


def test_random_ops_rand_and_randn() -> None:
    """Verify rand, randn, and randint under various backend capabilities."""
    # 1. rand with shape in kwargs as nested list
    res_rand1 = rand_ops_mod.rand(np, shape=[(2, 3)])
    assert typing.cast("EagerTensorProtocol", getattr(res_rand1, "shape")) == (2, 3)

    # rand with args shape
    res_rand2 = rand_ops_mod.rand(np, 2, 4)
    assert typing.cast("EagerTensorProtocol", getattr(res_rand2, "shape")) == (2, 4)

    # rand with random module without uniform (only rand)
    res_rand3 = rand_ops_mod.rand(typing.cast("EagerValue", MockBackendWithRandOnly()), 3, 3)
    assert typing.cast("EagerTensorProtocol", getattr(res_rand3, "shape")) == (3, 3)

    # rand with direct rand
    assert typing.cast("EagerTensorProtocol", getattr(rand_ops_mod.rand(typing.cast("EagerValue", MockBackendDirectRand()), 2, 2), "shape")) == (2, 2)

    # rand with direct uniform
    assert typing.cast("EagerTensorProtocol", getattr(rand_ops_mod.rand(typing.cast("EagerValue", MockBackendDirectUniform()), 2, 2), "shape")) == (2, 2)

    # rand fallback line 93
    fb_backend = MockBackendWithFallback()
    assert typing.cast("EagerTensorProtocol", getattr(rand_ops_mod.rand(typing.cast("EagerValue", fb_backend), 2, 2), "shape")) == (2, 2)

    # 2. randn tests
    res_randn1 = rand_ops_mod.randn(np, shape=[(3, 2)])
    assert typing.cast("EagerTensorProtocol", getattr(res_randn1, "shape")) == (3, 2)

    res_randn2 = rand_ops_mod.randn(np, 3, 5)
    assert typing.cast("EagerTensorProtocol", getattr(res_randn2, "shape")) == (3, 5)

    # randn with random module having only randn
    assert typing.cast("EagerTensorProtocol", getattr(rand_ops_mod.randn(typing.cast("EagerValue", MockBackendWithRandnOnly()), 2, 2), "shape")) == (2, 2)

    # randn with direct randn
    assert typing.cast("EagerTensorProtocol", getattr(rand_ops_mod.randn(typing.cast("EagerValue", MockBackendDirectRandn()), 2, 2), "shape")) == (2, 2)

    # randn with direct normal
    assert typing.cast("EagerTensorProtocol", getattr(rand_ops_mod.randn(typing.cast("EagerValue", MockBackendDirectNormal()), 2, 2), "shape")) == (2, 2)

    # randn fallback line 122
    fb_backend_normal = MockBackendWithFallback()
    assert typing.cast("EagerTensorProtocol", getattr(rand_ops_mod.randn(typing.cast("EagerValue", fb_backend_normal), 2, 2), "shape")) == (2, 2)

    # 3. randint tests
    res_randint1 = rand_ops_mod.randint(np, 0, 10, (2, 3))
    assert typing.cast("EagerTensorProtocol", getattr(res_randint1, "shape")) == (2, 3)

    res_randint2 = rand_ops_mod.randint(np, low=2, high=8, shape=(4,))
    assert typing.cast("EagerTensorProtocol", getattr(res_randint2, "shape")) == (4,)

    res_randint3 = rand_ops_mod.randint(np)
    assert typing.cast("EagerTensorProtocol", getattr(res_randint3, "shape")) == ()

    # randint fallback line 143
    fb_backend_randint = MockBackendWithFallback()
    assert typing.cast("EagerTensorProtocol", getattr(rand_ops_mod.randint(typing.cast("EagerValue", fb_backend_randint), 0, 5, (2, 2)), "shape")) == (2, 2)
