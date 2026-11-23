"""Behavioral unit tests for random ops extended."""

from __future__ import annotations

import typing

import numpy as np

import ml_switcheroo_compiler.backends.eager.random_ops as rand_ops_mod
from ml_switcheroo_compiler.backends.numpy.eager import ensure_ndarray
from ml_switcheroo_compiler.core.shape import TypedVariadicCallable
from tests.testing_utils import GenericNDArray, MockTestStub, coerce_shape, extract_test_shape

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


class MockTorchDetachOnly:
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


class MockHasNumpyOnly:
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

    def mean(
        self,
        x: GenericNDArray,
        dim: tuple[int, ...],
        keepdim: bool = True,
    ) -> GenericNDArray:
        """Compute mean across dimensions.

        Args:
            x (GenericNDArray): Input array.
            dim (tuple[int, ...]): Target dimensions.
            keepdim (bool): Whether to keep dimensions.

        Returns:
            np.ndarray: Computed mean.
        """
        xf = x.astype(np.float64)
        return ensure_ndarray(
            np.mean(xf, axis=dim, keepdims=True) if keepdim else np.mean(xf, axis=dim, keepdims=False)
        )

    def var(
        self,
        x: GenericNDArray,
        dim: tuple[int, ...],
        keepdim: bool = True,
        unbiased: bool = False,
    ) -> GenericNDArray:
        """Compute variance across dimensions.

        Args:
            x (GenericNDArray): Input array.
            dim (tuple[int, ...]): Target dimensions.
            keepdim (bool): Whether to keep dimensions.
            unbiased (bool): Whether to use unbiased estimator.

        Returns:
            np.ndarray: Computed variance.
        """
        ddof: int = 1 if unbiased else 0
        xf = x.astype(np.float64)
        return ensure_ndarray(
            np.var(xf, axis=dim, keepdims=True, ddof=ddof)
            if keepdim
            else np.var(xf, axis=dim, keepdims=False, ddof=ddof)
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
        x: GenericNDArray,
        *args: int,
        **kwargs: tuple[int, ...] | bool | int | str,
    ) -> GenericNDArray:
        """Compute mean or raise TypeError if axis in kwargs.

        Args:
            x (GenericNDArray): Input array.
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
        x: GenericNDArray,
        *args: int,
        **kwargs: tuple[int, ...] | bool | int | str,
    ) -> GenericNDArray:
        """Compute variance or raise TypeError if axis in kwargs.

        Args:
            x (GenericNDArray): Input array.
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


class MockBackendWithFallback(MockTestStub):
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


class MockBackendWithRandOnly(MockTestStub):
    """Mock backend having only rand on random attribute."""

    def __init__(self) -> None:
        """Initialize backend with sub-module."""
        self.random: MockSubRand = MockSubRand()


class MockBackendDirectRand(MockTestStub):
    """Mock backend having rand directly."""

    def rand(self, *shape: int) -> GenericNDArray:
        """Generate random array.

        Args:
            *shape (int): Dimensions.

        Returns:
            np.ndarray: Result array.
        """
        return np.ones(shape)


class MockBackendDirectUniform(MockTestStub):
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


class MockBackendWithRandnOnly(MockTestStub):
    """Mock backend having only randn on random attribute."""

    def __init__(self) -> None:
        """Initialize backend with sub-module."""
        self.random: MockSubRandn = MockSubRandn()


class MockBackendDirectRandn(MockTestStub):
    """Mock backend having randn directly."""

    def randn(self, *shape: int) -> GenericNDArray:
        """Generate standard normal array.

        Args:
            *shape (int): Dimensions.

        Returns:
            np.ndarray: Result array.
        """
        return np.ones(shape)


class MockBackendDirectNormal(MockTestStub):
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


def test_random_ops_prng_and_split() -> None:
    """Verify PRNGKey, RandomSplit, and RandomFoldIn branches."""
    backend_with_array = MockBackendModule()
    backend_without_array = 123  # does not have array attribute

    # 1. prng_key
    k1 = rand_ops_mod.prng_key(backend_with_array, 42)
    assert isinstance(k1, np.ndarray)
    np.testing.assert_array_equal(k1, np.array([0, 42], dtype=np.uint32))

    k2 = rand_ops_mod.prng_key(backend_without_array, 42)
    assert k2 == [0, 42]

    # 2. random_split
    s1 = rand_ops_mod.random_split(backend_with_array, k1, 3)
    assert isinstance(s1, np.ndarray)
    assert s1.shape == (3, 2)

    s2 = rand_ops_mod.random_split(backend_without_array, k2, 2)
    assert isinstance(s2, list)
    assert len(s2) == 2

    # 3. random_fold_in with indexed key
    f1 = rand_ops_mod.random_fold_in(backend_with_array, [10, 20], 5)
    assert isinstance(f1, np.ndarray)
    np.testing.assert_array_equal(f1, np.array([15, 20], dtype=np.uint32))

    # random_fold_in with key lacking __getitem__
    f2 = rand_ops_mod.random_fold_in(backend_without_array, 999, 7)
    assert f2 == [7, 0]


def test_random_ops_rand_and_randn() -> None:
    """Verify rand, randn, and randint under various backend capabilities."""
    # 1. rand with shape in kwargs as nested list
    res_rand1 = rand_ops_mod.rand(np, shape=[(2, 3)])
    assert extract_test_shape(res_rand1) == (2, 3)

    # rand with args shape
    res_rand2 = rand_ops_mod.rand(np, 2, 4)
    assert extract_test_shape(res_rand2) == (2, 4)

    # rand with random module without uniform (only rand)
    res_rand3 = rand_ops_mod.rand(MockBackendWithRandOnly(), 3, 3)
    assert extract_test_shape(res_rand3) == (3, 3)

    # rand with direct rand
    assert extract_test_shape(rand_ops_mod.rand(MockBackendDirectRand(), 2, 2)) == (2, 2)

    # rand with direct uniform
    assert extract_test_shape(rand_ops_mod.rand(MockBackendDirectUniform(), 2, 2)) == (2, 2)

    # rand fallback line 93
    fb_backend = MockBackendWithFallback()
    assert extract_test_shape(rand_ops_mod.rand(fb_backend, 2, 2)) == (2, 2)

    # 2. randn tests
    res_randn1 = rand_ops_mod.randn(np, shape=[(3, 2)])
    assert extract_test_shape(res_randn1) == (3, 2)

    res_randn2 = rand_ops_mod.randn(np, 3, 5)
    assert extract_test_shape(res_randn2) == (3, 5)

    # randn with random module having only randn
    assert extract_test_shape(rand_ops_mod.randn(MockBackendWithRandnOnly(), 2, 2)) == (
        2,
        2,
    )

    # randn with direct randn
    assert extract_test_shape(rand_ops_mod.randn(MockBackendDirectRandn(), 2, 2)) == (2, 2)

    # randn with direct normal
    assert extract_test_shape(rand_ops_mod.randn(MockBackendDirectNormal(), 2, 2)) == (2, 2)

    # randn fallback line 122
    fb_backend_normal = MockBackendWithFallback()
    assert extract_test_shape(rand_ops_mod.randn(fb_backend_normal, 2, 2)) == (2, 2)

    # 3. randint tests
    res_randint1 = rand_ops_mod.randint(np, 0, 10, (2, 3))
    assert extract_test_shape(res_randint1) == (2, 3)

    res_randint2 = rand_ops_mod.randint(np, low=2, high=8, shape=(4,))
    assert extract_test_shape(res_randint2) == (4,)

    res_randint3 = rand_ops_mod.randint(np)
    assert extract_test_shape(res_randint3) == ()

    # randint fallback line 143
    fb_backend_randint = MockBackendWithFallback()
    assert extract_test_shape(rand_ops_mod.randint(fb_backend_randint, 0, 5, (2, 2))) == (
        2,
        2,
    )
