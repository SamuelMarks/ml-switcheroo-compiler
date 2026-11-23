"""Behavioral unit tests for core group ops."""

from __future__ import annotations

import typing

import numpy as np
import pytest

import ml_switcheroo_compiler.backends.eager.core_group_ops as group_ops_mod
from ml_switcheroo_compiler.backends.numpy.eager import ensure_ndarray
from ml_switcheroo_compiler.core.shape import TypedVariadicCallable
from tests.testing_utils import MockTestStub, NumpyTestArray, coerce_shape, extract_test_shape

if typing.TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import BackendArg


def test_invoke_grouped_op_coverage() -> None:
    """Test function test_invoke_grouped_op_coverage."""
    from ml_switcheroo_compiler.backends.eager.core_group_ops import _invoke_grouped_op

    class TorchMock(MockTestStub):
        """Test class TorchMock."""

        def mean(self, x: BackendArg, **kwargs: BackendArg) -> BackendArg:
            """Test function mean.

            Args:
                x: Test parameter `x`.
                kwargs: Test parameter `kwargs`.

            Returns:
                BackendArg: Computed output value from `mean`.

            Raises:
                TypeError: If an argument passed to `mean` has an incompatible type.
            """
            if "axis" in kwargs:
                raise TypeError()
            return "torch_mean"

        def var(self, x: BackendArg, **kwargs: BackendArg) -> BackendArg:
            """Test function var.

            Args:
                x: Test parameter `x`.
                kwargs: Test parameter `kwargs`.

            Returns:
                BackendArg: Computed output value from `var`.

            Raises:
                TypeError: If an argument passed to `var` has an incompatible type.
            """
            if "axis" in kwargs:
                raise TypeError()
            return "torch_var"

    assert _invoke_grouped_op(TorchMock(), "mean", None, ()) == "torch_mean"
    assert _invoke_grouped_op(TorchMock(), "variance", None, ()) == "torch_var"

    import pytest

    with pytest.raises(ValueError):
        _invoke_grouped_op(TorchMock(), "unknown", None, ())


def test_group_norm_positive_axis_coverage() -> None:
    """Test function test_group_norm_positive_axis_coverage."""
    import numpy as np

    from ml_switcheroo_compiler.backends.eager.core_group_ops import _group_norm

    x = np.random.randn(2, 4, 4, 8).astype(np.float32)
    out = _group_norm(np, x, 2, axis=3, epsilon=1e-05)
    reshaped = x.reshape(2, 4, 4, 2, 4)
    mean = np.mean(reshaped, axis=(1, 2, 4), keepdims=True)
    var = np.var(reshaped, axis=(1, 2, 4), keepdims=True)
    expected = ((reshaped - mean) / np.sqrt(var + 1e-05)).reshape(x.shape)
    np.testing.assert_allclose(ensure_ndarray(out), expected, rtol=1e-05, atol=1e-05)


class MockCpuResult:
    """Mock cpu container with numpy conversion."""

    def __init__(self, arr: NumpyTestArray) -> None:
        """Initialize container.

        Args:
            arr (NumpyTestArray): Underlying array.
        """
        self._arr: NumpyTestArray = arr

    def numpy(self) -> NumpyTestArray:
        """Convert to numpy array.

        Returns:
            NumpyTestArray: Array.
        """
        return self._arr


class MockDetachResult:
    """Mock detach result providing cpu method."""

    def __init__(self, arr: NumpyTestArray) -> None:
        """Initialize mock detach.

        Args:
            arr (NumpyTestArray): Underlying array.
        """
        self._arr: NumpyTestArray = arr

    def cpu(self) -> MockCpuResult:
        """Move to cpu.

        Returns:
            MockCpuResult: Cpu container.
        """
        return MockCpuResult(self._arr)


class MockTorchDetachOnly:
    """Mock object with detach method but no direct numpy attribute."""

    def __init__(self, arr: NumpyTestArray) -> None:
        """Initialize mock.

        Args:
            arr (NumpyTestArray): Underlying array.
        """
        self._arr: NumpyTestArray = arr

    def detach(self) -> MockDetachResult:
        """Detach tensor.

        Returns:
            MockDetachResult: Detached mock.
        """
        return MockDetachResult(self._arr)


class MockHasNumpyOnly:
    """Mock object with only a numpy method."""

    def __init__(self, arr: NumpyTestArray) -> None:
        """Initialize mock.

        Args:
            arr (NumpyTestArray): Underlying numpy array.
        """
        self._arr: NumpyTestArray = arr

    def numpy(self) -> NumpyTestArray:
        """Convert to numpy array.

        Returns:
            NumpyTestArray: Converted array.
        """
        return self._arr


class MockBackendModule:
    """Mock backend module supporting array operations."""

    def array(
        self,
        val: list[int] | NumpyTestArray,
        dtype: str | None = None,
    ) -> NumpyTestArray:
        """Convert input to array.

        Args:
            val (list[int] | NumpyTestArray): Input value.
            dtype (str | None): Optional data type.

        Returns:
            NumpyTestArray: Converted array.
        """
        if dtype is not None:
            return ensure_ndarray(np.array(val, dtype=dtype))
        return ensure_ndarray(np.array(val))


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
        x: NumpyTestArray,
        dim: tuple[int, ...],
        keepdim: bool = True,
    ) -> NumpyTestArray:
        """Compute mean across dimensions.

        Args:
            x (NumpyTestArray): Input array.
            dim (tuple[int, ...]): Target dimensions.
            keepdim (bool): Whether to keep dimensions.

        Returns:
            NumpyTestArray: Computed mean.
        """
        arr_f = np.asarray(x, dtype=np.float64)
        return ensure_ndarray(
            np.mean(arr_f, axis=dim, keepdims=True) if keepdim else np.mean(arr_f, axis=dim, keepdims=False)
        )

    def var(
        self,
        x: NumpyTestArray,
        dim: tuple[int, ...],
        keepdim: bool = True,
        unbiased: bool = False,
    ) -> NumpyTestArray:
        """Compute variance across dimensions.

        Args:
            x (NumpyTestArray): Input array.
            dim (tuple[int, ...]): Target dimensions.
            keepdim (bool): Whether to keep dimensions.
            unbiased (bool): Whether to use unbiased estimator.

        Returns:
            NumpyTestArray: Computed variance.
        """
        ddof: int = 1 if unbiased else 0
        arr_f = np.asarray(x, dtype=np.float64)
        return ensure_ndarray(
            np.var(arr_f, axis=dim, keepdims=True, ddof=ddof)
            if keepdim
            else np.var(arr_f, axis=dim, keepdims=False, ddof=ddof)
        )


class MockFailingAxisBackend(MockTestStub):
    """Mock backend raising TypeError when axis is passed to mean or var."""

    def __init__(self, fallback: MockTorchStyleBackend) -> None:
        """Initialize mock.

        Args:
            fallback (MockTorchStyleBackend): Fallback implementation.
        """
        self.fallback: MockTorchStyleBackend = fallback

    def mean(
        self,
        x: NumpyTestArray,
        *args: int,
        **kwargs: tuple[int, ...] | bool | int | str,
    ) -> NumpyTestArray:
        """Compute mean or raise TypeError if axis in kwargs.

        Args:
            x (NumpyTestArray): Input array.
            *args (int): Additional positional args.
            **kwargs (tuple[int, ...] | bool | int | str): Keyword arguments.

        Returns:
            NumpyTestArray: Computed mean.

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
        x: NumpyTestArray,
        *args: int,
        **kwargs: tuple[int, ...] | bool | int | str,
    ) -> NumpyTestArray:
        """Compute variance or raise TypeError if axis in kwargs.

        Args:
            x (NumpyTestArray): Input array.
            *args (int): Additional positional args.
            **kwargs (tuple[int, ...] | bool | int | str): Keyword arguments.

        Returns:
            NumpyTestArray: Computed variance.

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
        a: NumpyTestArray,
        b: float | NumpyTestArray,
    ) -> NumpyTestArray:
        """Add two arrays or array and scalar.

        Args:
            a (NumpyTestArray): First operand.
            b (float | NumpyTestArray): Second operand.

        Returns:
            NumpyTestArray: Elementwise addition.
        """
        return ensure_ndarray(np.add(a, b))

    def subtract(
        self,
        a: NumpyTestArray,
        b: float | NumpyTestArray,
    ) -> NumpyTestArray:
        """Subtract two arrays or array and scalar.

        Args:
            a (NumpyTestArray): First operand.
            b (float | NumpyTestArray): Second operand.

        Returns:
            NumpyTestArray: Elementwise subtraction.
        """
        return ensure_ndarray(np.subtract(a, b))

    def multiply(
        self,
        a: NumpyTestArray,
        b: float | NumpyTestArray,
    ) -> NumpyTestArray:
        """Multiply two arrays or array and scalar.

        Args:
            a (NumpyTestArray): First operand.
            b (float | NumpyTestArray): Second operand.

        Returns:
            NumpyTestArray: Elementwise product.
        """
        return ensure_ndarray(np.multiply(a, b))

    def divide(
        self,
        a: NumpyTestArray,
        b: float | NumpyTestArray,
    ) -> NumpyTestArray:
        """Divide two arrays or array and scalar.

        Args:
            a (NumpyTestArray): First operand.
            b (float | NumpyTestArray): Second operand.

        Returns:
            NumpyTestArray: Elementwise division.
        """
        return ensure_ndarray(np.divide(a, b))

    def sqrt(self, a: NumpyTestArray) -> NumpyTestArray:
        """Compute square root.

        Args:
            a (NumpyTestArray): Input array.

        Returns:
            NumpyTestArray: Elementwise square root.
        """
        return ensure_ndarray(np.sqrt(a))


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
        arr: NumpyTestArray,
        pad_width: tuple[tuple[int, int], tuple[int, int], tuple[int, int], tuple[int, int]],
        mode: str = "reflect",
    ) -> NumpyTestArray:
        """Pad 4D array without relying on ufunc initial sentinel.

        Args:
            arr (NumpyTestArray): 4D array to pad.
            pad_width (tuple[tuple[int, int], tuple[int, int], tuple[int, int], tuple[int, int]]): Padding dimensions.
            mode (str): Padding mode. Defaults to "reflect".

        Returns:
            NumpyTestArray: Padded array.
        """
        _ = mode
        (b0, b1), (c0, c1), (h0, h1), (w0, w1) = pad_width
        b, c, h, w = arr.shape
        padded = np.zeros((b + b0 + b1, c + c0 + c1, h + h0 + h1, w + w0 + w1), dtype=arr.dtype)
        padded[b0 : b0 + b, c0 : c0 + c, h0 : h0 + h, w0 : w0 + w] = arr
        return ensure_ndarray(padded)


class MockRandModuleWithUniformAndRand:
    """Mock random module having uniform and rand."""

    def uniform(self, size: tuple[int, ...] = ()) -> NumpyTestArray:
        """Generate uniform random values.

        Args:
            size (tuple[int, ...]): Target size.

        Returns:
            NumpyTestArray: Result array.
        """
        return np.ones(size)

    def rand(self, *shape: int) -> NumpyTestArray:
        """Generate random values.

        Args:
            *shape (int): Dimensions.

        Returns:
            NumpyTestArray: Result array.
        """
        return np.ones(shape)


class MockRandFallbackModule:
    """Mock random module triggering fallback at line 93, 122, and 143."""

    def __init__(self) -> None:
        """Initialize mock module."""
        self.calls: dict[str, int] = {}

    def _get_fn(self, name: str) -> TypedVariadicCallable[NumpyTestArray]:
        """Return callable only on second check.

        Args:
            name (str): Input `name` value (name) for `_get_fn`.

        Returns:
            TypedVariadicCallable[NumpyTestArray]: Computed output value from `_get_fn`.

        Raises:
            AttributeError: If execution or validation fails in `_get_fn`.
        """
        count: int = self.calls.get(name, 0)
        self.calls[name] = count + 1
        if count == 0:
            raise AttributeError(f"Initial check fails for {name}")
        return lambda *args, **kwargs: np.ones((2, 2))

    @property
    def uniform(self) -> TypedVariadicCallable[NumpyTestArray]:
        """Dynamic uniform attribute.

        Returns:
            TypedVariadicCallable[NumpyTestArray]: Computed output value from `uniform`.
        """
        return self._get_fn("uniform")

    @property
    def normal(self) -> TypedVariadicCallable[NumpyTestArray]:
        """Dynamic normal attribute.

        Returns:
            TypedVariadicCallable[NumpyTestArray]: Computed output value from `normal`.
        """
        return self._get_fn("normal")

    @property
    def randint(self) -> TypedVariadicCallable[NumpyTestArray]:
        """Dynamic randint attribute.

        Returns:
            TypedVariadicCallable[NumpyTestArray]: Computed output value from `randint`.
        """
        return self._get_fn("randint")


class MockBackendWithFallback:
    """Mock backend wrapping fallback random module."""

    def __init__(self) -> None:
        """Initialize backend."""
        self.random: MockRandFallbackModule = MockRandFallbackModule()


class MockSubRand:
    """Sub-module offering rand operation."""

    def rand(self, *shape: int) -> NumpyTestArray:
        """Generate random values.

        Args:
            *shape (int): Dimensions.

        Returns:
            NumpyTestArray: Result array.
        """
        return np.zeros(shape)


class MockBackendWithRandOnly:
    """Mock backend having only rand on random attribute."""

    def __init__(self) -> None:
        """Initialize backend with sub-module."""
        self.random: MockSubRand = MockSubRand()


class MockBackendDirectRand:
    """Mock backend having rand directly."""

    def rand(self, *shape: int) -> NumpyTestArray:
        """Generate random array.

        Args:
            *shape (int): Dimensions.

        Returns:
            NumpyTestArray: Result array.
        """
        return np.ones(shape)


class MockBackendDirectUniform:
    """Mock backend having uniform directly."""

    def uniform(self, size: tuple[int, ...] = ()) -> NumpyTestArray:
        """Generate uniform array.

        Args:
            size (tuple[int, ...]): Dimensions.

        Returns:
            NumpyTestArray: Result array.
        """
        return np.ones(size)


class MockSubRandn:
    """Sub-module offering randn operation."""

    def randn(self, *shape: int) -> NumpyTestArray:
        """Generate standard normal array.

        Args:
            *shape (int): Dimensions.

        Returns:
            NumpyTestArray: Result array.
        """
        return np.zeros(shape)


class MockBackendWithRandnOnly:
    """Mock backend having only randn on random attribute."""

    def __init__(self) -> None:
        """Initialize backend with sub-module."""
        self.random: MockSubRandn = MockSubRandn()


class MockBackendDirectRandn:
    """Mock backend having randn directly."""

    def randn(self, *shape: int) -> NumpyTestArray:
        """Generate standard normal array.

        Args:
            *shape (int): Dimensions.

        Returns:
            NumpyTestArray: Result array.
        """
        return np.ones(shape)


class MockBackendDirectNormal:
    """Mock backend having normal directly."""

    def normal(self, size: tuple[int, ...] = ()) -> NumpyTestArray:
        """Generate normal array.

        Args:
            size (tuple[int, ...]): Dimensions.

        Returns:
            NumpyTestArray: Result array.
        """
        return np.ones(size)


class MockStringConvertibleGroup(MockTestStub):
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


def test_core_group_ops_coverage() -> None:
    """Verify group mean, variance, and group norm logic and fallbacks."""
    # 1. _get_reduction_axes
    axes = group_ops_mod._get_reduction_axes([2, 4, 8, 8], 1)
    assert axes == (2, 3)

    # 2. _invoke_grouped_op error on unknown op
    with pytest.raises(ValueError, match="Unknown grouped reduction op: unknown"):
        group_ops_mod._invoke_grouped_op(np, "unknown", np.ones((2, 2)), (1,))

    # 3. _invoke_grouped_op with failing axis triggering TypeError fallback to dim
    torch_backend = MockTorchStyleBackend()
    failing_backend = MockFailingAxisBackend(torch_backend)
    x_test = np.ones((2, 4, 8, 8))

    mean_res = group_ops_mod._invoke_grouped_op(failing_backend, "mean", x_test, (2, 3))
    assert extract_test_shape(mean_res) == (2, 4, 1, 1)

    var_res = group_ops_mod._invoke_grouped_op(failing_backend, "variance", x_test, (2, 3))
    assert extract_test_shape(var_res) == (2, 4, 1, 1)

    # 4. _group_mean and _group_variance with positional groups vs kwarg groups
    x = np.ones((2, 4, 8, 8), dtype=np.float32)

    # kwarg groups and negative axis
    mean_kw = group_ops_mod._group_mean(np, x, groups=2, axis=-1)
    assert extract_test_shape(mean_kw) == (2, 1, 1, 2, 1)

    # positional groups and non-negative axis
    mean_pos = group_ops_mod._group_mean(np, x, 2, axis=1)
    assert extract_test_shape(mean_pos) == (2, 2, 1, 1, 1)

    # positional groups with string conversion fallback
    str_group = MockStringConvertibleGroup("2")
    mean_str = group_ops_mod._group_mean(np, x, str_group, axis=1)
    assert extract_test_shape(mean_str) == (2, 2, 1, 1, 1)

    # variance kwarg groups and negative axis
    var_kw = group_ops_mod._group_variance(np, x, groups=2, axis=-1)
    assert extract_test_shape(var_kw) == (2, 1, 1, 2, 1)

    # variance positional groups
    var_pos = group_ops_mod._group_variance(np, x, 2, axis=1)
    assert extract_test_shape(var_pos) == (2, 2, 1, 1, 1)

    # variance with string group
    var_str = group_ops_mod._group_variance(np, x, str_group, axis=1)
    assert extract_test_shape(var_str) == (2, 2, 1, 1, 1)

    # 5. _apply_affine_transform with weight and bias combinations
    out = np.ones((2, 4, 8, 8), dtype=np.float32)
    weight = np.full((4,), 2.0, dtype=np.float32)
    bias = np.full((4,), 0.5, dtype=np.float32)

    aff_both = group_ops_mod._apply_affine_transform(np, out, 1, weight=weight, bias=bias)
    np.testing.assert_allclose(ensure_ndarray(aff_both), np.full((2, 4, 8, 8), 2.5))

    aff_w_only = group_ops_mod._apply_affine_transform(np, out, 1, weight=weight, bias=None)
    np.testing.assert_allclose(ensure_ndarray(aff_w_only), np.full((2, 4, 8, 8), 2.0))

    aff_b_only = group_ops_mod._apply_affine_transform(np, out, 1, weight=None, bias=bias)
    np.testing.assert_allclose(ensure_ndarray(aff_b_only), np.full((2, 4, 8, 8), 1.5))

    aff_none = group_ops_mod._apply_affine_transform(np, out, 1, weight=None, bias=None)
    np.testing.assert_allclose(ensure_ndarray(aff_none), out)

    # 6. _group_norm with negative axis and with positive axis
    gn_neg = group_ops_mod._group_norm(np, x, groups=2, axis=-1, weight=np.ones(8), bias=np.zeros(8))
    assert extract_test_shape(gn_neg) == extract_test_shape(x)

    gn_pos = group_ops_mod._group_norm(np, x, 2, axis=1)
    assert extract_test_shape(gn_pos) == extract_test_shape(x)

    # group norm with string group positional
    gn_str = group_ops_mod._group_norm(np, x, str_group, axis=1)
    assert extract_test_shape(gn_str) == extract_test_shape(x)
