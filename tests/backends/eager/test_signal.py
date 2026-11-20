"""Test module."""

from __future__ import annotations

import numpy as np
import pytest

import ml_switcheroo_compiler.backends.eager.signal as signal_mod
from ml_switcheroo_compiler.backends.eager.signal import (
    _apply_conv2d_batch,
    _apply_median_filter_batch,
    _generate_gaussian_kernel,
    _get_blur_config,
    gaussian_blur_eager,
    median_filter_eager,
)
from ml_switcheroo_compiler.backends.numpy.eager import ensure_ndarray
from ml_switcheroo_compiler.core.shape import TypedVariadicCallable
from ml_switcheroo_compiler.ops.configs import BlurConfig
from tests.testing_utils import BackendArg, MockTestStub, coerce_shape, extract_test_shape


class DummyConfig:
    """Test class DummyConfig."""

    pass


def test_signal() -> None:
    """Test function test_signal."""
    import numpy as np_mod

    # 1. _generate_gaussian_kernel
    kernel = _generate_gaussian_kernel(np_mod, (3, 3), (1.0, 1.0))
    assert extract_test_shape(kernel) == (3, 3)
    np.testing.assert_allclose(np_mod.sum(ensure_ndarray(kernel)), 1.0)

    # 2. _apply_conv2d_batch
    imgs = np.ones((1, 1, 3, 3))
    out = _apply_conv2d_batch(np_mod, imgs, kernel, "reflect")
    assert extract_test_shape(out) == (1, 1, 3, 3)

    with pytest.raises(ValueError):
        _apply_conv2d_batch(np_mod, np.ones((3, 3)), kernel, "reflect")

    # 3. _get_blur_config
    c1 = _get_blur_config({"kernel_size": (5, 5)}, None)
    assert c1.kernel_size == (5, 5)
    c2 = BlurConfig(kernel_size=(7, 7), sigma=(2.0, 2.0))
    assert id(_get_blur_config({}, c2)) == id(c2)

    # 4. gaussian_blur_eager
    imgs2 = np.ones((1, 1, 3, 3))
    res1 = gaussian_blur_eager(np_mod, imgs2, kernel_size=(3, 3), sigma=(1.0, 1.0))
    assert extract_test_shape(res1) == (1, 1, 3, 3)

    # 5. _apply_median_filter_batch
    imgs_med = np.random.rand(1, 1, 3, 3)
    out_med1 = _apply_median_filter_batch(np_mod, imgs_med, (3, 3), "same")
    assert extract_test_shape(out_med1) == (1, 1, 3, 3)

    out_med2 = _apply_median_filter_batch(np_mod, imgs_med, (3, 3), "reflect")
    assert extract_test_shape(out_med2) == (1, 1, 3, 3)

    with pytest.raises(ValueError):
        _apply_median_filter_batch(np_mod, np.ones((3, 3)), (3, 3), "same")

    # 6. median_filter_eager
    res2 = median_filter_eager(np_mod, imgs_med, (3, 3))
    assert extract_test_shape(res2) == (1, 1, 3, 3)


class MockCpuResult:
    """Mock cpu container with numpy conversion."""

    def __init__(self, arr: np.ndarray[tuple[int, ...], np.dtype[np.generic]]) -> None:
        """Initialize container.

        Args:
            arr (np.ndarray[tuple[int, ...], np.dtype[np.generic]]): Underlying array.
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
            arr (np.ndarray[tuple[int, ...], np.dtype[np.generic]]): Underlying array.
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
            arr (np.ndarray[tuple[int, ...], np.dtype[np.generic]]): Underlying array.
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
            arr (np.ndarray[tuple[int, ...], np.dtype[np.generic]]): Underlying numpy array.
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

    def array(
        self, val: list[int] | np.ndarray[tuple[int, ...], np.dtype[np.generic]], dtype: str | None = None
    ) -> np.ndarray[tuple[int, ...], np.dtype[np.generic]]:
        """Convert input to array.

        Args:
            val (list[int] | np.ndarray[tuple[int, ...], np.dtype[np.generic]]): Input value.
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
        self, x: np.ndarray[tuple[int, ...], np.dtype[np.generic]], dim: tuple[int, ...], keepdim: bool = True
    ) -> np.ndarray[tuple[int, ...], np.dtype[np.generic]]:
        """Compute mean across dimensions.

        Args:
            x (np.ndarray[tuple[int, ...], np.dtype[np.generic]]): Input array.
            dim (tuple[int, ...]): Target dimensions.
            keepdim (bool): Whether to keep dimensions.

        Returns:
            np.ndarray: Computed mean.
        """
        return ensure_ndarray(np.mean(x.astype(np.float64), axis=dim, keepdims=keepdim))

    def var(
        self,
        x: np.ndarray[tuple[int, ...], np.dtype[np.generic]],
        dim: tuple[int, ...],
        keepdim: bool = True,
        unbiased: bool = False,
    ) -> np.ndarray[tuple[int, ...], np.dtype[np.generic]]:
        """Compute variance across dimensions.

        Args:
            x (np.ndarray[tuple[int, ...], np.dtype[np.generic]]): Input array.
            dim (tuple[int, ...]): Target dimensions.
            keepdim (bool): Whether to keep dimensions.
            unbiased (bool): Whether to use unbiased estimator.

        Returns:
            np.ndarray: Computed variance.
        """
        ddof: int = 1 if unbiased else 0
        return ensure_ndarray(np.var(x.astype(np.float64), axis=dim, keepdims=keepdim, ddof=ddof))


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
        x: np.ndarray[tuple[int, ...], np.dtype[np.generic]],
        *args: int,
        **kwargs: tuple[int, ...] | bool | int | str,
    ) -> np.ndarray[tuple[int, ...], np.dtype[np.generic]]:
        """Compute mean or raise TypeError if axis in kwargs.

        Args:
            x (np.ndarray[tuple[int, ...], np.dtype[np.generic]]): Input array.
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
        x: np.ndarray[tuple[int, ...], np.dtype[np.generic]],
        *args: int,
        **kwargs: tuple[int, ...] | bool | int | str,
    ) -> np.ndarray[tuple[int, ...], np.dtype[np.generic]]:
        """Compute variance or raise TypeError if axis in kwargs.

        Args:
            x (np.ndarray[tuple[int, ...], np.dtype[np.generic]]): Input array.
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
        a: np.ndarray[tuple[int, ...], np.dtype[np.generic]],
        b: float | np.ndarray[tuple[int, ...], np.dtype[np.generic]],
    ) -> np.ndarray[tuple[int, ...], np.dtype[np.generic]]:
        """Add two arrays or array and scalar.

        Args:
            a (np.ndarray[tuple[int, ...], np.dtype[np.generic]]): First operand.
            b (float | np.ndarray[tuple[int, ...], np.dtype[np.generic]]): Second operand.

        Returns:
            np.ndarray: Elementwise addition.
        """
        return np.add(a, b)

    def subtract(
        self,
        a: np.ndarray[tuple[int, ...], np.dtype[np.generic]],
        b: float | np.ndarray[tuple[int, ...], np.dtype[np.generic]],
    ) -> np.ndarray[tuple[int, ...], np.dtype[np.generic]]:
        """Subtract two arrays or array and scalar.

        Args:
            a (np.ndarray[tuple[int, ...], np.dtype[np.generic]]): First operand.
            b (float | np.ndarray[tuple[int, ...], np.dtype[np.generic]]): Second operand.

        Returns:
            np.ndarray: Elementwise subtraction.
        """
        return np.subtract(a, b)

    def multiply(
        self,
        a: np.ndarray[tuple[int, ...], np.dtype[np.generic]],
        b: float | np.ndarray[tuple[int, ...], np.dtype[np.generic]],
    ) -> np.ndarray[tuple[int, ...], np.dtype[np.generic]]:
        """Multiply two arrays or array and scalar.

        Args:
            a (np.ndarray[tuple[int, ...], np.dtype[np.generic]]): First operand.
            b (float | np.ndarray[tuple[int, ...], np.dtype[np.generic]]): Second operand.

        Returns:
            np.ndarray: Elementwise product.
        """
        return np.multiply(a, b)

    def divide(
        self,
        a: np.ndarray[tuple[int, ...], np.dtype[np.generic]],
        b: float | np.ndarray[tuple[int, ...], np.dtype[np.generic]],
    ) -> np.ndarray[tuple[int, ...], np.dtype[np.generic]]:
        """Divide two arrays or array and scalar.

        Args:
            a (np.ndarray[tuple[int, ...], np.dtype[np.generic]]): First operand.
            b (float | np.ndarray[tuple[int, ...], np.dtype[np.generic]]): Second operand.

        Returns:
            np.ndarray: Elementwise division.
        """
        return np.divide(a, b)

    def sqrt(
        self, a: np.ndarray[tuple[int, ...], np.dtype[np.generic]]
    ) -> np.ndarray[tuple[int, ...], np.dtype[np.generic]]:
        """Compute square root.

        Args:
            a (np.ndarray[tuple[int, ...], np.dtype[np.generic]]): Input array.

        Returns:
            np.ndarray: Elementwise square root.
        """
        return np.sqrt(a)


class SignalBackendWrapper(MockTestStub):
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
        arr: np.ndarray[tuple[int, ...], np.dtype[np.generic]],
        pad_width: tuple[tuple[int, int], tuple[int, int], tuple[int, int], tuple[int, int]],
        mode: str = "reflect",
    ) -> np.ndarray[tuple[int, ...], np.dtype[np.generic]]:
        """Pad 4D array without relying on ufunc initial sentinel.

        Args:
            arr (np.ndarray[tuple[int, ...], np.dtype[np.generic]]): 4D array to pad.
            pad_width (tuple[tuple[int, int], tuple[int, int], tuple[int, int], tuple[int, int]]): Padding dimensions.
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

    def _get_fn(self, name: str) -> TypedVariadicCallable[np.ndarray[tuple[int, ...], np.dtype[np.generic]]]:
        """Return callable only on second check.

        Args:
            name (str): Input `name` value (name) for `_get_fn`.

        Returns:
            TypedVariadicCallable[np.ndarray[tuple[int, ...], np.dtype[np.generic]]]: Computed output value from
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
    def uniform(self) -> TypedVariadicCallable[np.ndarray[tuple[int, ...], np.dtype[np.generic]]]:
        """Dynamic uniform attribute.

        Returns:
            TypedVariadicCallable[np.ndarray[tuple[int, ...], np.dtype[np.generic]]]: Computed output value from
                `uniform`.
        """
        return self._get_fn("uniform")

    @property
    def normal(self) -> TypedVariadicCallable[np.ndarray[tuple[int, ...], np.dtype[np.generic]]]:
        """Dynamic normal attribute.

        Returns:
            TypedVariadicCallable[np.ndarray[tuple[int, ...], np.dtype[np.generic]]]: Computed output value from
                `normal`.
        """
        return self._get_fn("normal")

    @property
    def randint(self) -> TypedVariadicCallable[np.ndarray[tuple[int, ...], np.dtype[np.generic]]]:
        """Dynamic randint attribute.

        Returns:
            TypedVariadicCallable[np.ndarray[tuple[int, ...], np.dtype[np.generic]]]: Computed output value from
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


def test_signal_coverage() -> None:
    """Verify all signal operations including error branches and configurations."""
    backend = SignalBackendWrapper()

    # 1. Gaussian kernel calculation
    k = signal_mod._generate_gaussian_kernel(backend, (3, 3), (1.0, 1.0))
    assert extract_test_shape(k) == (3, 3)
    np.testing.assert_allclose(float(np.sum(ensure_ndarray(k))), 1.0, atol=1e-5)

    # 2. _apply_conv2d_batch error on non-4D
    with pytest.raises(ValueError, match="Images must be 4D"):
        signal_mod._apply_conv2d_batch(backend, np.ones((3, 3)), k, "reflect")

    # 3. _get_blur_config with config_obj None and with dict
    cfg1 = signal_mod._get_blur_config({"kernel_size": (5, 5), "sigma": (2.0, 2.0), "data_format": "NCHW"}, None)
    assert cfg1.kernel_size == (5, 5)
    assert cfg1.sigma == (2.0, 2.0)
    assert cfg1.data_format == "NCHW"

    # Default fallback when kwargs empty
    cfg2 = signal_mod._get_blur_config({}, None)
    assert cfg2.kernel_size == (3, 3)
    assert cfg2.sigma == (1.0, 1.0)
    assert cfg2.data_format == "None"

    # Fallback when kwargs does not have get attribute
    class _NoGetDict(dict[str, BackendArg]):
        """Dictionary subclass used to exercise missing `get` symbol fallback."""

    def _raise_missing_get(_self: _NoGetDict) -> None:
        """Raise AttributeError when `get` is accessed.

        Args:
            _self: Dictionary stub instance.

        Raises:
            AttributeError: Always raised to hide `get`.
        """
        raise AttributeError("get")

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(_NoGetDict, "get", property(_raise_missing_get))
        cfg_no_get = signal_mod._get_blur_config(_NoGetDict(), None)
    assert cfg_no_get.kernel_size == (3, 3)
    assert cfg_no_get.data_format is None

    # When config_obj is provided
    custom_cfg = BlurConfig(kernel_size=(3, 3), sigma=(0.5, 0.5))
    assert signal_mod._get_blur_config({}, custom_cfg) is custom_cfg

    # 4. gaussian_blur_eager full execution
    imgs = np.ones((1, 2, 8, 8), dtype=np.float32)
    blurred = signal_mod.gaussian_blur_eager(backend, imgs, custom_cfg)
    assert extract_test_shape(blurred) == (1, 2, 8, 8)

    # 5. _apply_median_filter_batch error on non-4D
    with pytest.raises(ValueError, match="Images must be 4D"):
        signal_mod._apply_median_filter_batch(backend, np.ones((2, 2)), (3, 3), "same")

    # 6. median_filter_eager with same padding and reflect padding
    med_same = signal_mod.median_filter_eager(backend, imgs, (3, 3), padding="same")
    assert extract_test_shape(med_same) == (1, 2, 8, 8)
    med_reflect = signal_mod.median_filter_eager(backend, imgs, (3, 3), padding="reflect")
    assert extract_test_shape(med_reflect) == (1, 2, 8, 8)
