"""Tests for test_signal_coverage."""

from __future__ import annotations

from typing import Callable

import numpy as np
import pytest

import ml_switcheroo_compiler.backends.eager.signal as signal_mod
from ml_switcheroo_compiler.ops.configs import BlurConfig


class MockCpuResult:
    """Mock cpu container with numpy conversion."""

    def __init__(self, arr: np.ndarray) -> None:
        """Initialize container.

        Args:
            arr (np.ndarray): Underlying array.
        """
        self._arr: np.ndarray = arr

    def numpy(self) -> np.ndarray:
        """Convert to numpy array.

        Returns:
            np.ndarray: Array.
        """
        return self._arr


class MockDetachResult:
    """Mock detach result providing cpu method."""

    def __init__(self, arr: np.ndarray) -> None:
        """Initialize mock detach.

        Args:
            arr (np.ndarray): Underlying array.
        """
        self._arr: np.ndarray = arr

    def cpu(self) -> MockCpuResult:
        """Move to cpu.

        Returns:
            MockCpuResult: Cpu container.
        """
        return MockCpuResult(self._arr)


class MockTorchDetachOnly:
    """Mock object with detach method but no direct numpy attribute."""

    def __init__(self, arr: np.ndarray) -> None:
        """Initialize mock.

        Args:
            arr (np.ndarray): Underlying array.
        """
        self._arr: np.ndarray = arr

    def detach(self) -> MockDetachResult:
        """Detach tensor.

        Returns:
            MockDetachResult: Detached mock.
        """
        return MockDetachResult(self._arr)


class MockHasNumpyOnly:
    """Mock object with only a numpy method."""

    def __init__(self, arr: np.ndarray) -> None:
        """Initialize mock.

        Args:
            arr (np.ndarray): Underlying numpy array.
        """
        self._arr: np.ndarray = arr

    def numpy(self) -> np.ndarray:
        """Convert to numpy array.

        Returns:
            np.ndarray: Converted array.
        """
        return self._arr


class MockBackendModule:
    """Mock backend module supporting array operations."""

    def array(self, val: list[int] | np.ndarray, dtype: str | None = None) -> np.ndarray:
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

    def mean(self, x: np.ndarray, dim: tuple[int, ...], keepdim: bool = True) -> np.ndarray:
        """Compute mean across dimensions.

        Args:
            x (np.ndarray): Input array.
            dim (tuple[int, ...]): Target dimensions.
            keepdim (bool): Whether to keep dimensions.

        Returns:
            np.ndarray: Computed mean.
        """
        return np.mean(x, axis=dim, keepdims=keepdim)

    def var(self, x: np.ndarray, dim: tuple[int, ...], keepdim: bool = True, unbiased: bool = False) -> np.ndarray:
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
        return np.var(x, axis=dim, keepdims=keepdim, ddof=ddof)


class MockFailingAxisBackend:
    """Mock backend raising TypeError when axis is passed to mean or var."""

    def __init__(self, fallback: MockTorchStyleBackend) -> None:
        """Initialize mock.

        Args:
            fallback (MockTorchStyleBackend): Fallback implementation.
        """
        self.fallback: MockTorchStyleBackend = fallback

    def mean(self, x: np.ndarray, *args: int, **kwargs: tuple[int, ...] | bool | int | str) -> np.ndarray:
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
        dim_val: tuple[int, ...] = kwargs["dim"]  # type: ignore[assignment]
        keepdim_val: bool = bool(kwargs.get("keepdim", True))
        return self.fallback.mean(x, dim=dim_val, keepdim=keepdim_val)

    def var(self, x: np.ndarray, *args: int, **kwargs: tuple[int, ...] | bool | int | str) -> np.ndarray:
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
        dim_val: tuple[int, ...] = kwargs["dim"]  # type: ignore[assignment]
        keepdim_val: bool = bool(kwargs.get("keepdim", True))
        unbiased_val: bool = bool(kwargs.get("unbiased", False))
        return self.fallback.var(x, dim=dim_val, keepdim=keepdim_val, unbiased=unbiased_val)


class MockMathBackend:
    """Mock backend providing elementwise mathematical operations."""

    def add(self, a: np.ndarray, b: float | np.ndarray) -> np.ndarray:
        """Add two arrays or array and scalar.

        Args:
            a (np.ndarray): First operand.
            b (float | np.ndarray): Second operand.

        Returns:
            np.ndarray: Elementwise addition.
        """
        return np.add(a, b)

    def subtract(self, a: np.ndarray, b: float | np.ndarray) -> np.ndarray:
        """Subtract two arrays or array and scalar.

        Args:
            a (np.ndarray): First operand.
            b (float | np.ndarray): Second operand.

        Returns:
            np.ndarray: Elementwise subtraction.
        """
        return np.subtract(a, b)

    def multiply(self, a: np.ndarray, b: float | np.ndarray) -> np.ndarray:
        """Multiply two arrays or array and scalar.

        Args:
            a (np.ndarray): First operand.
            b (float | np.ndarray): Second operand.

        Returns:
            np.ndarray: Elementwise product.
        """
        return np.multiply(a, b)

    def divide(self, a: np.ndarray, b: float | np.ndarray) -> np.ndarray:
        """Divide two arrays or array and scalar.

        Args:
            a (np.ndarray): First operand.
            b (float | np.ndarray): Second operand.

        Returns:
            np.ndarray: Elementwise division.
        """
        return np.divide(a, b)

    def sqrt(self, a: np.ndarray) -> np.ndarray:
        """Compute square root.

        Args:
            a (np.ndarray): Input array.

        Returns:
            np.ndarray: Elementwise square root.
        """
        return np.sqrt(a)


class SignalBackendWrapper:
    """Signal execution backend ensuring reliable array padding."""

    def __getattr__(self, name: str) -> Callable[..., np.ndarray]:
        """Delegate to numpy module.

        Args:
            name (str): Attribute name.

        Returns:
            Callable[..., np.ndarray]: Resolved callable attribute.
        """
        return getattr(np, name)  # type: ignore[no-any-return]

    def pad(
        self,
        arr: np.ndarray,
        pad_width: tuple[tuple[int, int], tuple[int, int], tuple[int, int], tuple[int, int]],
        mode: str = "reflect",
    ) -> np.ndarray:
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

    def uniform(self, size: tuple[int, ...] = ()) -> np.ndarray:
        """Generate uniform random values.

        Args:
            size (tuple[int, ...]): Target size.

        Returns:
            np.ndarray: Result array.
        """
        return np.ones(size)

    def rand(self, *shape: int) -> np.ndarray:
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

    def __getattr__(self, name: str) -> Callable[..., np.ndarray]:
        """Dynamically return callable only on second check.

        Args:
            name (str): Attribute name.

        Returns:
            Callable[..., np.ndarray]: Function result.

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

    def rand(self, *shape: int) -> np.ndarray:
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

    def rand(self, *shape: int) -> np.ndarray:
        """Generate random array.

        Args:
            *shape (int): Dimensions.

        Returns:
            np.ndarray: Result array.
        """
        return np.ones(shape)


class MockBackendDirectUniform:
    """Mock backend having uniform directly."""

    def uniform(self, size: tuple[int, ...] = ()) -> np.ndarray:
        """Generate uniform array.

        Args:
            size (tuple[int, ...]): Dimensions.

        Returns:
            np.ndarray: Result array.
        """
        return np.ones(size)


class MockSubRandn:
    """Sub-module offering randn operation."""

    def randn(self, *shape: int) -> np.ndarray:
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

    def randn(self, *shape: int) -> np.ndarray:
        """Generate standard normal array.

        Args:
            *shape (int): Dimensions.

        Returns:
            np.ndarray: Result array.
        """
        return np.ones(shape)


class MockBackendDirectNormal:
    """Mock backend having normal directly."""

    def normal(self, size: tuple[int, ...] = ()) -> np.ndarray:
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
    assert k.shape == (3, 3)
    np.testing.assert_allclose(float(np.sum(k)), 1.0, atol=1e-5)

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
    cfg_no_get = signal_mod._get_blur_config(None, None)  # type: ignore[arg-type]
    assert cfg_no_get.kernel_size == (3, 3)
    assert cfg_no_get.data_format is None

    # When config_obj is provided
    custom_cfg = BlurConfig(kernel_size=(3, 3), sigma=(0.5, 0.5))
    assert signal_mod._get_blur_config({}, custom_cfg) is custom_cfg

    # 4. gaussian_blur_eager full execution
    imgs = np.ones((1, 2, 8, 8), dtype=np.float32)
    blurred = signal_mod.gaussian_blur_eager(backend, imgs, custom_cfg)
    assert blurred.shape == (1, 2, 8, 8)

    # 5. _apply_median_filter_batch error on non-4D
    with pytest.raises(ValueError, match="Images must be 4D"):
        signal_mod._apply_median_filter_batch(backend, np.ones((2, 2)), (3, 3), "same")

    # 6. median_filter_eager with same padding and reflect padding
    med_same = signal_mod.median_filter_eager(backend, imgs, (3, 3), padding="same")
    assert med_same.shape == (1, 2, 8, 8)
    med_reflect = signal_mod.median_filter_eager(backend, imgs, (3, 3), padding="reflect")
    assert med_reflect.shape == (1, 2, 8, 8)
