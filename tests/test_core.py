"""Tests for the core module."""

from typing import Any

import pytest
from ml_switcheroo.core import (
    DType,
    QuantDType,
    Device,
    DeviceType,
    SwitcherooError,
    TracingError,
    CompilationError,
    ShapeMismatchError,
    DTypePromotionError,
    BackendNotSupportedError,
    UnimplementedMathError,
    config,
    ConfigContext,
    Tensor,
)


def test_dtype_enums() -> None:
    """Test DType and QuantDType."""
    assert DType.Float32.value == "float32"
    assert QuantDType.QInt8.value == "qint8"


def test_device() -> None:
    """Test Device and DeviceType."""
    d1 = Device(DeviceType.CPU, 0)
    d2 = Device(DeviceType.CPU, 0)
    d3 = Device(DeviceType.GPU, 0)
    d4 = Device(DeviceType.CPU, 1)

    assert d1 == d2
    assert d1 != d3
    assert d1 != d4
    assert d1 != "not a device"
    assert hash(d1) == hash(d2)
    assert repr(d1) == "Device(cpu:0)"


def test_errors() -> None:
    """Test exception hierarchy."""
    assert issubclass(TracingError, SwitcherooError)
    assert issubclass(CompilationError, SwitcherooError)
    assert issubclass(ShapeMismatchError, SwitcherooError)
    assert issubclass(DTypePromotionError, SwitcherooError)
    assert issubclass(BackendNotSupportedError, SwitcherooError)
    assert issubclass(UnimplementedMathError, SwitcherooError)


def test_config() -> None:
    """Test Config and ConfigContext."""
    orig_mode = config.eager_mode
    with ConfigContext(eager_mode=not orig_mode):
        assert config.eager_mode == (not orig_mode)
    assert config.eager_mode == orig_mode

    with pytest.raises(ValueError, match="Unknown config key: invalid_key"):
        with ConfigContext(invalid_key=True):
            pass


def test_config_env_var(monkeypatch: Any) -> None:
    """Test that config parses env vars."""
    from ml_switcheroo.core.config import Config

    monkeypatch.setenv("SWITCHEROO_EAGER_MODE", "1")
    new_cfg = Config()
    assert new_cfg.eager_mode is True


def test_tensor() -> None:
    """Test Tensor class and properties."""
    device = Device(DeviceType.CPU, 0)
    t = Tensor(
        data=[1, 2], shape=(2,), dtype=DType.Int32, device=device, requires_grad=True
    )

    assert t.shape == (2,)
    assert t.dtype == DType.Int32
    assert t.device == device
    assert t.requires_grad is True
    assert t.data == [1, 2]
