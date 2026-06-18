"""Unit tests for the core components of the ml_switcheroo_compiler library.

This module contains comprehensive test suites verifying the behavior, integrity, and
correctness of core data types, device representations, custom exceptions, configuration
contexts, and the base Tensor class.
"""

import pytest

from ml_switcheroo_compiler.core import (
    BackendNotSupportedError,
    CompilationError,
    ConfigContext,
    Device,
    DeviceType,
    DType,
    DTypePromotionError,
    QuantDType,
    ShapeMismatchError,
    SwitcherooError,
    Tensor,
    TracingError,
    UnimplementedMathError,
    config,
)


def test_dtype_enums() -> None:
    """Verifies the string values of DType and QuantDType enumeration members.

    This test ensures that the enum values match their expected string
    representations
    (e.g., 'float32' for DType.Float32 and 'qint8' for QuantDType.QInt8)

    Returns:
    None
    """
    assert DType.Float32.value == "float32"
    assert QuantDType.QInt8.value == "qint8"


def test_device() -> None:
    """Verifies the behavior, equality, hashing, and representation of the Device class.

    This test ensures that Device instances are correctly compared for equality,
    hash consistently, and produce the expected string representation

    Returns:
    None
    """
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
    """Verifies the exception hierarchy of the ml_switcheroo_compiler library.

    This test ensures that all custom exceptions inherit from the base
    SwitcherooError

    Returns:
    None
    """
    assert issubclass(TracingError, SwitcherooError)
    assert issubclass(CompilationError, SwitcherooError)
    assert issubclass(ShapeMismatchError, SwitcherooError)
    assert issubclass(DTypePromotionError, SwitcherooError)
    assert issubclass(BackendNotSupportedError, SwitcherooError)
    assert issubclass(UnimplementedMathError, SwitcherooError)


def test_config() -> None:
    """Verifies the behavior of the Config and ConfigContext classes.

    This test ensures that configuration options can be temporarily overridden using
    a context manager and that invalid configuration keys raise a ValueError

    Returns:
    None
    """
    from ml_switcheroo_compiler.core.config import EagerMode, StreamContext

    orig_mode = config.eager_mode
    with ConfigContext(eager_mode=not orig_mode):
        assert config.eager_mode == (not orig_mode)
    assert config.eager_mode == orig_mode

    with EagerMode():
        assert config.eager_mode is True

    orig_stream = config.current_stream
    with StreamContext("test_stream"):
        assert config.current_stream == "test_stream"
    assert config.current_stream == orig_stream

    with pytest.raises(ValueError, match="Unknown config key: invalid_key"):
        # Calling __enter__ directly to avoid pytest-cov generator misses
        ConfigContext(invalid_key=True).__enter__()


def test_config_env_var(monkeypatch: object) -> None:
    """Verifies that the Config class correctly parses environment variables.

    This test uses monkeypatch to set an environment variable and asserts that
    the Config class initializes with the corresponding value

    Args:
    monkeypatch (pytest.MonkeyPatch): Pytest fixture used to mock environment
    variables

    Returns:
    None
    """

    monkeypatch.setenv("SWITCHEROO_EAGER_MODE", "1")
    from ml_switcheroo_compiler.core.config import ConfigState

    new_state = ConfigState()
    assert new_state.eager_mode is True


def test_tensor() -> None:
    """Verifies the initialization and properties of the Tensor class.

    This test ensures that a Tensor instance correctly stores and exposes its shape,
    data type, device, gradient requirement, and underlying data

    Returns:
    None
    """
    device = Device(DeviceType.CPU, 0)
    t = Tensor(
        data=[1, 2],
        shape=(2,),
        dtype=DType.Int32,
        device=device,
        requires_grad=True,
    )

    assert t.shape == (2,)
    assert t.dtype == DType.Int32
    assert t.device == device
    assert t.requires_grad is True
    assert t.data == [1, 2]
