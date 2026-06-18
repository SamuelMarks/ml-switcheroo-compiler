"""Configuration settings and context management for the ml-switcheroo compiler.

This module provides the global `Config` singleton and context managers to temporarily
modify configuration states such as eager execution and stream contexts. It allows users
to dynamically adjust compiler behavior within specific execution scopes
"""

import os
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field

from ml_switcheroo_compiler.core.device import Device, DeviceType
from ml_switcheroo_compiler.core.dtype import DType


@dataclass
class ConfigState:
    """State held in ContextVar for configuration."""

    eager_mode: bool = field(
        default_factory=lambda: os.environ.get("SWITCHEROO_EAGER_MODE", "0") == "1"
    )
    backend: str = field(default_factory=lambda: os.environ.get("SWITCHEROO_BACKEND", "numpy"))
    default_float_dtype: DType = DType.Float32
    default_int_dtype: DType = DType.Int64
    default_device: Device = field(default_factory=lambda: Device(DeviceType.CPU, 0))
    current_stream: str = "default"
    jax_enable_x64: bool = False
    layout_map: object = None

    def clone(self) -> "ConfigState":
        """Clone."""
        import copy

        return copy.copy(self)


_config_state_var: ContextVar[ConfigState] = ContextVar("config_state")
_config_state_var.set(ConfigState())


class Config:
    """Global configuration state for the ml-switcheroo compiler.

    Manages settings such as eager execution mode, default data types for floats
    and integers, the default execution device, and the current execution stream
    This class is typically used as a global singleton instance

    Attributes:
    eager_mode (bool): Whether eager execution mode is enabled
    default_float_dtype (DType): The default data type for floating-point
    operations
    default_int_dtype (DType): The default data type for integer operations
    default_device (Device): The default device for execution
    current_stream (str): The name of the current execution stream
    """

    def __init__(self) -> None:
        """Initialize the Config."""
        pass

    @property
    def _state(self) -> ConfigState:
        return _config_state_var.get()

    @property
    def backend(self) -> str:
        """Backend."""
        return self._state.backend

    @backend.setter
    def backend(self, value: str) -> None:
        self._state.backend = value

    @property
    def default_float_dtype(self) -> DType:
        """Float dtype."""
        return self._state.default_float_dtype

    @default_float_dtype.setter
    def default_float_dtype(self, value: DType) -> None:
        self._state.default_float_dtype = value

    @property
    def default_int_dtype(self) -> DType:
        """Int dtype."""
        return self._state.default_int_dtype

    @default_int_dtype.setter
    def default_int_dtype(self, value: DType) -> None:
        self._state.default_int_dtype = value

    @property
    def default_device(self) -> Device:
        """Device."""
        return self._state.default_device

    @default_device.setter
    def default_device(self, value: Device) -> None:
        self._state.default_device = value

    @property
    def current_stream(self) -> str:
        """Stream."""
        return self._state.current_stream

    @current_stream.setter
    def current_stream(self, value: str) -> None:
        self._state.current_stream = value

    @property
    def layout_map(self) -> object:
        """Layout map."""
        return self._state.layout_map

    @layout_map.setter
    def layout_map(self, value: object) -> None:
        self._state.layout_map = value

    @property
    def jax_enable_x64(self) -> bool:
        """JAX x64."""
        return self._state.jax_enable_x64

    @jax_enable_x64.setter
    def jax_enable_x64(self, value: bool) -> None:
        self._state.jax_enable_x64 = value

    def clear_cache(self) -> None:
        """Clear memory cache. Hook for backends like MLX."""

    @property
    def eager_mode(self) -> bool:
        """Evaluate eager mode.

        Returns:
            bool: A boolean indicating the result of the check.
        """
        from ml_switcheroo_compiler.tracing import _tracer

        if _tracer.is_tracing:
            return False
        return self._state.eager_mode

    @eager_mode.setter
    def eager_mode(self, value: bool) -> None:
        """Eager mode.

        Args:
            value (bool): The value to set or add.
        """
        self._state.eager_mode = value

    def clone(self) -> ConfigState:
        """Clone the current configuration.

        Returns:
            ConfigState: The result of the operation
        """
        return self._state.clone()


# Singleton instance proxy
config = Config()


@contextmanager
def ConfigContext(**kwargs: object) -> Iterator[None]:
    """Context manager for temporarily overriding global configuration values.

    Args:
        **kwargs (object): Configuration keys and the values to temporarily set

    Yields:
    None: Yields control to the enclosed block with the overridden configuration

    Raises:
    ValueError: If any of the provided keys do not exist in the configuration

    Returns:
        Iterator[None]: The evaluated output resulting from this operation.
    """
    old_state = _config_state_var.get()
    new_state = old_state.clone()

    for k, v in kwargs.items():
        if hasattr(new_state, k) or k == "eager_mode":
            if k == "eager_mode":
                new_state.eager_mode = v
            else:
                setattr(new_state, k, v)
        else:
            msg = f"Unknown config key: {k}"
            raise ValueError(msg)

    token = _config_state_var.set(new_state)
    try:
        yield
    finally:
        _config_state_var.reset(token)


def EagerMode() -> Iterator[None]:
    """Context manager to temporarily enable eager execution mode.

    Yields:
    None: Yields control to the enclosed block with eager mode enabled

    Returns:
        Iterator[None]: The evaluated output resulting from this operation.
    """
    return ConfigContext(eager_mode=True)


def StreamContext(stream_name: str) -> Iterator[None]:
    """Context manager to temporarily switch the current execution stream.

    Args:
        stream_name (str): The name of the stream to execute on

    Yields:
    None: Yields control to the enclosed block with the specified stream active

    Returns:
        Iterator[None]: The evaluated output resulting from this operation.
    """
    return ConfigContext(current_stream=stream_name)
