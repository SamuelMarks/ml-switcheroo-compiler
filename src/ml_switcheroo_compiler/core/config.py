"""Configuration settings and context management for the ml-switcheroo compiler.

This module provides the global `Config` singleton and context managers to temporarily
modify configuration states such as eager execution and stream contexts. It allows users
to dynamically adjust compiler behavior within specific execution scopes
"""

import os
from collections.abc import Iterator
from contextlib import contextmanager

from ml_switcheroo_compiler.core.device import Device, DeviceType
from ml_switcheroo_compiler.core.dtype import DType


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
        self.eager_mode: bool = os.environ.get("SWITCHEROO_EAGER_MODE", "0") == "1"
        self.default_float_dtype: DType = DType.Float32
        self.default_int_dtype: DType = DType.Int64
        self.default_device: Device = Device(DeviceType.CPU, 0)
        self.current_stream: str = "default"

    def clear_cache(self) -> None:
        """Clear memory cache. Hook for backends like MLX."""

    @property
    def eager_mode(self) -> bool:
        """Evaluate eager mode."""
        from ml_switcheroo_compiler.tracing import _tracer

        if _tracer.is_tracing:
            return False
        return self._eager_mode

    @eager_mode.setter
    def eager_mode(self, value: bool) -> None:
        """Eager mode.

        Args:
            value (bool): The value parameter
        """
        self._eager_mode = value

    def clone(self) -> "Config":
        """Clone the current configuration.

        Args:
        Returns:
            'Config': The result of the operation
        """
        new_config = Config()
        new_config._eager_mode = self._eager_mode
        new_config.default_float_dtype = self.default_float_dtype
        new_config.default_int_dtype = self.default_int_dtype
        new_config.default_device = self.default_device
        new_config.current_stream = self.current_stream
        return new_config


# Singleton instance
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
    """
    old_config = config.clone()
    try:
        for k, v in kwargs.items():
            if hasattr(config, k):
                setattr(config, k, v)
            else:
                msg = f"Unknown config key: {k}"
                raise ValueError(msg)
        yield
    finally:
        config._eager_mode = old_config._eager_mode
        config.default_float_dtype = old_config.default_float_dtype
        config.default_int_dtype = old_config.default_int_dtype
        config.default_device = old_config.default_device


def EagerMode() -> Iterator[None]:
    """Context manager to temporarily enable eager execution mode.

    Yields:
    None: Yields control to the enclosed block with eager mode enabled
    """
    return ConfigContext(eager_mode=True)


def StreamContext(stream_name: str) -> Iterator[None]:
    """Context manager to temporarily switch the current execution stream.

    Args:
    stream_name (str): The name of the stream to execute on

    Yields:
    None: Yields control to the enclosed block with the specified stream active
    """
    return ConfigContext(current_stream=stream_name)
