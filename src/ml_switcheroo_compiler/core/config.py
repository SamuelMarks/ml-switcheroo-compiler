"""Configuration settings and context management for the ml-switcheroo compiler.

This module provides the global `Config` singleton and context managers to temporarily
modify configuration states such as eager execution and stream contexts. It allows users
to dynamically adjust compiler behavior within specific execution scopes
"""

import copy
import os
import sys
import typing
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field

from ml_switcheroo_compiler.core.device import Device, DeviceType
from ml_switcheroo_compiler.core.dtype import DType


@dataclass
class ExecutionConfig:
    """Execution settings."""

    eager_mode: bool = field(default_factory=lambda: os.environ.get("SWITCHEROO_EAGER_MODE", "0") == "1")
    backend: str = field(default_factory=lambda: os.environ.get("SWITCHEROO_BACKEND", "numpy"))
    current_stream: str = "default"
    op_determinism: bool = False
    tensor_float_32_execution: bool = True


@dataclass
class EnvironmentConfig:
    """Environment and typing settings."""

    default_float_dtype: DType = DType.Float32
    default_int_dtype: DType = DType.Int64
    default_device: Device = field(default_factory=lambda: Device(DeviceType.CPU, 0))
    jax_enable_x64: bool = False
    layout_map: object = None
    interactive_logging: bool = True


@dataclass
class ConfigState:
    """State held in ContextVar for configuration."""

    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    env: EnvironmentConfig = field(default_factory=EnvironmentConfig)

    def clone(self) -> "ConfigState":
        """Clone the configuration state.

        Returns:
            ConfigState: The cloned state.
        """
        return copy.deepcopy(self)


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

    @property
    def seed(self) -> typing.Optional[int]:
        """Get the global random seed.

        Returns:
            int: The random seed.
        """
        return self._state.env.seed

    @seed.setter
    def seed(self, value: typing.Optional[int]) -> None:
        """Set the global random seed.

        Args:
            value (int, optional): The random seed value.
        """
        self._state.env.seed = value

    @property
    def _state(self) -> ConfigState:
        """Get the current configuration state object.

        Returns:
            ConfigState: The state object.
        """
        return _config_state_var.get()

    @property
    def backend(self) -> str:
        """Get the active backend.

        Returns:
            str: The backend string.
        """
        return self._state.execution.backend

    @backend.setter
    def backend(self, value: str) -> None:
        """Set the active execution backend.

        Args:
            value (str): The name of the backend (e.g., 'numpy', 'mlx').
        """
        self._state.execution.backend = value

    @property
    def default_float_dtype(self) -> DType:
        """Get the default float dtype.

        Returns:
            DType: The default float dtype.
        """
        return self._state.env.default_float_dtype

    @default_float_dtype.setter
    def default_float_dtype(self, value: DType) -> None:
        """Set the default float dtype.

        Args:
            value (DType): The default float dtype.
        """
        self._state.env.default_float_dtype = value

    @property
    def default_int_dtype(self) -> DType:
        """Get the default int dtype.

        Returns:
            DType: The default int dtype.
        """
        return self._state.env.default_int_dtype

    @default_int_dtype.setter
    def default_int_dtype(self, value: DType) -> None:
        """Set the default int dtype.

        Args:
            value (DType): The default int dtype.
        """
        self._state.env.default_int_dtype = value

    @property
    def default_device(self) -> Device:
        """Get the default device.

        Returns:
            Device: The default device.
        """
        return self._state.env.default_device

    @default_device.setter
    def default_device(self, value: Device) -> None:
        """Set the default device.

        Args:
            value (Device): The default device.
        """
        self._state.env.default_device = value

    @property
    def current_stream(self) -> str:
        """Get the current stream.

        Returns:
            str: The current stream.
        """
        return self._state.execution.current_stream

    @current_stream.setter
    def current_stream(self, value: str) -> None:
        """Set the current stream.

        Args:
            value (str): The current stream name.
        """
        self._state.execution.current_stream = value

    @property
    def layout_map(self) -> object:
        """Get the layout map.

        Returns:
            object: The layout map.
        """
        return self._state.env.layout_map

    @layout_map.setter
    def layout_map(self, value: object) -> None:
        """Set the layout map.

        Args:
            value (object): The layout map.
        """
        self._state.env.layout_map = value

    @property
    def jax_enable_x64(self) -> bool:
        """Get whether JAX x64 is enabled.

        Returns:
            bool: True if JAX x64 is enabled.
        """
        return self._state.env.jax_enable_x64

    @jax_enable_x64.setter
    def jax_enable_x64(self, value: bool) -> None:
        """Set whether JAX x64 is enabled.

        Args:
            value (bool): True to enable x64.
        """
        self._state.env.jax_enable_x64 = value

    @property
    def op_determinism(self) -> bool:
        """Get op_determinism.

        Returns:
            bool: True if enabled.
        """
        return self._state.execution.op_determinism

    @op_determinism.setter
    def op_determinism(self, value: bool) -> None:
        """Set operation determinism.

        Args:
            value (bool): True to enable op determinism.
        """
        self._state.execution.op_determinism = value

    @property
    def tensor_float_32_execution(self) -> bool:
        """Get tensor_float_32_execution.

        Returns:
            bool: True if enabled.
        """
        return self._state.execution.tensor_float_32_execution

    @tensor_float_32_execution.setter
    def tensor_float_32_execution(self, value: bool) -> None:
        """Set whether TF32 execution is enabled.

        Args:
            value (bool): True to enable TF32.
        """
        self._state.execution.tensor_float_32_execution = value

    def clear_cache(self) -> None:
        """Clear memory cache. Hook for backends like MLX."""
        from ml_switcheroo_compiler.core.device import clear_cache

        clear_cache()

    @property
    def eager_mode(self) -> bool:
        """Check if eager mode is active.

        Returns:
            bool: True if eager mode is active.
        """
        if "ml_switcheroo_compiler.tracing.state" in sys.modules:
            global_tracing_state = sys.modules["ml_switcheroo_compiler.tracing.state"].global_tracing_state
            if global_tracing_state.is_tracing:
                return False
        if os.environ.get("SWITCHEROO_EAGER_MODE") == "1":
            return True
        return self._state.execution.eager_mode

    @eager_mode.setter
    def eager_mode(self, value: bool) -> None:
        """Eager mode.

        Args:
            value (bool): The value to set or add.
        """
        self._state.execution.eager_mode = value

    def clone(self) -> ConfigState:
        """Clone the current configuration.

        Returns:
            ConfigState: The cloned configuration state.
        """
        return self._state.clone()


# Singleton instance proxy
config = Config()


@contextmanager
def ConfigContext(**kwargs: object) -> Iterator[None]:
    """Provide context manager for temporarily overriding global configuration values.

    Args:
        **kwargs (object): Configuration keys and the values to temporarily set.

    Yields:
        None: Yields control to the enclosed block with the overridden configuration.

    Raises:
        ValueError: If any of the provided keys do not exist in the configuration.
    """
    old_state = _config_state_var.get()
    new_state = old_state.clone()

    for k, v in kwargs.items():
        if hasattr(new_state.execution, k):
            setattr(new_state.execution, k, v)
        elif hasattr(new_state.env, k):
            setattr(new_state.env, k, v)
        else:
            msg = f"Unknown config key: {k}"
            raise ValueError(msg)

    token = _config_state_var.set(new_state)
    try:
        yield
    finally:
        _config_state_var.reset(token)


def EagerMode() -> Iterator[None]:
    """Provide context manager to temporarily enable eager execution mode.

    Returns:
        Iterator[None]: A context manager with eager mode enabled.
    """
    return ConfigContext(eager_mode=True)


def StreamContext(stream_name: str) -> Iterator[None]:
    """Provide context manager to temporarily switch the current execution stream.

    Args:
        stream_name (str): The name of the stream to execute on.

    Returns:
        Iterator[None]: A context manager with the specified stream active.
    """
    return ConfigContext(current_stream=stream_name)


def disable_compile() -> None:
    """Disable compilation globally (enables eager mode)."""
    config.eager_mode = True


def enable_compile() -> None:
    """Enable compilation globally (disables eager mode)."""
    config.eager_mode = False


def enable_op_determinism() -> None:
    """Enable operation determinism."""
    config.op_determinism = True


def enable_tensor_float_32_execution(enabled: bool) -> None:
    """Enable or disable TF32 execution.

    Args:
        enabled (bool): True to enable.
    """
    config.tensor_float_32_execution = enabled


def tensor_float_32_execution_enabled() -> bool:
    """Check if TF32 execution is enabled.

    Returns:
        bool: True if enabled.
    """
    return config.tensor_float_32_execution
