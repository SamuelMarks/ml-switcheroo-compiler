"""Configuration settings and context for the ml-switcheroo compiler."""

import os
from contextlib import contextmanager
from collections.abc import Iterator
from ml_switcheroo.core.dtype import DType
from ml_switcheroo.core.device import Device, DeviceType


class Config:
    """Singleton configuration state for the compiler."""

    def __init__(self) -> None:
        """Initialize the Config."""
        self.eager_mode: bool = os.environ.get("SWITCHEROO_EAGER_MODE", "0") == "1"
        self.default_float_dtype: DType = DType.Float32
        self.default_int_dtype: DType = DType.Int64
        self.default_device: Device = Device(DeviceType.CPU, 0)

    @property
    def eager_mode(self) -> bool:
        """Docstring."""
        from ml_switcheroo.tracing import _tracer

        if _tracer.is_tracing:
            return False
        return self._eager_mode

    @eager_mode.setter
    def eager_mode(self, value: bool) -> None:
        """Docstring."""
        self._eager_mode = value

    def clone(self) -> "Config":
        """Clone the current configuration."""
        new_config = Config()
        new_config._eager_mode = self._eager_mode
        new_config.default_float_dtype = self.default_float_dtype
        new_config.default_int_dtype = self.default_int_dtype
        new_config.default_device = self.default_device
        return new_config


# Singleton instance
config = Config()


@contextmanager
def ConfigContext(**kwargs: object) -> Iterator[None]:
    """Context manager for temporarily overriding config values."""
    old_config = config.clone()
    try:
        for k, v in kwargs.items():
            if hasattr(config, k):
                setattr(config, k, v)
            else:
                raise ValueError(f"Unknown config key: {k}")
        yield
    finally:
        config._eager_mode = old_config._eager_mode
        config.default_float_dtype = old_config.default_float_dtype
        config.default_int_dtype = old_config.default_int_dtype
        config.default_device = old_config.default_device


def EagerMode() -> Iterator[None]:
    """Context manager for eager execution."""
    return ConfigContext(eager_mode=True)
