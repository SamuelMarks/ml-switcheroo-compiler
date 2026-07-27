"""Tests for config and device additions."""

from ml_switcheroo_compiler.core.config import (
    config,
    enable_op_determinism,
    enable_tensor_float_32_execution,
    tensor_float_32_execution_enabled,
)
from ml_switcheroo_compiler.core.device import DeviceType, get_logical_devices, get_memory_info, get_physical_devices


def test_config_flags() -> None:
    """Test global runtime flags."""
    enable_op_determinism()
    assert config.op_determinism is True

    enable_tensor_float_32_execution(False)
    assert not tensor_float_32_execution_enabled()
    assert config.tensor_float_32_execution is False

    enable_tensor_float_32_execution(True)
    assert tensor_float_32_execution_enabled()
    assert config.tensor_float_32_execution is True


def test_device_queries() -> None:
    """Test device queries."""
    log_devs = get_logical_devices("cpu")
    assert len(log_devs) == 1
    assert log_devs[0].device_type == DeviceType.CPU

    log_devs_all = get_logical_devices()
    assert len(log_devs_all) == 1

    phys_devs = get_physical_devices("cpu")
    assert len(phys_devs) == 1
    assert phys_devs[0].device_type == DeviceType.CPU

    phys_devs_all = get_physical_devices()
    assert len(phys_devs_all) == 1

    mem_info = get_memory_info()
    assert "current" in mem_info
    assert "peak" in mem_info
    assert mem_info["current"] == 0


def test_clear_cache():
    """Test."""
    from ml_switcheroo_compiler.core.config import config
    from ml_switcheroo_compiler.core.device import clear_cache

    clear_cache()
    config.clear_cache()


def test_function_exporter():
    """Test."""
    from ml_switcheroo_compiler.core.device import FunctionExporter, export_function

    with FunctionExporter():
        pass
    with exporter():
        pass
    export_function()


import pytest

from ml_switcheroo_compiler.core.config import disable_compile, enable_compile
from ml_switcheroo_compiler.core.device import Device, Stream, StreamContext, exporter


def test_stream_context_extra():
    """Test."""
    stream = Stream(Device(DeviceType.CPU))
    with StreamContext(stream):
        pass
    assert repr(Device(DeviceType.CPU)) == "Device(cpu:0)"


def test_config_methods_extra():
    """Test."""
    config.seed = 42
    assert config.seed == 42

    config.backend = "numpy"
    assert config.backend == "numpy"

    config.current_stream = "main"
    assert config.current_stream == "main"

    config.layout_map = "layout"
    assert config.layout_map == "layout"

    config.jax_enable_x64 = True
    assert config.jax_enable_x64

    config.op_determinism = True
    assert config.op_determinism

    config.tensor_float_32_execution = True
    assert config.tensor_float_32_execution

    disable_compile()
    assert config.eager_mode

    enable_compile()
    assert not config.eager_mode

    enable_op_determinism()
    assert config.op_determinism

    enable_tensor_float_32_execution(False)
    assert not config.tensor_float_32_execution
    assert not tensor_float_32_execution_enabled()


def test_eager_mode_with_tracing_state_extra():
    """Test."""
    config.eager_mode = True
    from ml_switcheroo_compiler.tracing.state import global_tracing_state

    orig = global_tracing_state.is_tracing
    global_tracing_state.is_tracing = True
    assert not config.eager_mode

    global_tracing_state.is_tracing = False
    assert config.eager_mode
    global_tracing_state.is_tracing = orig


def test_config_context_exceptions_extra():
    """Test."""
    from ml_switcheroo_compiler.core.config import ConfigContext, EagerMode
    from ml_switcheroo_compiler.core.config import StreamContext as ConfigStreamContext

    with pytest.raises(ValueError):
        with ConfigContext(unknown_key="val"):
            pass
    with EagerMode():
        pass
    with ConfigStreamContext("stream_name"):
        pass


def test_clear_cache_try_except_extra():
    """Test."""
    import sys

    from ml_switcheroo_compiler.core.device import clear_cache

    orig = sys.modules.get("ml_switcheroo_compiler.backends.registry")
    sys.modules["ml_switcheroo_compiler.backends.registry"] = type("Mock", (), {"get_active_backend": lambda: type("Mock", (), {"clear_cache": lambda: 1 / 0})()})
    clear_cache()
    if orig:
        sys.modules["ml_switcheroo_compiler.backends.registry"] = orig
    else:
        sys.modules.pop("ml_switcheroo_compiler.backends.registry")


def test_device_functions_extra():
    """Test."""
    devs = get_logical_devices("gpu")
    assert devs[0].device_type == DeviceType.GPU
    devs = get_physical_devices("gpu")
    assert devs[0].device_type == DeviceType.GPU
    info = get_memory_info("gpu")
    assert info["current"] == 0


def test_config_missing_lines():
    """Test."""
    from ml_switcheroo_compiler.core.config import config
    from ml_switcheroo_compiler.core.device import Device, DeviceType
    from ml_switcheroo_compiler.core.dtype import DType

    config.default_float_dtype = DType.Float32
    config.default_int_dtype = DType.Int64
    config.default_device = Device(DeviceType.CPU)
    assert config.default_float_dtype == DType.Float32
    assert config.default_int_dtype == DType.Int64
    assert config.default_device == Device(DeviceType.CPU)
    cloned = config.clone()
    assert cloned.env.default_float_dtype == DType.Float32


def test_config_context_env():
    """Test."""
    from ml_switcheroo_compiler.core.config import ConfigContext, config

    with ConfigContext(jax_enable_x64=True):
        assert config.jax_enable_x64
