"""Test module."""

import pytest

from ml_switcheroo_compiler.core.config import config, disable_compile, enable_compile, enable_op_determinism, enable_tensor_float_32_execution, tensor_float_32_execution_enabled


def test_config_extras():
    config.seed = 42
    assert config.seed == 42

    config.op_determinism = False
    enable_op_determinism()
    assert config.op_determinism is True

    config.tensor_float_32_execution = False
    enable_tensor_float_32_execution(True)
    assert tensor_float_32_execution_enabled() is True

    disable_compile()
    assert config.eager_mode is True
    enable_compile()
    assert config.eager_mode is False

    # Trigger the tracing check by mocking
    import sys

    class MockTracingState:
        is_tracing = True

    class MockStateMod:
        global_tracing_state = MockTracingState()

    orig_state = sys.modules.get("ml_switcheroo_compiler.tracing.state")
    sys.modules["ml_switcheroo_compiler.tracing.state"] = MockStateMod()
    config.eager_mode = True  # Even if true, is_tracing forces false
    assert config.eager_mode is False

    MockTracingState.is_tracing = False
    assert config.eager_mode is True

    if orig_state is not None:
        sys.modules["ml_switcheroo_compiler.tracing.state"] = orig_state
    else:
        del sys.modules["ml_switcheroo_compiler.tracing.state"]


def test_config_more_props():
    from ml_switcheroo_compiler.core.device import Device, DeviceType
    from ml_switcheroo_compiler.core.dtype import DType

    config.backend = "mlx"
    assert config.backend == "mlx"

    config.default_float_dtype = DType.Float16
    assert config.default_float_dtype == DType.Float16

    config.default_int_dtype = DType.Int32
    assert config.default_int_dtype == DType.Int32

    d = Device(DeviceType.GPU, 0)
    config.default_device = d
    assert config.default_device == d

    config.current_stream = "my_stream"
    assert config.current_stream == "my_stream"

    config.layout_map = "layout"
    assert config.layout_map == "layout"

    config.jax_enable_x64 = True
    assert config.jax_enable_x64 is True

    c2 = config.clone()
    assert c2.execution.backend == "mlx"


def test_config_clear_cache(monkeypatch):
    import ml_switcheroo_compiler.core.device as dev

    monkeypatch.setattr(dev, "clear_cache", lambda: None)
    config.clear_cache()


def test_config_context_unknown():
    from ml_switcheroo_compiler.core.config import ConfigContext

    with pytest.raises(ValueError, match="Unknown config key"):
        with ConfigContext(unknown_key=True):
            pass


def test_eager_mode_stream_context():
    from ml_switcheroo_compiler.core.config import EagerMode, StreamContext

    with EagerMode():
        assert config.eager_mode is True
    with StreamContext("test"):
        assert config.current_stream == "test"


def test_config_context_env():
    from ml_switcheroo_compiler.core.config import ConfigContext

    with ConfigContext(jax_enable_x64=False):
        assert config.jax_enable_x64 is False


def test_config_eager_mode_branch():
    import sys

    from ml_switcheroo_compiler.core.config import config

    class FakeTracingState:
        is_tracing = False

    class FakeMod:
        global_tracing_state = FakeTracingState()

    old = sys.modules.get("ml_switcheroo_compiler.tracing.state")
    sys.modules["ml_switcheroo_compiler.tracing.state"] = FakeMod()

    config.eager_mode = True
    assert config.eager_mode is True

    if old:
        sys.modules["ml_switcheroo_compiler.tracing.state"] = old
    else:
        del sys.modules["ml_switcheroo_compiler.tracing.state"]


def test_config_eager_mode_no_module():
    import sys

    from ml_switcheroo_compiler.core.config import config

    old = sys.modules.get("ml_switcheroo_compiler.tracing.state")
    if old:
        del sys.modules["ml_switcheroo_compiler.tracing.state"]

    config.eager_mode = True
    assert config.eager_mode is True

    if old:
        sys.modules["ml_switcheroo_compiler.tracing.state"] = old
