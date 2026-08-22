from ml_switcheroo_compiler.grad.config_models import FiniteDifferenceConfig, FiniteDifferenceDtypeConfig


def test_finite_difference_config():
    f32_cfg = FiniteDifferenceDtypeConfig(epsilon=1e-5)
    f64_cfg = FiniteDifferenceDtypeConfig(epsilon=1e-9)
    cfg = FiniteDifferenceConfig(float32=f32_cfg, float64=f64_cfg)
    assert cfg.float32.epsilon == 1e-5
    assert cfg.float64.epsilon == 1e-9
