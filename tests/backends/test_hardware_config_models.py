def test_hardware_config_models_dump():
    from ml_switcheroo_compiler.backends.hardware_config_models import HardwareTemplateConfig, HardwareTemplatesConfig

    cfg = HardwareTemplatesConfig(templates={"a": HardwareTemplateConfig(body="body", workgroup_size=[1, 2, 3])}, orchestration={"k": "v"})
    dumped = cfg.model_dump()
    assert dumped["templates"]["a"]["body"] == "body"
    assert dumped["templates"]["a"]["workgroup_size"] == [1, 2, 3]
    assert dumped["orchestration"]["k"] == "v"
