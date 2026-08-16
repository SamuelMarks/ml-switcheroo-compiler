from ml_switcheroo_compiler.distributed.config_models import MeshMappingConfig, MicrobatchSplittingConfig, PipelineTopologiesConfig, StageCommunicationConfig


def test_pipeline_config_model():
    data = {"default": {"microbatch_splitting": {"num_microbatches": 4, "strategy": "chunk"}, "mesh_mapping": {"devices_per_stage": 1}, "stage_communication": {"protocol": "p2p_queue"}}}
    config = PipelineTopologiesConfig(root=data)
    default_config = config.get("default")
    assert default_config.microbatch_splitting.num_microbatches == 4
    assert default_config.stage_communication.protocol == "p2p_queue"

    assert list(config.items())[0][0] == "default"
    assert config.model_dump()["default"]["mesh_mapping"]["devices_per_stage"] == 1
    assert config.dict()["default"]["mesh_mapping"]["devices_per_stage"] == 1
    assert config.get("missing") is None

    assert MicrobatchSplittingConfig(num_microbatches=1, strategy="strat").num_microbatches == 1
    assert MeshMappingConfig(devices_per_stage=2).devices_per_stage == 2
    assert StageCommunicationConfig(protocol="p").protocol == "p"
