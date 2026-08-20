import os

import yaml


def test_webrtc_topology():
    yaml_path = "src/ml_switcheroo_compiler/distributed/webrtc_topology.yaml"
    assert os.path.exists(yaml_path)
    with open(yaml_path) as f:
        data = yaml.safe_load(f)
    assert "webrtc_config" in data
    assert "templates" in data
