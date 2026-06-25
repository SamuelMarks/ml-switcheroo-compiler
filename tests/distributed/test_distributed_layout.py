import numpy as np

from ml_switcheroo_compiler.backends.numpy.distributed.dummy import _dummy_all_gather
from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.distributed import DeviceMesh, LayoutMap, ShardingSpec


def test_distributed_layout():
    # Setup DeviceMesh and LayoutMap
    mesh = DeviceMesh(shape=(2, 4), axis_names=("data", "model"))
    spec = ShardingSpec(mesh, ("data", None))

    layout_map = LayoutMap()
    layout_map.insert("dense/kernel", spec)

    # Store in config
    config.layout_map = layout_map

    # IR validation
    # Check if the mock dummy works
    tensor = np.ones((10, 10))
    gathered = _dummy_all_gather(tensor, 0, mesh)

    np.testing.assert_allclose(tensor, gathered)

    assert config.layout_map.get("dense/kernel") == spec
    assert config.layout_map.get("other/kernel") is None
