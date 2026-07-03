"""Module docstring."""

from unittest.mock import patch

import numpy as np

from ml_switcheroo_compiler.core.config import ConfigContext
from ml_switcheroo_compiler.core.device import Device
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.ops.distributed import (
    AllToAll,
    AxisIndex,
    Infeed,
    Outfeed,
    Pbroadcast,
    Pdot,
    Pmax,
    Pmin,
    Ppermute,
    Pshuffle,
    PsumScatter,
    Pswapaxes,
    WithShardingConstraint,
    axis_index,
    infeed,
    outfeed,
    with_sharding_constraint,
)


def test_distributed_infer_shapes() -> object:
    """Function docstring."""
    device = Device("cpu")
    t1 = Tensor(np.ones((2,)), TensorConfig((2,), "float32", device))

    assert Pmax().infer_shape() == ()
    assert Pmin().infer_shape() == ()
    assert PsumScatter().infer_shape() == ()
    assert Pswapaxes().infer_shape() == ()
    assert Pbroadcast().infer_shape(t1) == (2,)
    assert Pdot().infer_shape() == ()
    assert Ppermute().infer_shape(t1) == (2,)
    assert Pshuffle().infer_shape(t1) == (2,)
    assert Infeed().infer_shape(shape=(3, 3)) == (3, 3)
    assert Outfeed().infer_shape() == ()
    assert AxisIndex().infer_shape() == ()
    assert WithShardingConstraint().infer_shape(t1) == (2,)
    assert AllToAll().infer_shape() == ()


def test_distributed_helpers() -> object:
    """Function docstring."""
    with ConfigContext(eager_mode=True):
        device = Device("cpu")
        t1 = Tensor(np.ones((2,)), TensorConfig((2,), "float32", device))

        with patch("ml_switcheroo_compiler.ops.distributed._emit_shape_node") as mock_emit:
            infeed((2, 2), "float32")
            outfeed(t1)
            axis_index("batch")
            with_sharding_constraint(t1, "sharding")

            assert mock_emit.call_count == 4
