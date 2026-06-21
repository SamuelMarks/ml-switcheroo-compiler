"""Test indexing ops."""

from unittest.mock import MagicMock

import numpy as np

from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.device import Device
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.ops.shape.indexing import (
    gather,
    gather_nd,
    scatter,
    scatter_add,
    scatter_nd,
    searchsorted,
    select,
    take,
    take_along_axis,
    where,
)
from ml_switcheroo_compiler.tracing.tracer import _tracer


def test_indexing_eager() -> None:
    """Test indexing eager."""
    config.eager_mode = True
    config.backend = "numpy"

    x = Tensor(
        np.arange(12).reshape(2, 3, 2).astype(np.float32),
        TensorConfig((2, 3, 2), DType.Float32, Device("cpu")),
    )
    indices = Tensor(np.array([[0, 1], [1, 2]]), TensorConfig((2, 2), DType.Int32, Device("cpu")))

    g = gather_nd(x, indices)
    assert g.shape == (2, 2)
    assert g.data[0, 0] == 2.0  # x[0, 1, 0]

    updates = Tensor(
        np.array([[9, 9], [8, 8]], dtype=np.float32),
        TensorConfig((2, 2), DType.Float32, Device("cpu")),
    )
    s_nd = scatter_nd(indices, updates, (2, 3, 2))
    assert s_nd.shape == (2, 3, 2)
    assert s_nd.data[0, 1, 0] == 9.0

    # scatter
    input_tensor = Tensor(
        np.zeros((2, 3), dtype=np.float32), TensorConfig((2, 3), DType.Float32, Device("cpu"))
    )
    idx_tensor = Tensor(
        np.array([[1], [2]], dtype=np.int32), TensorConfig((2, 1), DType.Int32, Device("cpu"))
    )
    src_tensor = Tensor(
        np.array([[9], [8]], dtype=np.float32), TensorConfig((2, 1), DType.Float32, Device("cpu"))
    )
    s = scatter(input_tensor, 1, idx_tensor, src_tensor)
    assert s.shape == (2, 3)
    assert s.data[0, 1] == 9.0

    # scatter_add
    s_add = scatter_add(input_tensor, 1, idx_tensor, src_tensor)
    assert s_add.data[0, 1] == 9.0

    # take
    t = take(input_tensor, Tensor(np.array([1, 2]), TensorConfig((2,), DType.Int32, Device("cpu"))))
    assert t.shape == (2,)

    # take_along_axis
    take_along_axis(input_tensor, idx_tensor, axis=1)

    # searchsorted
    a = Tensor(np.array([1, 2, 3, 4, 5]), TensorConfig((5,), DType.Int32, Device("cpu")))
    v = Tensor(np.array([3]), TensorConfig((1,), DType.Int32, Device("cpu")))
    ss = searchsorted(a, v)
    assert ss.data[0] == 2

    # gather
    gg = gather(input_tensor, 1, idx_tensor)
    assert gg.shape == (2, 1)

    # where
    cond = Tensor(np.array([True, False]), TensorConfig((2,), DType.Bool, Device("cpu")))
    t_true = Tensor(np.array([1, 1]), TensorConfig((2,), DType.Int32, Device("cpu")))
    t_false = Tensor(np.array([0, 0]), TensorConfig((2,), DType.Int32, Device("cpu")))
    w = where(cond, t_true, t_false)
    assert w.data[0] == 1
    assert w.data[1] == 0

    sel = select(cond, t_true, t_false)
    assert sel.data[0] == 1
    assert sel.data[1] == 0


def test_indexing_lazy() -> None:
    """Test indexing lazy."""
    config.eager_mode = False
    _tracer.start_tracing()

    m1 = MagicMock(id="n1")
    m2 = MagicMock(id="n2")
    m3 = MagicMock(id="n3")

    x = Tensor(m1, TensorConfig((2, 3, 2), DType.Float32, Device("cpu")))
    indices = Tensor(m2, TensorConfig((2, 2), DType.Int32, Device("cpu")))

    gather_nd(x, indices)
    assert list(_tracer.active_graph.nodes.values())[-1].op_type == "GatherNd"

    updates = Tensor(m3, TensorConfig((2, 2), DType.Float32, Device("cpu")))
    scatter_nd(indices, updates, (2, 3, 2))
    assert list(_tracer.active_graph.nodes.values())[-1].op_type == "ScatterNd"

    input_tensor = Tensor(m1, TensorConfig((2, 3), DType.Float32, Device("cpu")))
    idx_tensor = Tensor(m2, TensorConfig((2, 1), DType.Int32, Device("cpu")))
    src_tensor = Tensor(m3, TensorConfig((2, 1), DType.Float32, Device("cpu")))

    scatter(input_tensor, 1, idx_tensor, src_tensor)
    assert list(_tracer.active_graph.nodes.values())[-1].op_type == "Scatter"

    scatter_add(input_tensor, 1, idx_tensor, src_tensor)
    assert list(_tracer.active_graph.nodes.values())[-1].op_type == "ScatterAdd"

    take(input_tensor, Tensor(m2, TensorConfig((2,), DType.Int32, Device("cpu"))))
    assert list(_tracer.active_graph.nodes.values())[-1].op_type == "Take"

    a = Tensor(m1, TensorConfig((5,), DType.Int32, Device("cpu")))
    v = Tensor(m2, TensorConfig((1,), DType.Int32, Device("cpu")))
    searchsorted(a, v)
    assert list(_tracer.active_graph.nodes.values())[-1].op_type == "SearchSorted"

    gather(input_tensor, 1, idx_tensor)
    assert list(_tracer.active_graph.nodes.values())[-1].op_type == "Gather"

    cond = Tensor(m1, TensorConfig((2,), DType.Bool, Device("cpu")))
    t_true = Tensor(m2, TensorConfig((2,), DType.Int32, Device("cpu")))
    t_false = Tensor(m3, TensorConfig((2,), DType.Int32, Device("cpu")))
    where(cond, t_true, t_false)
    assert list(_tracer.active_graph.nodes.values())[-1].op_type == "Where"

    _tracer.stop_tracing()
    config.eager_mode = True
