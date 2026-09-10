import pytest
from ml_switcheroo_ir import LogicalGraph

from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.ops.control_flow_utils import _get_tensor_ids, _process_trace_outputs, _trace_function, _wrap_proxy_inputs


class DummyData:
    id = "id"


def test_control_flow_utils():
    t_cfg = TensorConfig((1,), DType.Float32, "cpu")
    t = Tensor(data=DummyData(), config=t_cfg)

    g = LogicalGraph("g")
    # wrap proxy inputs
    ids, args = _wrap_proxy_inputs(((t,), 42), g)
    assert len(ids) == 1
    assert args[1] == 42
    assert isinstance(args[0], tuple)

    # get tensor ids
    assert _get_tensor_ids(t) == ["id"]
    assert _get_tensor_ids([t]) == ["id"]

    # Nested Tensor unwrap (line 78)
    t_inner = Tensor(data=DummyData(), config=t_cfg)
    t_outer = Tensor(data=t_inner, config=t_cfg)
    assert _get_tensor_ids(t_outer) == ["id"]

    # Direct id on Tensor (lines 81-82)
    class DirectIdTensor(Tensor):
        pass

    t_direct = DirectIdTensor(data="no_id", config=t_cfg)
    t_direct.id = "direct_123"
    assert _get_tensor_ids(t_direct) == ["direct_123"]

    # No id anywhere (line 83)
    t_no_id = Tensor(data="no_id", config=t_cfg)
    if hasattr(t_no_id, "id"):
        delattr(t_no_id, "id")
    generated_ids = _get_tensor_ids(t_no_id)
    assert len(generated_ids) == 1
    assert len(generated_ids[0]) > 0

    with pytest.raises(TypeError):
        _get_tensor_ids(42)

    # process trace outputs
    g2 = LogicalGraph("g2")
    out_id = _process_trace_outputs(t, g2)
    assert out_id in g2.nodes

    # trace function
    def my_fn(x):
        return x

    block = _trace_function(my_fn, (t,), "my_block")
    assert block.id == "my_block"
    assert len(block.inputs) == 1
    assert len(block.outputs) == 1
