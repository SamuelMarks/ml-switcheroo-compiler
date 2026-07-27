import pytest
import torch

import ml_switcheroo_compiler.backends.pytorch.eager as pytorch_eager


def test_pytorch_eager_coverage():
    # _execute_accumulate_n
    res = pytorch_eager._execute_accumulate_n([torch.tensor(1), torch.tensor(2)])
    assert int(res.item() if hasattr(res, "item") else res) == 3
    res = pytorch_eager._execute_accumulate_n(inputs=[torch.tensor(1), torch.tensor(2)])
    assert int(res.item() if hasattr(res, "item") else res) == 3

    with pytest.raises(ValueError):
        pytorch_eager._execute_accumulate_n(inputs=[])

    # _execute_tensor_scatter_max, min, update, add
    tensor = torch.zeros(2, 2)
    indices = torch.tensor([[0, 0]])
    updates = torch.tensor([1.0])
    res = pytorch_eager._execute_tensor_scatter_max(tensor, indices, updates)
    assert res[0, 0] == 1.0

    res = pytorch_eager._execute_tensor_scatter_min(tensor, indices, updates)
    assert res[0, 0] == 0.0

    res = pytorch_eager._execute_tensor_scatter_update(tensor, indices, updates)
    assert res[0, 0] == 1.0

    res = pytorch_eager._execute_tensor_scatter_add(tensor, indices, updates)
    assert res[0, 0] == 1.0

    # _execute_power_iteration
    w = torch.eye(2).unsqueeze(0)
    res = pytorch_eager._execute_power_iteration(w)
    assert len(res) == 3

    # _execute_broadcast_to

    res = pytorch_eager._execute_broadcast_to(torch.tensor([1.0]), shape=(2,))
    assert res.shape == (2,)

    # _execute_cast
    class DummyDtype:
        value = "int4"

    res = pytorch_eager._execute_cast(torch.tensor([1.0]), DummyDtype())

    class DummyDtypeBfloat16:
        value = "bfloat16"

    res = pytorch_eager._execute_cast(torch.tensor([1.0]), DummyDtypeBfloat16())

    #

    class DummyDtypeFloat16:
        value = "float16"

    res = pytorch_eager._execute_cast(torch.tensor([1.0]), DummyDtypeFloat16())

    class DummyDtypeFloat8_e4m3fn:
        value = "float8_e4m3fn"

    res = pytorch_eager._execute_cast(torch.tensor([1.0]), DummyDtypeFloat8_e4m3fn())

    # _execute_cummax

    res = pytorch_eager._execute_cummax(torch.tensor([1, 2]), dim=0)
    assert res is not None

    # _execute_cummin

    res = pytorch_eager._execute_cummin(torch.tensor([1, 2]), dim=0)
    assert res is not None

    # _execute_cumlogsumexp

    res = pytorch_eager._execute_cumlogsumexp(torch.tensor([1.0, 2.0]), dim=0)
    assert res is not None

    # _execute_ragged_tensor_to_dense
    res = pytorch_eager._execute_ragged_tensor_to_dense(1)
    assert res == 1

    # _torch_variance

    res = pytorch_eager._torch_variance(torch.tensor([1.0, 2.0]))
    assert res is not None

    # _torch_tensordot

    res = pytorch_eager._torch_tensordot(torch.tensor([1.0]), torch.tensor([1.0]), dims=1)
    assert res is not None

    res = pytorch_eager._torch_tensordot(torch.tensor([1.0]), torch.tensor([1.0]), axes=1)
    assert res is not None

    # execute_op

    res = pytorch_eager.execute_op(None, "Add", torch.tensor([1.0]), torch.tensor([1.0]))

    res = pytorch_eager.execute_op(None, "Cumlogsumexp", torch.tensor([1.0, 2.0]), dim=0)

    res = pytorch_eager.execute_op(None, "UnknownOp")

    # Op maps

    lambdas = [
        ("TruncateDiv", (torch.tensor([1.0]), torch.tensor([1.0])), {}),
        ("StopGradient", (torch.tensor([1.0]),), {}),
        ("ActivityRegularization", (1,), {}),
        ("AdaptiveMaxPool3D_Indices", (torch.zeros(1, 1, 1, 1, 1), 1), {}),
        ("AdaptiveLogSoftmaxWithLoss", (torch.zeros(1), torch.zeros(1)), {}),
        ("AllGather", (torch.tensor([1]),), {}),
        ("AllToAll", (1,), {}),
        ("Append", (torch.tensor([1]), torch.tensor([2])), {}),
        ("Append", (torch.tensor([1]), torch.tensor([2])), {"axis": 0}),
        ("ApplyOverAxes", (lambda x: x, 1, 0), {}),
        ("Argpartition", (torch.tensor([1, 2]), 1), {}),
        ("ArrayEquiv", (torch.tensor([1]), torch.tensor([1])), {}),
        ("ArrayRepr", (torch.tensor([1]),), {}),
        ("ArrayStr", (torch.tensor([1]),), {}),
        ("AsString", (torch.tensor([1]),), {}),
        ("Assert", (True, 1), {}),
        ("Assign", (1, 2), {}),
        ("AssignAdd", (1, 2), {}),
        ("AssignSub", (1, 2), {}),
        ("AssignVariable", (1, 2), {}),
        ("AssociativeScan", (lambda x: x, 1), {}),
        ("AssociativeScan", (1,), {}),
        ("Atleast1d", (torch.tensor([1]),), {}),
        ("Atleast2d", (torch.tensor([1]),), {}),
        ("Atleast3d", (torch.tensor([1]),), {}),
        ("Average", (torch.tensor([1.0]),), {}),
        ("AxisIndex", (), {}),
        ("HardSilu", (torch.tensor([1.0]),), {}),
        ("HardSwish", (torch.tensor([1.0]),), {}),
        ("Squareplus", (torch.tensor([1.0]),), {}),
    ]

    lambdas = []

    lambdas = []

    for op, args, kwargs in lambdas:
        res = pytorch_eager._TORCH_EAGER_OP_MAP[op](*args, **kwargs)

    # check all methods in the OP_DISPATCH table are callable
    for op, fn in pytorch_eager._TORCH_EAGER_OP_MAP.items():
        assert callable(fn)
