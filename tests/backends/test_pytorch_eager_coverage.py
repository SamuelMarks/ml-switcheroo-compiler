import torch
from ml_switcheroo_compiler.backends.pytorch.eager import (
    _execute_cummax,
    _execute_cummin,
    _execute_cumlogsumexp,
    execute_op,
)


def test_pytorch_eager_coverage():
    t = torch.tensor([1, 2, 3])

    assert torch.equal(_execute_cummax(t, dim=0), torch.tensor([1, 2, 3]))
    assert torch.equal(_execute_cummin(t, dim=0), torch.tensor([1, 1, 1]))

    t_float = torch.tensor([1.0, 2.0, 3.0])
    res = _execute_cumlogsumexp(t_float, dim=0)
    assert res.shape == (3,)

    assert execute_op(None, "Add", torch.tensor(1), torch.tensor(2)) == 3
    assert execute_op(None, "Amax", torch.tensor([1, 2])) == 2

    try:
        execute_op(None, "UnknownFakeOp", torch.tensor(1))
    except NotImplementedError:
        pass

    from ml_switcheroo_compiler.backends.pytorch.types import zeros, array, asarray, item

    assert zeros(None, (2,)) is not None
    assert array(None, [1, 2]) is not None
    assert asarray(None, [3, 4]) is not None
    assert item(None, torch.tensor([5])) == 5
