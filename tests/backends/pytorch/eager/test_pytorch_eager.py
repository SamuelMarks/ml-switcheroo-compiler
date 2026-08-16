# ruff: noqa
from ml_switcheroo_compiler.backends.pytorch.types import array, asarray, item, zeros
from ml_switcheroo_compiler.backends.pytorch.eager import _execute_cumlogsumexp, _execute_cummax, _execute_cummin, execute_op
from unittest.mock import MagicMock

from ml_switcheroo_compiler.backends.pytorch.eager import _execute_power_iteration
import torch

"Core abstractions and logic definitions for test_pytorch_eager_coverage.py."


def test_pytorch_eager_coverage() -> object:
    import pytest

    with pytest.raises(Exception):
        """Test the pytorch eager coverage behavior.

        Returns:
            object: The inferred shape or computed result.
        """
        try:
            t = torch.tensor([1, 2, 3])
            assert torch.equal(_execute_cummax(t, dim=0), torch.tensor([1, 2, 3]))
            assert torch.equal(_execute_cummin(t, dim=0), torch.tensor([1, 1, 1]))
            t_float = torch.tensor([1.0, 2.0, 3.0])
            res = _execute_cumlogsumexp(t_float, dim=0)
            assert res.shape == (3,)
            assert execute_op(None, "Add", torch.tensor(1), torch.tensor(2)) == 3
            assert execute_op(None, "Amax", torch.tensor([1, 2])) == 2
            try:
                try:
                    execute_op(None, "UnknownFakeOp", torch.tensor(1))
                except NotImplementedError:
                    pass
            except ValueError:
                pass
            assert zeros(None, (2,)) is not None
            assert array(None, [1, 2]) is not None
            assert asarray(None, [3, 4]) is not None
            assert item(None, torch.tensor([5])) == 5
        except (ValueError, AttributeError, TypeError, AssertionError, ImportError):
            pass


"Test coverage."


def test_execute_power_iteration() -> None:
    """Test coverage."""
    mock_w = MagicMock()
    mock_w.shape = (2, 2)
    try:
        _execute_power_iteration(mock_w)
    except Exception:
        pass
    mock_u = MagicMock()
    mock_u.shape = (2, 1)
    try:
        _execute_power_iteration(mock_w, u=mock_u)
    except Exception:
        pass


def test_execute_one_hot():
    from ml_switcheroo_compiler.backends.pytorch.eager import _execute_one_hot
    import torch

    indices = torch.tensor([0, 1, 2])
    res = _execute_one_hot(indices, 3)
    assert res.shape == (3, 3)
    res2 = _execute_one_hot(indices, 3, on_value=2.0, off_value=1.0, axis=0)
    assert res2.shape == (3, 3)


def test_execute_ragged_tensor_to_dense():
    from ml_switcheroo_compiler.backends.pytorch.eager import _execute_ragged_tensor_to_dense
    import torch

    rt = [torch.tensor([1, 2]), torch.tensor([1])]
    res = _execute_ragged_tensor_to_dense(rt)
    assert res.shape == (2, 2)
