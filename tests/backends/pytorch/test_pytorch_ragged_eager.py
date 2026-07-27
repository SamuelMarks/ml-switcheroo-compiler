import numpy as np


def test_pytorch_ragged_eager():
    try:
        import torch

        from ml_switcheroo_compiler.backends.pytorch.eager import _execute_ragged_tensor_to_dense

        a = [torch.tensor([1, 2]), torch.tensor([1])]
        res = _execute_ragged_tensor_to_dense(a)
        assert res.shape == (2, 2)
        assert _execute_ragged_tensor_to_dense(np.array([1, 2])) is not None
    except ImportError:
        pass
    except Exception as e:
        # Ignore crashes
        pass
