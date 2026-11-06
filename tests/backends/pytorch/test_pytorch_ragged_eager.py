"""Tests for PyTorch ragged eager operations."""

from __future__ import annotations

import typing

import numpy as np
import pytest

from ml_switcheroo_compiler.backends.pytorch.eager import _execute_ragged_tensor_to_dense
from tests.testing_utils import ensure_eager_tensor

if typing.TYPE_CHECKING:
    import torch

torch = pytest.importorskip("torch")


def test_pytorch_ragged_eager() -> None:
    """Test PyTorch ragged tensor to dense conversion branches."""
    a = [torch.tensor([1, 2]), torch.tensor([1])]
    res = _execute_ragged_tensor_to_dense(a)
    assert ensure_eager_tensor(res).shape == (2, 2)
    assert _execute_ragged_tensor_to_dense(np.array([1, 2])) is not None

    # Test dictionary with values and row_splits including empty row (length == 0 branch)
    ragged_dict = {
        "values": torch.tensor([10.0, 20.0, 30.0]),
        "row_splits": torch.tensor([0, 2, 2, 3]),
    }
    dense = _execute_ragged_tensor_to_dense(ragged_dict, default_value=-1.0)
    assert ensure_eager_tensor(dense).shape == (3, 2)
    assert (
        ensure_eager_tensor(dense)[
            (
                0,
                0,
            )
        ]
        == 10.0
    )
    assert (
        ensure_eager_tensor(dense)[
            (
                0,
                1,
            )
        ]
        == 20.0
    )
    assert (
        ensure_eager_tensor(dense)[
            (
                1,
                0,
            )
        ]
        == -1.0
    )
    assert (
        ensure_eager_tensor(dense)[
            (
                1,
                1,
            )
        ]
        == -1.0
    )
    assert (
        ensure_eager_tensor(dense)[
            (
                2,
                0,
            )
        ]
        == 30.0
    )
    assert (
        ensure_eager_tensor(dense)[
            (
                2,
                1,
            )
        ]
        == -1.0
    )
