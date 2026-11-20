"""Tests for test_mlx_eager_distributed."""

from __future__ import annotations

import sys
import typing
from unittest.mock import MagicMock, patch

from tests.testing_utils import extract_test_shape

if typing.TYPE_CHECKING:
    pass


def test_mlx_eager_distributed_all_reduce_fallback() -> None:
    """Test function test_mlx_eager_distributed_all_reduce_fallback."""
    mock_mx = MagicMock(spec=["expand_dims", "array"])
    mock_tensor = MagicMock()
    mock_tensor.tolist.return_value = [1.0]

    with patch.dict(sys.modules, {"mlx.core": mock_mx}):
        from ml_switcheroo_compiler.backends.mlx.eager import _mlx_all_reduce

        res = _mlx_all_reduce(mock_mx, mock_tensor)
        assert res is mock_tensor
        assert mock_tensor.tolist() == [1.0]


def test_mlx_eager_distributed_all_gather_fallback() -> None:
    """Test function test_mlx_eager_distributed_all_gather_fallback."""
    mock_mx = MagicMock(spec=["expand_dims", "array"])
    mock_tensor = MagicMock()
    mock_gathered = MagicMock()
    mock_gathered.shape = (1, 1)
    mock_mx.expand_dims.return_value = mock_gathered

    with patch.dict(sys.modules, {"mlx.core": mock_mx}):
        from ml_switcheroo_compiler.backends.mlx.eager import _mlx_all_gather

        res = _mlx_all_gather(mock_mx, mock_tensor)
        assert extract_test_shape(res) == (1, 1)
        mock_mx.expand_dims.assert_called_once_with(mock_tensor, axis=0)


def test_mlx_eager_distributed_all_to_all() -> None:
    """Test function test_mlx_eager_distributed_all_to_all."""
    mock_mx = MagicMock()
    mock_tensor = MagicMock()
    mock_dist = MagicMock()
    mock_dist.all_to_all.return_value = "mocked_all_to_all"
    mock_mx.distributed = mock_dist
    mock_mx.array.return_value = mock_tensor

    with patch.dict(sys.modules, {"mlx.core": mock_mx}):
        from ml_switcheroo_compiler.backends.mlx.eager import _mlx_all_to_all

        res = _mlx_all_to_all(mock_mx, mock_mx.array([1.0]))
        assert res == "mocked_all_to_all"
