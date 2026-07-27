# ruff: noqa: E501
import sys
from unittest.mock import patch

import numpy as np

from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.ops.linalg.einsum_frontend import _get_remaining_dims, _infer_dot_general_shape, einsum, tensordot


def test_tensordot_eager_trace() -> None:
    t = Tensor(np.zeros((2, 2)), TensorConfig((2, 2), "float32", "cpu"))
    config.eager_mode = True
    with patch("ml_switcheroo_compiler.backends.registry.get_active_backend") as mock_backend:
        mock_backend.return_value.execute_op.return_value = np.zeros((2, 2))
        res = tensordot(t, t, axes=1)
        assert isinstance(res, Tensor)
    config.eager_mode = False
    mod = sys.modules["ml_switcheroo_compiler.ops.linalg.einsum_frontend"]
    with patch.object(mod, "_emit_linalg_node") as mock_emit:
        mock_emit.return_value = "mock"
        assert tensordot(t, t, axes=1) == "mock"


def test_tensordot_routing() -> None:
    t = Tensor(np.zeros((2, 2, 2)), TensorConfig((2, 2, 2), "float32", "cpu"))
    config.eager_mode = True
    with patch("ml_switcheroo_compiler.backends.registry.get_active_backend") as mock_backend:
        mock_backend.return_value.execute_op.return_value = np.zeros((2, 2))
        res = tensordot(t, t, axes=([1], [0]))
        assert isinstance(res, Tensor)


def test_einsum_eager_trace() -> None:
    t = Tensor(np.zeros((2, 2)), TensorConfig((2, 2), "float32", "cpu"))
    config.eager_mode = True
    with patch("ml_switcheroo_compiler.backends.registry.get_active_backend") as mock_backend:
        mock_backend.return_value.execute_op.return_value = np.zeros((2, 2))
        res = einsum("ij,jk->ik", t, t)
        assert isinstance(res, Tensor)
    config.eager_mode = False
    mod = sys.modules["ml_switcheroo_compiler.ops.linalg.einsum_frontend"]
    with patch.object(mod, "_emit_linalg_node") as mock_emit:
        mock_emit.return_value = "mock"
        assert einsum("ij,jk->ik", t, t) == "mock"


def test_helpers() -> None:
    assert _get_remaining_dims(3, [0], [1]) == [2]
    out_shape = _infer_dot_general_shape((2, 3, 4), (2, 4, 5), (((2,), (1,)), ((0,), (0,))))
    assert out_shape == (2, 3, 5)


def test_generate_tensordot_einsum_strings_empty() -> None:
    from ml_switcheroo_compiler.ops.linalg.einsum_frontend import _generate_tensordot_einsum_strings

    assert _generate_tensordot_einsum_strings((), (), (), ()) == ("", "", "")
