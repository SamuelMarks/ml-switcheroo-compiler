# ruff: noqa: E501
import sys
from unittest.mock import patch

import numpy as np

from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.ops.linalg.decompositions.eig import Eig, Eigh, Eigvals, Eigvalsh, eigvals


def test_eig_dummy_methods() -> None:
    assert Eig().infer_shape() == ()
    assert Eigh().infer_shape() == ()
    assert Eigvals().infer_shape() == ()
    assert Eigvalsh().infer_shape() == ()


def test_eigvals() -> None:
    config.eager_mode = True
    input_data = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    t = Tensor(input_data, TensorConfig(input_data.shape, "float32", "cpu"))
    res = eigvals(t)
    assert isinstance(res, Tensor)
    config.eager_mode = False
    eig_mod = sys.modules["ml_switcheroo_compiler.ops.linalg.decompositions.eig"]
    with patch.object(eig_mod, "_emit_linalg_node") as mock_emit:
        mock_emit.return_value = "mock_trace"
        res_trace = eigvals(t)
        assert res_trace == "mock_trace"
        mock_emit.assert_called_once()


def test_eig() -> None:
    from ml_switcheroo_compiler.ops.linalg.decompositions.eig import eig

    config.eager_mode = True
    input_data = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    t = Tensor(input_data, TensorConfig(input_data.shape, "float32", "cpu"))
    res = eig(t)
    assert isinstance(res, tuple)
    config.eager_mode = False
    eig_mod = sys.modules["ml_switcheroo_compiler.ops.linalg.decompositions.eig"]
    with patch.object(eig_mod, "_emit_linalg_node") as mock_emit:
        mock_emit.return_value = ("mock_trace1", "mock_trace2")
        res_trace = eig(t)
        assert res_trace == ("mock_trace1", "mock_trace2")
        mock_emit.assert_called_once()


def test_eigh() -> None:
    from ml_switcheroo_compiler.ops.linalg.decompositions.eig import eigh

    config.eager_mode = True
    input_data = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    t = Tensor(input_data, TensorConfig(input_data.shape, "float32", "cpu"))
    res = eigh(t)
    assert isinstance(res, tuple)
    config.eager_mode = False
    eig_mod = sys.modules["ml_switcheroo_compiler.ops.linalg.decompositions.eig"]
    with patch.object(eig_mod, "_emit_linalg_node") as mock_emit:
        mock_emit.return_value = ("mock_trace1", "mock_trace2")
        res_trace = eigh(t)
        assert res_trace == ("mock_trace1", "mock_trace2")
        mock_emit.assert_called_once()


def test_eigvalsh() -> None:
    from ml_switcheroo_compiler.ops.linalg.decompositions.eig import eigvalsh

    config.eager_mode = True
    input_data = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    t = Tensor(input_data, TensorConfig(input_data.shape, "float32", "cpu"))
    res = eigvalsh(t)
    assert isinstance(res, Tensor)
    config.eager_mode = False
    eig_mod = sys.modules["ml_switcheroo_compiler.ops.linalg.decompositions.eig"]
    with patch.object(eig_mod, "_emit_linalg_node") as mock_emit:
        mock_emit.return_value = "mock_trace"
        res_trace = eigvalsh(t)
        assert res_trace == "mock_trace"
        mock_emit.assert_called_once()
