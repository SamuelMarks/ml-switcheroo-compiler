"""Tests for lazy frontend shape, creation, and linalg API tracing."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.ops import creation, linalg, shape
from ml_switcheroo_compiler.ops.linalg import decompositions as linalg_decomp
from tests.testing_utils import as_tensor


@patch("ml_switcheroo_compiler.tracing.tracer.global_tracing_state.is_tracing", True)
@patch("ml_switcheroo_compiler.tracing.tracer.global_tracing_state.add_node")
@patch("ml_switcheroo_compiler.tracing.tracer.ProxyTensor")
def test_lazy_frontend_mocked_1(mock_proxy: MagicMock, mock_add_node: MagicMock) -> None:
    """Verify lazy frontend shape manipulation operations emit tracing nodes.

    Args:
        mock_proxy (MagicMock): Mocked ProxyTensor class.
        mock_add_node (MagicMock): Mocked global_tracing_state.add_node method.
    """
    config.eager_mode = False
    mock_proxy.return_value = MagicMock(id="dummy")
    p_x = MagicMock(id="x", shape=(2, 2))
    p_y = MagicMock(id="y", shape=(2, 2))
    x = Tensor(p_x, TensorConfig((2, 2), DType.Float32, "cpu"))
    y = Tensor(p_y, TensorConfig((2, 2), DType.Float32, "cpu"))
    assert isinstance(shape.frontend.reshape(x, (4,)), Tensor)
    assert isinstance(shape.squeeze(x), Tensor)
    assert isinstance(shape.expand_dims(x, 0), Tensor)
    assert isinstance(shape.permute(x, (1, 0)), Tensor)
    assert isinstance(shape.swapaxes(x, 0, 1), Tensor)
    assert isinstance(shape.moveaxis(x, 0, 1), Tensor)
    assert isinstance(shape.roll(x, 1), Tensor)
    assert isinstance(shape.concatenate((x, y)), Tensor)
    assert isinstance(shape.stack((x, y)), Tensor)
    assert len(as_tensor(shape.split(x, 2))) == 1
    assert len(as_tensor(shape.hsplit(x, 2))) == 2
    assert len(as_tensor(shape.vsplit(x, 2))) == 2
    assert mock_add_node.called


@patch("ml_switcheroo_compiler.tracing.tracer.global_tracing_state.is_tracing", True)
@patch("ml_switcheroo_compiler.tracing.tracer.global_tracing_state.add_node")
@patch("ml_switcheroo_compiler.tracing.tracer.ProxyTensor")
def test_lazy_frontend_mocked_2(mock_proxy: MagicMock, mock_add_node: MagicMock) -> None:
    """Verify lazy frontend indexing, tiling, padding, and sorting operations emit tracing nodes.

    Args:
        mock_proxy (MagicMock): Mocked ProxyTensor class.
        mock_add_node (MagicMock): Mocked global_tracing_state.add_node method.
    """
    config.eager_mode = False
    mock_proxy.return_value = MagicMock(id="dummy")
    p_x = MagicMock(id="x", shape=(2, 2))
    p_y = MagicMock(id="y", shape=(2, 2))
    x = Tensor(p_x, TensorConfig((2, 2), DType.Float32, "cpu"))
    y = Tensor(p_y, TensorConfig((2, 2), DType.Float32, "cpu"))
    t_3d = Tensor(MagicMock(id="3d", shape=(2, 2, 2)), TensorConfig((2, 2, 2), DType.Float32, "cpu"))
    assert len(as_tensor(shape.dsplit(t_3d, 2))) == 2
    assert isinstance(shape.misc.tile(x, (2, 2)), Tensor)
    assert isinstance(shape.misc.repeat(x, 2), Tensor)
    assert isinstance(shape.misc.pad(x, ((1, 1), (1, 1))), Tensor)
    t_bool = Tensor(MagicMock(id="bool", shape=(2, 2)), TensorConfig((2, 2), DType.Bool, "cpu"))
    assert isinstance(shape.indexing.where(t_bool, x, y), Tensor)
    t_int = Tensor(MagicMock(id="int", shape=(2, 2)), TensorConfig((2, 2), DType.Int32, "cpu"))
    assert isinstance(shape.indexing.gather(x, 0, t_int), Tensor)
    assert isinstance(shape.indexing.take(x, t_int), Tensor)
    assert isinstance(shape.scatter.scatter(x, 0, t_int, t_int), Tensor)
    assert isinstance(shape.slicing.slice(x, 0, 0, 1), Tensor)
    assert isinstance(shape.indexing.select(t_bool, x, y), Tensor)
    assert isinstance(shape.misc.sort(x), Tensor)
    assert isinstance(shape.misc.argsort(x), Tensor)
    assert mock_add_node.called


@patch("ml_switcheroo_compiler.tracing.tracer.global_tracing_state.is_tracing", True)
@patch("ml_switcheroo_compiler.tracing.tracer.global_tracing_state.add_node")
@patch("ml_switcheroo_compiler.tracing.tracer.ProxyTensor")
def test_lazy_frontend_mocked_3(mock_proxy: MagicMock, mock_add_node: MagicMock) -> None:
    """Verify lazy frontend creation and basic matrix multiplication operations emit tracing nodes.

    Args:
        mock_proxy (MagicMock): Mocked ProxyTensor class.
        mock_add_node (MagicMock): Mocked global_tracing_state.add_node method.
    """
    config.eager_mode = False
    mock_proxy.return_value = MagicMock(id="dummy")
    p_x = MagicMock(id="x", shape=(2, 2))
    p_y = MagicMock(id="y", shape=(2, 2))
    p_z = MagicMock(id="z", shape=(4,))
    x = Tensor(p_x, TensorConfig((2, 2), DType.Float32, "cpu"))
    y = Tensor(p_y, TensorConfig((2, 2), DType.Float32, "cpu"))
    z = Tensor(p_z, TensorConfig((4,), DType.Float32, "cpu"))
    assert len(shape.misc.meshgrid(z, z)) == 2
    assert isinstance(shape.misc.tril(x), Tensor)
    assert isinstance(shape.misc.triu(x), Tensor)
    assert isinstance(creation.diag(x), Tensor)
    assert isinstance(creation.zeros((2, 2)), Tensor)
    assert isinstance(creation.zeros_like(x), Tensor)
    assert isinstance(creation.ones((2, 2)), Tensor)
    assert isinstance(creation.ones_like(x), Tensor)
    assert isinstance(creation.empty((2, 2)), Tensor)
    assert isinstance(creation.empty_like(x), Tensor)
    assert isinstance(creation.full((2, 2), 1.0), Tensor)
    assert isinstance(creation.full_like(x, 1.0), Tensor)
    assert isinstance(creation.eye(2), Tensor)
    assert isinstance(creation.identity(2), Tensor)
    assert isinstance(creation.arange(10), Tensor)
    assert isinstance(creation.linspace(0, 1, 10), Tensor)
    assert isinstance(linalg.dot(x, y), Tensor)
    assert isinstance(linalg.matmul(x, y), Tensor)
    assert isinstance(linalg.vdot(x, y), Tensor)
    assert isinstance(linalg.inner(x, y), Tensor)
    assert isinstance(linalg.outer(x, y), Tensor)
    assert isinstance(linalg.tensordot(x, y, axes=1), Tensor)
    assert isinstance(linalg.einsum("ab,bc->ac", x, y), Tensor)
    assert mock_add_node.called


@patch("ml_switcheroo_compiler.tracing.tracer.global_tracing_state.is_tracing", True)
@patch("ml_switcheroo_compiler.tracing.tracer.global_tracing_state.add_node")
@patch("ml_switcheroo_compiler.tracing.tracer.ProxyTensor")
def test_lazy_frontend_mocked_4(mock_proxy: MagicMock, mock_add_node: MagicMock) -> None:
    """Verify lazy frontend linear algebra decomposition and solver operations emit tracing nodes.

    Args:
        mock_proxy (MagicMock): Mocked ProxyTensor class.
        mock_add_node (MagicMock): Mocked global_tracing_state.add_node method.
    """
    config.eager_mode = False
    mock_proxy.return_value = MagicMock(id="dummy")
    p_x = MagicMock(id="x", shape=(2, 2))
    p_y = MagicMock(id="y", shape=(2, 2))
    p_z = MagicMock(id="z", shape=(3,))
    x = Tensor(p_x, TensorConfig((2, 2), DType.Float32, "cpu"))
    y = Tensor(p_y, TensorConfig((2, 2), DType.Float32, "cpu"))
    z = Tensor(p_z, TensorConfig((3,), DType.Float32, "cpu"))
    assert isinstance(linalg_decomp.matrix_power(x, 2), Tensor)
    assert isinstance(linalg_decomp.cholesky(x), Tensor)
    assert len(as_tensor(linalg_decomp.qr(x))) == 2
    assert len(as_tensor(linalg_decomp.svd(x))) == 3
    assert len(as_tensor(linalg_decomp.eig(x))) == 2
    assert len(as_tensor(linalg_decomp.eigh(x))) == 2
    assert isinstance(linalg_decomp.eigvals(x), Tensor)
    assert isinstance(linalg_decomp.eigvalsh(x), Tensor)
    assert isinstance(linalg_decomp.norm(x), Tensor)
    assert isinstance(linalg.cond(x), Tensor)
    assert isinstance(linalg_decomp.det(x), Tensor)
    assert len(as_tensor(linalg_decomp.slogdet(x))) == 2
    assert isinstance(linalg.trace(x), Tensor)
    assert isinstance(linalg_decomp.solve(x, y), Tensor)
    assert isinstance(linalg_decomp.inv(x), Tensor)
    assert isinstance(linalg_decomp.pinv(x), Tensor)
    assert isinstance(linalg.cross(z, z), Tensor)
    assert mock_add_node.called
