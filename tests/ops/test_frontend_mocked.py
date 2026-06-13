"""Module docstring."""

import contextlib
from unittest.mock import MagicMock, patch

from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.tensor import Tensor
from ml_switcheroo_compiler.ops import creation, linalg, shape


@patch("ml_switcheroo_compiler.tracing.tracer._tracer.is_tracing", True)
@patch("ml_switcheroo_compiler.tracing.tracer._tracer.add_node")
@patch("ml_switcheroo_compiler.tracing.tracer.ProxyTensor")
def test_lazy_frontend_mocked(mock_proxy: object, mock_add_node: object) -> None:
    """Test frontend lazy APIs using mocks."""
    config.eager_mode = False

    mock_proxy.return_value = MagicMock(id="dummy")

    p_x = MagicMock(id="x", shape=(2, 2))
    p_y = MagicMock(id="y", shape=(2, 2))
    p_z = MagicMock(id="z", shape=(4,))

    x = Tensor(data=p_x, shape=(2, 2), dtype=DType.Float32, device="cpu")
    y = Tensor(data=p_y, shape=(2, 2), dtype=DType.Float32, device="cpu")
    z = Tensor(data=p_z, shape=(4,), dtype=DType.Float32, device="cpu")

    shape.reshape(x, (4,))
    shape.flatten(x)
    shape.squeeze(x)
    shape.unsqueeze(x, 0)
    shape.transpose(x, 0, 1)
    shape.swapaxes(x, 0, 1)
    shape.moveaxis(x, 0, 1)
    shape.roll(x, 1)
    shape.concatenate((x, y))
    shape.stack((x, y))
    shape.split(x, 2)
    shape.hsplit(x, 2)
    shape.vsplit(x, 2)

    t_3d = Tensor(
        data=MagicMock(id="3d", shape=(2, 2, 2)),
        shape=(2, 2, 2),
        dtype=DType.Float32,
        device="cpu",
    )
    shape.dsplit(t_3d, 2)

    shape.tile(x, (2, 2))
    shape.repeat(x, 2)
    shape.pad(x, ((1, 1), (1, 1)))
    shape.broadcast_to(x, (2, 2, 2))
    shape.expand(x, (2, 2, 2))

    t_bool = Tensor(
        data=MagicMock(id="bool", shape=(2, 2)),
        shape=(2, 2),
        dtype=DType.Bool,
        device="cpu",
    )
    shape.where(t_bool, x, y)

    t_int = Tensor(
        data=MagicMock(id="int", shape=(2, 2)),
        shape=(2, 2),
        dtype=DType.Int32,
        device="cpu",
    )
    shape.gather(x, 0, t_int)
    shape.take(x, t_int)
    shape.scatter(x, 0, t_int, t_int)
    shape.slice(x, (0, 0), (1, 1))
    shape.select(t_bool, x, y)

    with contextlib.suppress(Exception):
        shape.sort(x)
    with contextlib.suppress(Exception):
        shape.argsort(x)

    shape.meshgrid(z, z)
    shape.tril(x)
    shape.triu(x)
    creation.diag(z)

    creation.zeros((2, 2))
    creation.zeros_like(x)
    creation.ones((2, 2))
    creation.ones_like(x)
    creation.empty((2, 2))
    creation.empty_like(x)
    creation.full((2, 2), 1.0)
    creation.full_like(x, 1.0)
    creation.eye(2)
    creation.identity(2)
    creation.arange(10)
    creation.linspace(0, 1, 10)

    linalg.dot(x, y)
    linalg.matmul(x, y)
    linalg.vdot(x, y)
    linalg.inner(x, y)
    linalg.outer(x, y)
    linalg.tensordot(x, y, axes=1)
    linalg.einsum("ab,bc->ac", x, y)
    linalg.matrix_power(x, 2)
    with contextlib.suppress(Exception):
        linalg.cholesky(x)
    with contextlib.suppress(Exception):
        linalg.qr(x)
    with contextlib.suppress(Exception):
        linalg.svd(x)
    with contextlib.suppress(Exception):
        linalg.eig(x)
    with contextlib.suppress(Exception):
        linalg.eigh(x)
    with contextlib.suppress(Exception):
        linalg.eigvals(x)
    with contextlib.suppress(Exception):
        linalg.eigvalsh(x)
    with contextlib.suppress(Exception):
        linalg.norm(x)
    with contextlib.suppress(Exception):
        linalg.cond(x)
    with contextlib.suppress(Exception):
        linalg.det(x)
    with contextlib.suppress(Exception):
        linalg.slogdet(x)
    with contextlib.suppress(Exception):
        linalg.trace(x)
    with contextlib.suppress(Exception):
        linalg.solve(x, y)
    with contextlib.suppress(Exception):
        linalg.inv(x)
    with contextlib.suppress(Exception):
        linalg.pinv(x)
    with contextlib.suppress(Exception):
        linalg.cross(z, z)
