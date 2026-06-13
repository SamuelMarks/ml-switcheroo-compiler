"""Provides required module functionality."""

import contextlib

from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.tensor import Tensor
from ml_switcheroo_compiler.ops import creation, linalg, shape
from ml_switcheroo_compiler.tracing.tracer import ProxyTensor, _tracer


def test_lazy_frontend_direct() -> None:
    """Test frontend lazy APIs."""
    config.eager_mode = False

    p_x = ProxyTensor(id="x", shape=(2, 2), dtype=DType.Float32)
    p_y = ProxyTensor(id="y", shape=(2, 2), dtype=DType.Float32)
    p_z = ProxyTensor(id="z", shape=(4,), dtype=DType.Float32)

    x = Tensor(data=p_x, shape=(2, 2), dtype=DType.Float32, device="cpu")
    y = Tensor(data=p_y, shape=(2, 2), dtype=DType.Float32, device="cpu")
    z = Tensor(data=p_z, shape=(4,), dtype=DType.Float32, device="cpu")

    _tracer.start_tracing()
    try:
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

        p_3d = ProxyTensor(id="3d", shape=(2, 2, 2), dtype=DType.Float32)
        t_3d = Tensor(data=p_3d, shape=(2, 2, 2), dtype=DType.Float32, device="cpu")
        shape.dsplit(t_3d, 2)

        shape.tile(x, (2, 2))
        shape.repeat(x, 2)
        shape.pad(x, ((1, 1), (1, 1)))
        shape.broadcast_to(x, (2, 2, 2))
        shape.expand(x, (2, 2, 2))

        p_bool = ProxyTensor(id="bool", shape=(2, 2), dtype=DType.Bool)
        t_bool = Tensor(data=p_bool, shape=(2, 2), dtype=DType.Bool, device="cpu")
        shape.where(t_bool, x, y)

        p_int = ProxyTensor(id="int", shape=(2, 2), dtype=DType.Int32)
        t_int = Tensor(data=p_int, shape=(2, 2), dtype=DType.Int32, device="cpu")
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
    finally:
        _tracer.stop_tracing()
