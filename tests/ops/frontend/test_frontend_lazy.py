"""Tests for lazy symbolic tracing of creation, shape, and linear algebra operations."""

from __future__ import annotations

import typing

from ml_switcheroo_compiler.backends.eager_registry import EagerTensorProtocol
from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.ops import creation, linalg, shape
from ml_switcheroo_compiler.ops.linalg import decompositions as linalg_decomp
from ml_switcheroo_compiler.tracing.state import global_tracing_state
from ml_switcheroo_compiler.tracing.tracer import ProxyTensor


def test_lazy_frontend_direct_1() -> None:
    """Verify lazy tracing of shape manipulation operations."""
    config.eager_mode = False
    p_x = ProxyTensor(id="x", shape=(2, 2), dtype=DType.Float32)
    p_y = ProxyTensor(id="y", shape=(2, 2), dtype=DType.Float32)
    x = Tensor(p_x, TensorConfig((2, 2), DType.Float32, "cpu"))
    y = Tensor(p_y, TensorConfig((2, 2), DType.Float32, "cpu"))
    global_tracing_state.start_tracing()
    try:
        r1 = shape.frontend.reshape(x, (4,))
        assert isinstance(r1, Tensor)
        assert isinstance(shape.squeeze(x), Tensor)
        assert isinstance(shape.expand_dims(x, 0), Tensor)
        assert isinstance(shape.permute(x, (1, 0)), Tensor)
        assert isinstance(shape.swapaxes(x, 0, 1), Tensor)
        assert isinstance(shape.moveaxis(x, 0, 1), Tensor)
        assert isinstance(shape.roll(x, 1), Tensor)
        assert isinstance(shape.concatenate((x, y)), Tensor)
        assert isinstance(shape.stack((x, y)), Tensor)
        assert len(typing.cast("EagerTensorProtocol", shape.split(x, 2))) == 1
        assert len(typing.cast("EagerTensorProtocol", shape.hsplit(x, 2))) == 2
        assert len(typing.cast("EagerTensorProtocol", shape.vsplit(x, 2))) == 2
        assert global_tracing_state.active_graph is not None
        assert len(global_tracing_state.active_graph.nodes) > 0
    finally:
        global_tracing_state.stop_tracing()


def test_lazy_frontend_direct_2() -> None:
    """Verify lazy tracing of indexing, tiling, and triangular shape operations."""
    config.eager_mode = False
    p_x = ProxyTensor(id="x", shape=(2, 2), dtype=DType.Float32)
    p_y = ProxyTensor(id="y", shape=(2, 2), dtype=DType.Float32)
    x = Tensor(p_x, TensorConfig((2, 2), DType.Float32, "cpu"))
    y = Tensor(p_y, TensorConfig((2, 2), DType.Float32, "cpu"))
    global_tracing_state.start_tracing()
    try:
        p_3d = ProxyTensor(id="3d", shape=(2, 2, 2), dtype=DType.Float32)
        t_3d = Tensor(p_3d, TensorConfig((2, 2, 2), DType.Float32, "cpu"))
        assert len(typing.cast("EagerTensorProtocol", shape.dsplit(t_3d, 2))) == 2
        assert isinstance(shape.misc.tile(x, (2, 2)), Tensor)
        assert isinstance(shape.misc.repeat(x, 2), Tensor)
        assert isinstance(shape.misc.pad(x, ((1, 1), (1, 1))), Tensor)
        p_bool = ProxyTensor(id="bool", shape=(2, 2), dtype=DType.Bool)
        t_bool = Tensor(p_bool, TensorConfig((2, 2), DType.Bool, "cpu"))
        assert isinstance(shape.indexing.where(t_bool, x, y), Tensor)
        p_int = ProxyTensor(id="int", shape=(2, 2), dtype=DType.Int32)
        t_int = Tensor(p_int, TensorConfig((2, 2), DType.Int32, "cpu"))
        assert isinstance(shape.indexing.gather(x, 0, t_int), Tensor)
        assert isinstance(shape.indexing.take(x, t_int), Tensor)
        assert isinstance(shape.scatter.scatter(x, 0, t_int, t_int), Tensor)
        assert isinstance(shape.slicing.slice(x, 0, 0, 1), Tensor)
        assert isinstance(shape.indexing.select(t_bool, x, y), Tensor)
        assert isinstance(shape.misc.sort(x), Tensor)
        assert isinstance(shape.misc.argsort(x), Tensor)
        p_z = ProxyTensor(id="z", shape=(4,), dtype=DType.Float32)
        z = Tensor(p_z, TensorConfig((4,), DType.Float32, "cpu"))
        assert len(shape.misc.meshgrid(z, z)) == 2
        assert isinstance(shape.misc.tril(x), Tensor)
        assert isinstance(shape.misc.triu(x), Tensor)
        res_diag = creation.diag(x)
        assert isinstance(res_diag, Tensor)
    finally:
        global_tracing_state.stop_tracing()


def test_lazy_frontend_direct_3() -> None:
    """Verify lazy tracing of tensor creation and matrix product operations."""
    config.eager_mode = False
    p_x = ProxyTensor(id="x", shape=(2, 2), dtype=DType.Float32)
    p_y = ProxyTensor(id="y", shape=(2, 2), dtype=DType.Float32)
    x = Tensor(p_x, TensorConfig((2, 2), DType.Float32, "cpu"))
    y = Tensor(p_y, TensorConfig((2, 2), DType.Float32, "cpu"))
    global_tracing_state.start_tracing()
    try:
        z0 = creation.zeros((2, 2))
        assert z0.shape == (2, 2)
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
        res_ein = linalg.einsum("ab,bc->ac", x, y)
        assert res_ein is not None
    finally:
        global_tracing_state.stop_tracing()


def test_lazy_frontend_direct_4() -> None:
    """Verify lazy tracing of linear algebra decompositions and solvers."""
    config.eager_mode = False
    p_x = ProxyTensor(id="x", shape=(2, 2), dtype=DType.Float32)
    p_y = ProxyTensor(id="y", shape=(2, 2), dtype=DType.Float32)
    p_z = ProxyTensor(id="z", shape=(3,), dtype=DType.Float32)
    x = Tensor(p_x, TensorConfig((2, 2), DType.Float32, "cpu"))
    y = Tensor(p_y, TensorConfig((2, 2), DType.Float32, "cpu"))
    z = Tensor(p_z, TensorConfig((3,), DType.Float32, "cpu"))
    global_tracing_state.start_tracing()
    try:
        assert isinstance(linalg_decomp.matrix_power(x, 2), Tensor)
        assert isinstance(linalg_decomp.cholesky(x), Tensor)
        assert len(typing.cast("EagerTensorProtocol", linalg_decomp.qr(x))) == 2
        assert len(typing.cast("EagerTensorProtocol", linalg_decomp.svd(x))) == 3
        assert len(typing.cast("EagerTensorProtocol", linalg_decomp.eig(x))) == 2
        assert len(typing.cast("EagerTensorProtocol", linalg_decomp.eigh(x))) == 2
        assert isinstance(linalg_decomp.eigvals(x), Tensor)
        assert isinstance(linalg_decomp.eigvalsh(x), Tensor)
        assert isinstance(linalg_decomp.norm(x), Tensor)
        assert isinstance(linalg.cond(x), Tensor)
        assert isinstance(linalg_decomp.det(x), Tensor)
        assert len(typing.cast("EagerTensorProtocol", linalg_decomp.slogdet(x))) == 2
        assert isinstance(linalg.trace(x), Tensor)
        assert isinstance(linalg_decomp.solve(x, y), Tensor)
        assert isinstance(linalg_decomp.inv(x), Tensor)
        assert isinstance(linalg_decomp.pinv(x), Tensor)
        res_cross = linalg.cross(z, z)
        assert res_cross.shape == (3,)
    finally:
        global_tracing_state.stop_tracing()
