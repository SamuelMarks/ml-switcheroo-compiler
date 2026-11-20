"""Unit tests verifying collective operation shape inference, gradient propagation, and communication semantics."""

from __future__ import annotations

import typing

import numpy as np

from ml_switcheroo_compiler.backends.edge.config_models import WebrtcCollectivesConfig, load_webrtc_collectives
from ml_switcheroo_compiler.backends.edge.webgpu_webrtc import emit_webrtc_init, emit_webrtc_op
from ml_switcheroo_compiler.backends.numpy.eager import ensure_ndarray
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.distributed.collectives import dispatch_collective
from ml_switcheroo_compiler.ops.distributed_ops import AllGather, AllReduce, AllToAll, Broadcast, ReduceScatter
from ml_switcheroo_compiler.transforms.autodiff_rules.jvp_registry import get_jvp
from ml_switcheroo_compiler.transforms.autodiff_rules.vjp_registry import get_vjp

if typing.TYPE_CHECKING:
    pass


def test_webrtc_collectives_config_validation() -> None:
    """Test loading and validating declarative WebRTC collectives configuration."""
    cfg = load_webrtc_collectives()
    assert isinstance(cfg, WebrtcCollectivesConfig)
    assert cfg.schema_def.chunk_size_bytes == 65536
    assert cfg.schema_def.message_format == "binary"
    assert "allreduce_handler" in cfg.handlers
    assert "allgather_handler" in cfg.handlers
    assert "alltoall_handler" in cfg.handlers
    assert "reducescatter_handler" in cfg.handlers
    assert "broadcast_handler" in cfg.handlers


def test_webrtc_broadcast_emission() -> None:
    """Test emission of Broadcast JavaScript WebRTC templates."""
    op_code = emit_webrtc_op("Broadcast", "buf_input_0", "bc_node_1")
    assert "window.__ml_collective.broadcast" in op_code
    assert "bc_node_1" in op_code

    # Test all collective types in emit_webrtc_op
    assert "window.__ml_collective.allReduce" in emit_webrtc_op("AllReduce", "buf_0", "ar_1")
    assert "window.__ml_collective.allGather" in emit_webrtc_op("AllGather", "buf_0", "ag_1")
    assert "window.__ml_collective.allToAll" in emit_webrtc_op("AllToAll", "buf_0", "a2a_1")
    assert "window.__ml_collective.reduceScatter" in emit_webrtc_op("ReduceScatter", "buf_0", "rs_1")
    assert emit_webrtc_op("UnknownOp", "buf_0", "unk_1") == ""

    init_code = emit_webrtc_init()
    assert "RTCPeerConnection" in init_code


def test_collective_shape_inference() -> None:
    """Test shape inference across all collective operation definitions."""
    t_cfg = TensorConfig(shape=(8, 16), dtype="float32", device="cpu")
    t = Tensor(np.zeros((8, 16), dtype=np.float32), t_cfg)

    # 1. AllReduce preserves shape
    ar_op = AllReduce()
    assert ar_op.infer_shape(t) == (8, 16)

    # 2. Broadcast preserves shape
    bc_op = Broadcast()
    assert bc_op.infer_shape(t) == (8, 16)

    # 3. AllGather concatenates along axis
    ag_op = AllGather()
    assert ag_op.infer_shape(t, axis=0, world_size=4) == (32, 16)
    assert ag_op.infer_shape(t, axis=1, world_size=2) == (8, 32)
    assert ag_op.infer_shape(t) == (8, 16)

    # 4. ReduceScatter splits along axis
    rs_op = ReduceScatter()
    assert rs_op.infer_shape(t, axis=0, world_size=4) == (2, 16)
    assert rs_op.infer_shape(t, axis=1, world_size=2) == (8, 8)
    assert rs_op.infer_shape(t) == (8, 16)

    # 5. AllToAll preserves element count
    a2a_op = AllToAll()
    assert a2a_op.infer_shape(t) == (8, 16)


def test_collective_gradient_propagation() -> None:
    """Test VJP and JVP rules for collective operations."""
    # VJP checks
    vjp_ar = get_vjp("AllReduce")
    assert type(vjp_ar).__name__ == "function"

    vjp_ag = get_vjp("AllGather")
    assert type(vjp_ag).__name__ == "function"

    vjp_rs = get_vjp("ReduceScatter")
    assert type(vjp_rs).__name__ == "function"

    vjp_bc = get_vjp("Broadcast")
    assert type(vjp_bc).__name__ == "function"

    vjp_a2a = get_vjp("AllToAll")
    assert type(vjp_a2a).__name__ == "function"

    # JVP checks
    jvp_ar = get_jvp("AllReduce")
    assert type(jvp_ar).__name__ == "function"

    jvp_ag = get_jvp("AllGather")
    assert type(jvp_ag).__name__ == "function"

    jvp_rs = get_jvp("ReduceScatter")
    assert type(jvp_rs).__name__ == "function"

    jvp_bc = get_jvp("Broadcast")
    assert type(jvp_bc).__name__ == "function"

    jvp_a2a = get_jvp("AllToAll")
    assert type(jvp_a2a).__name__ == "function"


def test_collective_communication_semantics_eager() -> None:
    """Test eager communication execution semantics across simulated collective operations."""
    rank0_data = np.array([10.0, 20.0], dtype=np.float32)
    rank1_data = np.array([30.0, 40.0], dtype=np.float32)
    ranks = [rank0_data, rank1_data]

    # AllReduce SUM
    ar_res = dispatch_collective("AllReduce", rank0_data, backend="numpy", op_type="SUM", all_ranks_data=ranks)
    np.testing.assert_allclose(ensure_ndarray(ar_res), np.array([40.0, 60.0]))

    # AllGather
    ag_res = dispatch_collective("AllGather", rank0_data, backend="numpy", axis=0, all_ranks_data=ranks)
    np.testing.assert_allclose(ensure_ndarray(ag_res), np.array([10.0, 20.0, 30.0, 40.0]))

    # ReduceScatter
    rs_rank0 = dispatch_collective(
        "ReduceScatter", rank0_data, backend="numpy", op_type="SUM", scatter_dim=0, rank=0, all_ranks_data=ranks
    )
    rs_rank1 = dispatch_collective(
        "ReduceScatter", rank1_data, backend="numpy", op_type="SUM", scatter_dim=0, rank=1, all_ranks_data=ranks
    )
    np.testing.assert_allclose(ensure_ndarray(rs_rank0), np.array([40.0]))
    np.testing.assert_allclose(ensure_ndarray(rs_rank1), np.array([60.0]))

    # Broadcast from rank 1 to all
    bc_res = dispatch_collective("Broadcast", rank0_data, backend="numpy", root=1, all_ranks_data=ranks)
    np.testing.assert_allclose(ensure_ndarray(bc_res), rank1_data)
