"""Unit tests verifying collective operation shape inference, gradient propagation, and communication semantics."""

from __future__ import annotations

import numpy as np

from ml_switcheroo_compiler.backends.edge.config_models import (
    WebrtcCollectivesConfig,
    load_webrtc_collectives,
)
from ml_switcheroo_compiler.backends.edge.webgpu_webrtc import emit_webrtc_init, emit_webrtc_op
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.distributed.collectives import dispatch_collective
from ml_switcheroo_compiler.ops.distributed_ops import (
    AllGather,
    AllReduce,
    AllToAll,
    Broadcast,
    ReduceScatter,
)
from ml_switcheroo_compiler.transforms.autodiff_rules.jvp_registry import get_jvp
from ml_switcheroo_compiler.transforms.autodiff_rules.vjp_registry import get_vjp


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
    assert "waitForBroadcast" in op_code
    assert "bc_node_1" in op_code

    # Test all collective types in emit_webrtc_op
    assert "waitForReduce" in emit_webrtc_op("AllReduce", "buf_0", "ar_1")
    assert "waitForGather" in emit_webrtc_op("AllGather", "buf_0", "ag_1")
    assert "waitForAllToAll" in emit_webrtc_op("AllToAll", "buf_0", "a2a_1")
    assert "waitForReduceScatter" in emit_webrtc_op("ReduceScatter", "buf_0", "rs_1")
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
    assert vjp_ar is not None

    vjp_ag = get_vjp("AllGather")
    assert vjp_ag is not None

    vjp_rs = get_vjp("ReduceScatter")
    assert vjp_rs is not None

    vjp_bc = get_vjp("Broadcast")
    assert vjp_bc is not None

    vjp_a2a = get_vjp("AllToAll")
    assert vjp_a2a is not None

    # JVP checks
    jvp_ar = get_jvp("AllReduce")
    assert jvp_ar is not None

    jvp_ag = get_jvp("AllGather")
    assert jvp_ag is not None

    jvp_rs = get_jvp("ReduceScatter")
    assert jvp_rs is not None

    jvp_bc = get_jvp("Broadcast")
    assert jvp_bc is not None

    jvp_a2a = get_jvp("AllToAll")
    assert jvp_a2a is not None


def test_collective_communication_semantics_eager() -> None:
    """Test eager communication execution semantics across simulated collective operations."""
    rank0_data = np.array([10.0, 20.0], dtype=np.float32)
    rank1_data = np.array([30.0, 40.0], dtype=np.float32)
    ranks = [rank0_data, rank1_data]

    # AllReduce SUM
    ar_res = dispatch_collective("AllReduce", rank0_data, backend="numpy", op_type="SUM", all_ranks_data=ranks)
    np.testing.assert_allclose(ar_res, np.array([40.0, 60.0]))

    # AllGather
    ag_res = dispatch_collective("AllGather", rank0_data, backend="numpy", axis=0, all_ranks_data=ranks)
    np.testing.assert_allclose(ag_res, np.array([10.0, 20.0, 30.0, 40.0]))

    # ReduceScatter
    rs_rank0 = dispatch_collective("ReduceScatter", rank0_data, backend="numpy", op_type="SUM", scatter_dim=0, rank=0, all_ranks_data=ranks)
    rs_rank1 = dispatch_collective("ReduceScatter", rank1_data, backend="numpy", op_type="SUM", scatter_dim=0, rank=1, all_ranks_data=ranks)
    np.testing.assert_allclose(rs_rank0, np.array([40.0]))
    np.testing.assert_allclose(rs_rank1, np.array([60.0]))

    # Broadcast from rank 1 to all
    bc_res = dispatch_collective("Broadcast", rank0_data, backend="numpy", root=1, all_ranks_data=ranks)
    np.testing.assert_allclose(bc_res, rank1_data)
