"""Unit tests for WebRTC distributed collective communication algorithms and streaming."""

import asyncio
import struct
import sys
import zlib
from typing import Optional
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from ml_switcheroo_compiler.backends.edge.distributed_webrtc.collectives import (
    DistributedBarrier,
    WebRTCDataChannelStream,
    _recv_data,
    _send_data,
    all_gather,
    all_reduce,
    all_to_all,
    broadcast,
    dispatch_collective,
    get_collective_backend,
    recursive_halving_doubling_all_gather,
    reduce_scatter,
    register_collective_backend,
    ring_all_reduce,
    ring_reduce_scatter,
)


class MockChannel:
    """Mock RTCDataChannel for testing streaming and event handlers."""

    def __init__(self) -> None:
        """Initialize mock channel."""
        self.sent_messages: list[bytes] = []
        self.bufferedAmount: int = 0
        self.bufferedAmountLowThreshold: int = 0
        self.onmessage: Optional[object] = None
        self.onbufferedamountlow: Optional[object] = None

    async def send(self, data: bytes) -> None:
        """Mock async send."""
        self.sent_messages.append(data)


class MockSnakeChannel:
    """Mock RTCDataChannel with snake_case attributes."""

    def __init__(self) -> None:
        """Initialize mock snake_case channel."""
        self.sent: list[bytes] = []
        self.buffered_amount: int = 0
        self.buffered_amount_low_threshold: int = 0
        self.on_message: Optional[object] = None
        self.on_buffered_amount_low: Optional[object] = None

    def sendMessage(self, data: bytes) -> None:
        """Mock sync sendMessage."""
        self.sent.append(data)


class BiPipeChannel:
    """In-memory bidirectional paired pipe simulating WebRTC DataChannel."""

    def __init__(
        self,
        send_queue: asyncio.Queue[bytes],
        recv_queue: asyncio.Queue[bytes],
    ) -> None:
        """Initialize bidirectional pipe with complementary queues."""
        self.send_queue = send_queue
        self.recv_queue = recv_queue
        self.bufferedAmount: int = 0

    async def send(self, data: bytes) -> None:
        """Send by putting into send queue."""
        await self.send_queue.put(data)

    async def recv(self) -> bytes:
        """Receive by pulling from recv queue."""
        return await self.recv_queue.get()


def make_bipipe_pair() -> tuple[BiPipeChannel, BiPipeChannel]:
    """Create a complementary pair of bidirectional pipe channels."""
    q_a2b: asyncio.Queue[bytes] = asyncio.Queue()
    q_b2a: asyncio.Queue[bytes] = asyncio.Queue()
    return BiPipeChannel(q_a2b, q_b2a), BiPipeChannel(q_b2a, q_a2b)


def create_interconnected_mesh(world_size: int) -> list[list[Optional[BiPipeChannel]]]:
    """Create a fully connected mesh of simulated bidirectional pipe channels."""
    conns: list[list[Optional[BiPipeChannel]]] = [[None for _ in range(world_size)] for _ in range(world_size)]
    for i in range(world_size):
        for j in range(i + 1, world_size):
            ch_ij, ch_ji = make_bipipe_pair()
            conns[i][j] = ch_ij
            conns[j][i] = ch_ji
    return conns


@pytest.mark.asyncio
async def test_stream_event_handlers_and_messages() -> None:
    """Test WebRTCDataChannelStream event listener attachment and message receiving."""
    mock = MockChannel()
    stream = WebRTCDataChannelStream(mock, high_watermark=100, low_watermark=20)
    assert stream.channel is mock

    stream._on_message(b"bytes_msg")
    assert await stream.recv_bytes() == b"bytes_msg"

    stream._on_message("str_msg")
    assert await stream.recv_bytes() == b"str_msg"

    class ToBytesObj:
        def tobytes(self) -> bytes:
            return b"tobytes_msg"

    stream._on_message(ToBytesObj())
    assert await stream.recv_bytes() == b"tobytes_msg"

    stream._on_message(bytearray(b"bytearray_msg"))
    assert await stream.recv_bytes() == b"bytearray_msg"

    mock_snake = MockSnakeChannel()
    stream_snake = WebRTCDataChannelStream(mock_snake)
    assert mock_snake.on_message is not None

    await stream_snake.send_bytes(b"hello")
    assert mock_snake.sent == [b"hello"]

    class CallableAmount:
        def bufferedAmount(self) -> int:
            return 42

    stream_cb = WebRTCDataChannelStream(CallableAmount())
    assert stream_cb.get_buffered_amount() == 42

    # Async sendMessage channel
    class AsyncSendMessageChannel:
        def __init__(self):
            self.msgs = []

        async def sendMessage(self, data: bytes):
            self.msgs.append(data)

    ch_async_sendmsg = AsyncSendMessageChannel()
    st_async_sendmsg = WebRTCDataChannelStream(ch_async_sendmsg)
    await st_async_sendmsg.send_bytes(b"async_sendmsg")
    assert ch_async_sendmsg.msgs == [b"async_sendmsg"]

    # Sync send returning coroutine
    class SyncSendCoroutineChannel:
        def __init__(self):
            self.msgs = []

        def send(self, data: bytes):
            async def _coro():
                self.msgs.append(data)

            return _coro()

    ch_sync_coro = SyncSendCoroutineChannel()
    st_sync_coro = WebRTCDataChannelStream(ch_sync_coro)
    await st_sync_coro.send_bytes(b"coro_msg")
    assert ch_sync_coro.msgs == [b"coro_msg"]

    # Sync send returning None
    class SyncSendNoneChannel:
        def __init__(self):
            self.msgs = []

        def send(self, data: bytes):
            self.msgs.append(data)

    ch_sync_none = SyncSendNoneChannel()
    st_sync_none = WebRTCDataChannelStream(ch_sync_none)
    await st_sync_none.send_bytes(b"none_msg")
    assert ch_sync_none.msgs == [b"none_msg"]

    # Sync recv returning coroutine
    class SyncRecvChannel:
        def recv(self):
            async def _r():
                return b"from_recv"

            return _r()

    st_recv = WebRTCDataChannelStream(SyncRecvChannel())
    assert await st_recv.recv_bytes() == b"from_recv"

    # Non-coroutine recv
    class NonCoroRecvChannel:
        def recv(self):
            return b"non_coro_bytes"

    st_noncoro = WebRTCDataChannelStream(NonCoroRecvChannel())
    assert await st_noncoro.recv_bytes() == b"non_coro_bytes"

    # Non-coroutine recv returning bytearray
    class BytearrayRecvChannel:
        def recv(self):
            return bytearray(b"bytearray_res")

    st_ba = WebRTCDataChannelStream(BytearrayRecvChannel())
    assert await st_ba.recv_bytes() == b"bytearray_res"

    # Property buffered_amount
    class PropAmount:
        buffered_amount = 99

    st_prop = WebRTCDataChannelStream(PropAmount())
    assert st_prop.get_buffered_amount() == 99

    # Sync sendMessage returning coroutine
    class SyncSendMessageCoroChannel:
        def __init__(self):
            self.msgs = []

        def sendMessage(self, data: bytes):
            async def _c():
                self.msgs.append(data)

            return _c()

    st_sync_sendmsg = WebRTCDataChannelStream(SyncSendMessageCoroChannel())
    await st_sync_sendmsg.send_bytes(b"sync_sendmsg_coro")
    assert len(st_sync_sendmsg.channel.msgs) == 1


@pytest.mark.asyncio
async def test_stream_backpressure() -> None:
    """Test wait_backpressure handling with drain callbacks and events."""
    mock = MockChannel()
    stream = WebRTCDataChannelStream(mock, high_watermark=50, low_watermark=10)

    mock.bufferedAmount = 20
    await stream.wait_backpressure()

    drain_called = False

    async def mock_wait_drain() -> None:
        nonlocal drain_called
        drain_called = True
        mock.bufferedAmount = 5

    mock.bufferedAmount = 100
    mock.wait_drain = mock_wait_drain  # type: ignore[attr-defined]
    await stream.wait_backpressure()
    assert drain_called is True

    mock.bufferedAmount = 100

    def mock_sync_drain() -> None:
        mock.bufferedAmount = 5

    mock.wait_drain = mock_sync_drain  # type: ignore[attr-defined]
    await stream.wait_backpressure()

    del mock.wait_drain  # type: ignore[attr-defined]
    mock.bufferedAmount = 100

    async def trigger_drain():
        await asyncio.sleep(0.001)
        mock.bufferedAmount = 5
        stream._on_buffered_amount_low(None)

    asyncio.create_task(trigger_drain())
    await stream.wait_backpressure()


@pytest.mark.asyncio
async def test_send_recv_data_chunking_and_errors() -> None:
    """Test chunked binary tensor transmission, endianness, and validation errors."""
    ch_a, ch_b = make_bipipe_pair()
    data = np.arange(100, dtype=np.float32).reshape(10, 10)

    send_task = _send_data(ch_a, data, chunk_size=64)
    recv_task = _recv_data(ch_b)
    _, received = await asyncio.gather(send_task, recv_task)
    np.testing.assert_array_equal(received, data)

    # Endianness conversion test
    foreign_order = ">" if sys.byteorder == "little" else "<"
    raw = data.tobytes()
    meta = {
        "protocol": "webrtc_stream_v1",
        "shape": list(data.shape),
        "dtype": data.dtype.str,
        "byteorder": foreign_order,
        "total_bytes": len(raw),
        "num_chunks": 1,
        "chunk_size": len(raw),
    }
    import json

    ch_end1, ch_end2 = make_bipipe_pair()
    await ch_end1.send(b"META:" + json.dumps(meta).encode("utf-8"))
    header = struct.pack("!6sIII", b"CHUNK:", 0, 1, zlib.crc32(raw))
    await ch_end1.send(header + raw)
    res_endian = await _recv_data(ch_end2)
    assert res_endian is not None

    # CRC error test
    ch_bad1, ch_bad2 = make_bipipe_pair()
    await ch_bad1.send(b"META:" + json.dumps(meta).encode("utf-8"))
    bad_chksum = 999999
    header = struct.pack("!6sIII", b"CHUNK:", 0, 1, bad_chksum)
    await ch_bad1.send(header + raw)

    with pytest.raises(ValueError, match="Checksum mismatch"):
        await _recv_data(ch_bad2)

    # Payload length mismatch test
    ch_len1, ch_len2 = make_bipipe_pair()
    meta["total_bytes"] = len(raw) + 10
    chksum = zlib.crc32(raw)
    header = struct.pack("!6sIII", b"CHUNK:", 0, 1, chksum)
    await ch_len1.send(b"META:" + json.dumps(meta).encode("utf-8"))
    await ch_len1.send(header + raw)

    with pytest.raises(ValueError, match="Payload length mismatch"):
        await _recv_data(ch_len2)

    # Legacy header fallback test
    ch_leg1, ch_leg2 = make_bipipe_pair()
    legacy_meta = b"2,2|float32"
    arr_2x2 = np.ones((2, 2), dtype=np.float32)
    await ch_leg1.send(legacy_meta)
    await ch_leg1.send(arr_2x2.tobytes())
    res_legacy = await _recv_data(ch_leg2)
    np.testing.assert_array_equal(res_legacy, arr_2x2)


@pytest.mark.asyncio
async def test_broadcast_collective() -> None:
    """Test Broadcast collective across ranks."""
    world_size = 3
    mesh = create_interconnected_mesh(world_size)
    tensor = np.array([10.0, 20.0, 30.0], dtype=np.float32)

    async def run_rank(r: int) -> np.ndarray:
        local_t = tensor if r == 0 else np.zeros_like(tensor)
        return await broadcast(local_t, root_rank=0, conns=mesh[r], rank=r)

    results = await asyncio.gather(*(run_rank(r) for r in range(world_size)))
    for res in results:
        np.testing.assert_array_equal(res, tensor)

    with pytest.raises(ValueError, match="Connection to root rank"):
        await broadcast(tensor, root_rank=1, conns=[None, None], rank=0)


@pytest.mark.asyncio
async def test_ring_all_reduce_collective() -> None:
    """Test Ring AllReduce collective across cyclic topology."""
    world_size = 3

    single = np.array([1.0, 2.0])
    np.testing.assert_array_equal(await ring_all_reduce(single, "SUM", [None], 0), single)

    ch_a, ch_b = make_bipipe_pair()
    with pytest.raises(ValueError, match="Unsupported reduction operator"):
        await ring_all_reduce(single, "INVALID", [ch_a, ch_b], 0)

    # Missing neighbor connection: rank 0 expects rank 1 (next) and rank 2 (prev)
    with pytest.raises(ValueError, match="Ring topology requires connections"):
        await ring_all_reduce(single, "SUM", [None, None, ch_b], 0)

    for op in ("SUM", "PROD", "MAX", "MIN"):
        mesh_op = create_interconnected_mesh(world_size)
        inputs = [np.arange(6, dtype=np.float32) + (r + 1) * 2 for r in range(world_size)]

        async def run_ring(r: int, op_name: str = op, m: list[list[Optional[BiPipeChannel]]] = mesh_op, inp: list[np.ndarray] = inputs) -> np.ndarray:
            return await ring_all_reduce(inp[r], op_name, m[r], r)

        outputs = await asyncio.gather(*(run_ring(r) for r in range(world_size)))
        if op == "SUM":
            expected = sum(inputs)
        elif op == "PROD":
            expected = inputs[0] * inputs[1] * inputs[2]
        elif op == "MAX":
            expected = np.maximum(inputs[0], np.maximum(inputs[1], inputs[2]))
        else:
            expected = np.minimum(inputs[0], np.minimum(inputs[1], inputs[2]))

        for out in outputs:
            np.testing.assert_allclose(out, expected)


@pytest.mark.asyncio
async def test_all_reduce_direct_and_ring() -> None:
    """Test AllReduce collective for direct algorithm with all ops."""
    world_size = 3
    for op in ("SUM", "PROD", "MAX", "MIN"):
        mesh = create_interconnected_mesh(world_size)
        inputs = [np.arange(6, dtype=np.float32) + (r + 1) for r in range(world_size)]

        async def run_direct(r: int, op_name: str = op, m: list[list[Optional[BiPipeChannel]]] = mesh, inp: list[np.ndarray] = inputs) -> np.ndarray:
            return await all_reduce(inp[r], op_name, m[r], r, algorithm="direct")

        outputs = await asyncio.gather(*(run_direct(r) for r in range(world_size)))
        if op == "SUM":
            expected = sum(inputs)
        elif op == "PROD":
            expected = inputs[0] * inputs[1] * inputs[2]
        elif op == "MAX":
            expected = np.maximum(inputs[0], np.maximum(inputs[1], inputs[2]))
        else:
            expected = np.minimum(inputs[0], np.minimum(inputs[1], inputs[2]))

        for out in outputs:
            np.testing.assert_allclose(out, expected)

    # Test ring algorithm dispatch from all_reduce
    mesh_ring = create_interconnected_mesh(world_size)
    inputs_ring = [np.arange(6, dtype=np.float32) + (r + 1) for r in range(world_size)]

    async def run_ring_disp(r: int) -> np.ndarray:
        return await all_reduce(inputs_ring[r], "SUM", mesh_ring[r], r, algorithm="ring")

    outputs_ring = await asyncio.gather(*(run_ring_disp(r) for r in range(world_size)))
    for out in outputs_ring:
        np.testing.assert_allclose(out, sum(inputs_ring))

    # Unsupported op with peers
    mesh_bad = create_interconnected_mesh(2)

    async def run_bad_op():
        task1 = all_reduce(np.array([1.0]), "BAD_OP", mesh_bad[0], 0, algorithm="direct")
        task2 = all_reduce(np.array([1.0]), "BAD_OP", mesh_bad[1], 1, algorithm="direct")
        await asyncio.gather(task1, task2)

    with pytest.raises(ValueError, match="Unsupported reduction operator"):
        await run_bad_op()


@pytest.mark.asyncio
async def test_all_gather_direct_and_recursive_doubling() -> None:
    """Test AllGather collective using direct and recursive doubling algorithms."""
    # 1. Direct AllGather (3 ranks)
    world_size_3 = 3
    mesh_dir = create_interconnected_mesh(world_size_3)
    inputs_3 = [np.array([[r * 10, r * 10 + 1]], dtype=np.float32) for r in range(world_size_3)]

    async def run_direct(r: int) -> np.ndarray:
        return await all_gather(inputs_3[r], axis=0, conns=mesh_dir[r], rank=r, algorithm="direct")

    outputs_dir = await asyncio.gather(*(run_direct(r) for r in range(world_size_3)))
    expected_gather_3 = np.concatenate(inputs_3, axis=0)
    for out in outputs_dir:
        np.testing.assert_array_equal(out, expected_gather_3)

    # 2. Recursive Doubling AllGather (power of 2: 2 ranks)
    world_size_2 = 2
    mesh_rd = create_interconnected_mesh(world_size_2)
    inputs_2 = [np.array([[r * 10, r * 10 + 1]], dtype=np.float32) for r in range(world_size_2)]

    async def run_rd(r: int) -> np.ndarray:
        return await all_gather(inputs_2[r], axis=0, conns=mesh_rd[r], rank=r, algorithm="recursive_doubling")

    outputs_rd = await asyncio.gather(*(run_rd(r) for r in range(world_size_2)))
    expected_gather_2 = np.concatenate(inputs_2, axis=0)
    for out in outputs_rd:
        np.testing.assert_array_equal(out, expected_gather_2)

    single = np.array([5.0])
    np.testing.assert_array_equal(await recursive_halving_doubling_all_gather(single, 0, [None], 0), single)


@pytest.mark.asyncio
async def test_reduce_scatter_and_ring() -> None:
    """Test ReduceScatter and Ring ReduceScatter collective operations."""
    world_size = 3
    mesh_ring = create_interconnected_mesh(world_size)
    inputs = [np.arange(6, dtype=np.float32) + (r + 1) for r in range(world_size)]

    # 1. Ring ReduceScatter across ops (SUM, PROD, MAX, MIN)
    for op in ("SUM", "PROD", "MAX", "MIN"):
        mesh_op = create_interconnected_mesh(world_size)

        async def run_ring_op(r: int, op_name: str = op, m: list[list[Optional[BiPipeChannel]]] = mesh_op, inp: list[np.ndarray] = inputs) -> np.ndarray:
            return await reduce_scatter(inp[r], op_name, scatter_dim=0, conns=m[r], rank=r, algorithm="ring")

        res_ring = await asyncio.gather(*(run_ring_op(r) for r in range(world_size)))
        if op == "SUM":
            tot = sum(inputs)
        elif op == "PROD":
            tot = inputs[0] * inputs[1] * inputs[2]
        elif op == "MAX":
            tot = np.maximum(inputs[0], np.maximum(inputs[1], inputs[2]))
        else:
            tot = np.minimum(inputs[0], np.minimum(inputs[1], inputs[2]))
        exp_chunks = np.array_split(tot, world_size, axis=0)
        for r in range(world_size):
            np.testing.assert_allclose(res_ring[r], exp_chunks[r])

    mesh_dir = create_interconnected_mesh(world_size)

    async def run_dir_rs(r: int) -> np.ndarray:
        return await reduce_scatter(inputs[r], "SUM", scatter_dim=0, conns=mesh_dir[r], rank=r, algorithm="direct")

    res_dir = await asyncio.gather(*(run_dir_rs(r) for r in range(world_size)))
    exp_dir_chunks = np.array_split(sum(inputs), world_size, axis=0)
    for r in range(world_size):
        np.testing.assert_array_equal(res_dir[r], exp_dir_chunks[r])

    single = np.array([1.0, 2.0])
    np.testing.assert_array_equal(await ring_reduce_scatter(single, "SUM", 0, [None], 0), single)

    ch_a, ch_b = make_bipipe_pair()
    with pytest.raises(ValueError, match="Unsupported reduction operator"):
        await ring_reduce_scatter(single, "BAD", 0, [ch_a, ch_b], 0)

    # Missing neighbor connection: rank 0 expects rank 1 (next) and rank 2 (prev)
    with pytest.raises(ValueError, match="Ring topology requires connections"):
        await ring_reduce_scatter(single, "SUM", 0, [None, None, ch_b], 0)


@pytest.mark.asyncio
async def test_all_to_all_collective() -> None:
    """Test AllToAll collective data redistribution."""
    world_size = 3
    mesh = create_interconnected_mesh(world_size)
    inputs = [np.arange(6, dtype=np.float32).reshape(3, 2) * 10 + r for r in range(world_size)]

    async def run_all_to_all(r: int) -> np.ndarray:
        return await all_to_all(inputs[r], scatter_dim=0, gather_dim=0, conns=mesh[r], rank=r)

    outputs = await asyncio.gather(*(run_all_to_all(r) for r in range(world_size)))
    for r in range(world_size):
        expected_r = np.concatenate([np.array_split(inputs[src], world_size, axis=0)[r] for src in range(world_size)], axis=0)
        np.testing.assert_array_equal(outputs[r], expected_r)


@pytest.mark.asyncio
async def test_distributed_barrier() -> None:
    """Test DistributedBarrier synchronization."""
    world_size = 3
    mesh = create_interconnected_mesh(world_size)
    barriers = [DistributedBarrier(world_size=world_size, rank=r, leader_rank=0) for r in range(world_size)]

    order = []

    async def run_barrier(r: int):
        order.append(f"enter_{r}")
        await barriers[r].wait(mesh[r])
        order.append(f"exit_{r}")

    await asyncio.gather(*(run_barrier(r) for r in range(world_size)))
    assert len(order) == 6


def test_dispatch_collective_and_registry() -> None:
    """Test collective dispatcher and backend registry."""

    def custom_disp(op: str, tensor: np.ndarray, **kwargs: object) -> np.ndarray:
        return tensor * 100.0

    register_collective_backend("custom_be", custom_disp)
    assert get_collective_backend("custom_be") is custom_disp

    arr = np.array([1.0, 2.0])
    res_custom = dispatch_collective("AllReduce", arr, backend="custom_be")
    np.testing.assert_array_equal(res_custom, arr * 100.0)

    t0 = np.array([1.0, 2.0])
    t1 = np.array([3.0, 4.0])
    all_data = [t0, t1]

    # AllReduce SUM, ADD, PROD, PRODUCT, MAX, MIN
    assert np.allclose(dispatch_collective("AllReduce", t0, backend="numpy", op_type="SUM", all_ranks_data=all_data), t0 + t1)
    assert np.allclose(dispatch_collective("AllReduce", t0, backend="numpy", op_type="ADD", all_ranks_data=all_data), t0 + t1)
    assert np.allclose(dispatch_collective("AllReduce", t0, backend="numpy", op_type="PROD", all_ranks_data=all_data), t0 * t1)
    assert np.allclose(dispatch_collective("AllReduce", t0, backend="numpy", op_type="PRODUCT", all_ranks_data=all_data), t0 * t1)
    assert np.allclose(dispatch_collective("AllReduce", t0, backend="numpy", op_type="MAX", all_ranks_data=all_data), np.maximum(t0, t1))
    assert np.allclose(dispatch_collective("AllReduce", t0, backend="numpy", op_type="MIN", all_ranks_data=all_data), np.minimum(t0, t1))

    gathered = dispatch_collective("AllGather", t0, backend="numpy", axis=0, all_ranks_data=all_data)
    assert np.allclose(gathered, np.concatenate(all_data, axis=0))

    # ReduceScatter ops: SUM, ADD, PROD, PRODUCT, MAX, MIN
    assert np.allclose(dispatch_collective("ReduceScatter", t0, backend="numpy", op_type="SUM", scatter_dim=0, rank=0, all_ranks_data=all_data), np.array_split(t0 + t1, 2, axis=0)[0])
    assert np.allclose(dispatch_collective("ReduceScatter", t0, backend="numpy", op_type="ADD", scatter_dim=0, rank=0, all_ranks_data=all_data), np.array_split(t0 + t1, 2, axis=0)[0])
    assert np.allclose(dispatch_collective("ReduceScatter", t0, backend="numpy", op_type="PROD", scatter_dim=0, rank=0, all_ranks_data=all_data), np.array_split(t0 * t1, 2, axis=0)[0])
    assert np.allclose(dispatch_collective("ReduceScatter", t0, backend="numpy", op_type="PRODUCT", scatter_dim=0, rank=0, all_ranks_data=all_data), np.array_split(t0 * t1, 2, axis=0)[0])
    assert np.allclose(dispatch_collective("ReduceScatter", t0, backend="numpy", op_type="MAX", scatter_dim=0, rank=0, all_ranks_data=all_data), np.array_split(np.maximum(t0, t1), 2, axis=0)[0])
    assert np.allclose(dispatch_collective("ReduceScatter", t0, backend="numpy", op_type="MIN", scatter_dim=0, rank=0, all_ranks_data=all_data), np.array_split(np.minimum(t0, t1), 2, axis=0)[0])

    a2a = dispatch_collective("AllToAll", t0, backend="numpy", scatter_dim=0, gather_dim=0, rank=0, all_ranks_data=all_data)
    assert a2a is not None

    bcast = dispatch_collective("Broadcast", t1, backend="numpy", root=0, all_ranks_data=all_data)
    assert np.allclose(bcast, t0)

    fallback = dispatch_collective("AllReduce", t0, backend="numpy")
    assert np.allclose(fallback, t0)

    # PyTorch backend import failure fallback
    with patch.dict(sys.modules, {"ml_switcheroo_compiler.backends.pytorch.distributed_collectives": None}):
        fb_pt = dispatch_collective("AllReduce", t0, backend="pytorch", all_ranks_data=all_data)
        assert np.allclose(fb_pt, t0 + t1)

    # JAX backend import failure fallback
    with patch.dict(sys.modules, {"ml_switcheroo_compiler.backends.jax.distributed_collectives": None}):
        fb_jax = dispatch_collective("AllReduce", t0, backend="jax", all_ranks_data=all_data)
        assert np.allclose(fb_jax, t0 + t1)

    # PyTorch backend mock dispatch
    mock_pt = MagicMock()
    with patch.dict(sys.modules, {"ml_switcheroo_compiler.backends.pytorch.distributed_collectives": mock_pt}):
        mock_pt.pytorch_all_reduce.return_value = t0 + 1
        mock_pt.pytorch_all_gather.return_value = t0 + 2
        mock_pt.pytorch_reduce_scatter.return_value = t0 + 3
        mock_pt.pytorch_broadcast.return_value = t0 + 4

        res_ar = dispatch_collective("AllReduce", t0, backend="pytorch")
        assert np.allclose(res_ar, t0 + 1)

        res_ag = dispatch_collective("AllGather", t0, backend="pytorch")
        assert np.allclose(res_ag, t0 + 2)

        res_rs = dispatch_collective("ReduceScatter", t0, backend="pytorch")
        assert np.allclose(res_rs, t0 + 3)

        res_bc = dispatch_collective("Broadcast", t0, backend="pytorch")
        assert np.allclose(res_bc, t0 + 4)

    # JAX backend mock dispatch
    mock_jax = MagicMock()
    with patch.dict(sys.modules, {"ml_switcheroo_compiler.backends.jax.distributed_collectives": mock_jax}):
        mock_jax.jax_all_reduce.return_value = t0 + 10
        mock_jax.jax_all_gather.return_value = t0 + 20
        mock_jax.jax_broadcast.return_value = t0 + 30

        res_jar = dispatch_collective("AllReduce", t0, backend="jax")
        assert np.allclose(res_jar, t0 + 10)

        res_jag = dispatch_collective("AllGather", t0, backend="jax")
        assert np.allclose(res_jag, t0 + 20)

        res_jbc = dispatch_collective("Broadcast", t0, backend="jax")
        assert np.allclose(res_jbc, t0 + 30)


@pytest.mark.asyncio
async def test_webrtc_collectives_error_handling() -> None:
    """Test fallback branches including None partners, endianness conversion, and dispatch fallthroughs."""
    # Barrier when rank != leader and conn is None
    b_none = DistributedBarrier(world_size=2, rank=1, leader_rank=0)
    await b_none.wait([None, None])

    # Recursive doubling when partner_conn is None
    res_rd_none = await recursive_halving_doubling_all_gather(np.array([1.0]), 0, [None, None], 0)
    assert res_rd_none is not None

    # Byteorder '=' in _send_data
    ch_eq1, ch_eq2 = make_bipipe_pair()
    arr_eq = np.array([1, 2], dtype="=i4")
    send_task = _send_data(ch_eq1, arr_eq)
    recv_task = _recv_data(ch_eq2)
    _, res_eq = await asyncio.gather(send_task, recv_task)
    np.testing.assert_array_equal(res_eq, arr_eq)

    # Endian swap in _recv_data
    ch_f1, ch_f2 = make_bipipe_pair()
    foreign_order = ">" if sys.byteorder == "little" else "<"
    import json

    f_meta = {
        "protocol": "webrtc_stream_v1",
        "shape": [2],
        "dtype": f"{foreign_order}i4",
        "byteorder": foreign_order,
        "total_bytes": 8,
        "num_chunks": 1,
        "chunk_size": 8,
    }
    raw_f = np.array([10, 20], dtype=np.dtype(f"{foreign_order}i4")).tobytes()
    await ch_f1.send(b"META:" + json.dumps(f_meta).encode("utf-8"))
    header_f = struct.pack("!6sIII", b"CHUNK:", 0, 1, zlib.crc32(raw_f))
    await ch_f1.send(header_f + raw_f)
    res_f = await _recv_data(ch_f2)
    assert res_f is not None

    # Raw non-CHUNK frame in _recv_data (hits line 246)
    ch_raw1, ch_raw2 = make_bipipe_pair()
    meta_raw = {
        "protocol": "webrtc_stream_v1",
        "shape": [2],
        "dtype": "int32",
        "total_bytes": 8,
        "num_chunks": 1,
        "chunk_size": 8,
    }
    await ch_raw1.send(b"META:" + json.dumps(meta_raw).encode("utf-8"))
    await ch_raw1.send(np.array([1, 2], dtype=np.int32).tobytes())
    res_raw = await _recv_data(ch_raw2)
    assert res_raw is not None

    # Empty queue with channel lacking recv method (hits line 166)
    class NoRecvChannel:
        pass

    st_empty = WebRTCDataChannelStream(NoRecvChannel())

    async def put_msg():
        await asyncio.sleep(0.01)
        st_empty._recv_queue.put_nowait(b"delayed")

    asyncio.create_task(put_msg())
    assert await st_empty.recv_bytes() == b"delayed"

    # Timeout loop in wait_backpressure (hits line 121 TimeoutError)
    mock_to = MockChannel()
    st_to = WebRTCDataChannelStream(mock_to, high_watermark=10, low_watermark=5)
    mock_to.bufferedAmount = 20

    async def drain_later():
        await asyncio.sleep(0.02)
        mock_to.bufferedAmount = 2
        st_to._on_buffered_amount_low(None)

    asyncio.create_task(drain_later())
    await st_to.wait_backpressure()

    # Fallthrough for unknown op in pytorch / jax
    t = np.array([1.0, 2.0])
    all_data = [t, t]
    with patch.dict(sys.modules, {"ml_switcheroo_compiler.backends.pytorch.distributed_collectives": MagicMock()}):
        res_pt_un = dispatch_collective("UnknownOp", t, backend="pytorch", all_ranks_data=all_data)
        np.testing.assert_array_equal(res_pt_un, t)

    with patch.dict(sys.modules, {"ml_switcheroo_compiler.backends.jax.distributed_collectives": MagicMock()}):
        res_jax_un = dispatch_collective("UnknownOp", t, backend="jax", all_ranks_data=all_data)
        np.testing.assert_array_equal(res_jax_un, t)

    # Fallthrough when all_ranks_data is None for Broadcast
    res_bcast_none = dispatch_collective("Broadcast", t, backend="numpy")
    np.testing.assert_array_equal(res_bcast_none, t)

    # Unknown op_type in AllReduce and ReduceScatter with all_ranks_data
    assert np.allclose(dispatch_collective("AllReduce", t, backend="numpy", op_type="UNKNOWN", all_ranks_data=all_data), t)
    assert np.allclose(dispatch_collective("ReduceScatter", t, backend="numpy", op_type="UNKNOWN", scatter_dim=0, rank=0, all_ranks_data=all_data), np.array_split(t, 2, axis=0)[0])
