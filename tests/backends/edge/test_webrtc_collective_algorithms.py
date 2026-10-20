"""Unit tests for WebRTC distributed collective communication algorithms and streaming."""

from __future__ import annotations

import asyncio
import struct
import sys
import typing
import zlib
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

if typing.TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import EagerCallableProtocol, EagerTensorProtocol, EagerValue


class MockChannel:
    """Mock RTCDataChannel for testing streaming and event handlers."""

    def __init__(self) -> None:
        """Initialize mock channel."""
        self.sent_messages: list[bytes] = []
        self.bufferedAmount: int = 0
        self.bufferedAmountLowThreshold: int = 0
        self.onmessage: EagerValue | None = None
        self.onbufferedamountlow: EagerValue | None = None

    async def send(self, data: bytes) -> None:
        """Mock async send.

        Args:
            data: Test parameter `data`.
        """
        self.sent_messages.append(data)


class MockSnakeChannel:
    """Mock RTCDataChannel with snake_case attributes."""

    def __init__(self) -> None:
        """Initialize mock snake_case channel."""
        self.sent: list[bytes] = []
        self.buffered_amount: int = 0
        self.buffered_amount_low_threshold: int = 0
        self.on_message: EagerValue | None = None
        self.on_buffered_amount_low: EagerValue | None = None

    def sendMessage(self, data: bytes) -> None:
        """Mock sync sendMessage.

        Args:
            data: Test parameter `data`.
        """
        self.sent.append(data)


class BiPipeChannel:
    """In-memory bidirectional paired pipe simulating WebRTC DataChannel."""

    def __init__(
        self,
        send_queue: asyncio.Queue[bytes],
        recv_queue: asyncio.Queue[bytes],
    ) -> None:
        """Initialize bidirectional pipe with complementary queues.

        Args:
            send_queue: Test parameter `send_queue`.
            recv_queue: Test parameter `recv_queue`.
        """
        self.send_queue = send_queue
        self.recv_queue = recv_queue
        self.bufferedAmount: int = 0

    async def send(self, data: bytes) -> None:
        """Send by putting into send queue.

        Args:
            data: Test parameter `data`.
        """
        await self.send_queue.put(data)

    async def recv(self) -> bytes:
        """Receive by pulling from recv queue."""
        return await self.recv_queue.get()


def make_bipipe_pair() -> tuple[BiPipeChannel, BiPipeChannel]:
    """Create a complementary pair of bidirectional pipe channels."""
    q_a2b: asyncio.Queue[bytes] = asyncio.Queue()
    q_b2a: asyncio.Queue[bytes] = asyncio.Queue()
    return BiPipeChannel(q_a2b, q_b2a), BiPipeChannel(q_b2a, q_a2b)


def create_interconnected_mesh(world_size: int) -> list[list[BiPipeChannel | None]]:
    """Create a fully connected mesh of simulated bidirectional pipe channels.

    Args:
        world_size: Test parameter `world_size`.
    """
    conns: list[list[BiPipeChannel | None]] = [[None for _ in range(world_size)] for _ in range(world_size)]
    for i in range(world_size):
        for j in range(i + 1, world_size):
            ch_ij, ch_ji = make_bipipe_pair()
            conns[i][j] = ch_ij
            conns[j][i] = ch_ji
    return conns


async def _test_stream_event_handlers_and_messages_part_1(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 1 of test_stream_event_handlers_and_messages.

    Args:
        _state: Shared test state dictionary.
    """
    mock = MockChannel()
    stream = typing.cast("EagerCallableProtocol", WebRTCDataChannelStream)(mock, high_watermark=100, low_watermark=20)
    assert id(stream.channel) == id(mock)

    stream._on_message(b"bytes_msg")
    assert await stream.recv_bytes() == b"bytes_msg"

    stream._on_message("str_msg")
    assert await stream.recv_bytes() == b"str_msg"

    class ToBytesObj:
        """Test class ToBytesObj."""

        def tobytes(self) -> bytes:
            """Test function tobytes."""
            return b"tobytes_msg"

    typing.cast("EagerCallableProtocol", stream._on_message)(typing.cast("EagerCallableProtocol", ToBytesObj)())
    assert await stream.recv_bytes() == b"tobytes_msg"

    stream._on_message(bytearray(b"bytearray_msg"))
    assert await stream.recv_bytes() == b"bytearray_msg"

    mock_snake = MockSnakeChannel()
    stream_snake = typing.cast("EagerCallableProtocol", WebRTCDataChannelStream)(mock_snake)
    assert mock_snake.on_message is not None

    await stream_snake.send_bytes(b"hello")
    assert mock_snake.sent == [b"hello"]
    getattr(_state, "update")({"ToBytesObj": ToBytesObj, "mock": mock, "mock_snake": mock_snake, "stream": stream})
    getattr(_state, "update")({"stream_snake": stream_snake})
    _state.update(locals())


async def _test_stream_event_handlers_and_messages_part_2(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 2 of test_stream_event_handlers_and_messages.

    Args:
        _state: Shared test state dictionary.
    """

    class CallableAmount:
        """Test class CallableAmount."""

        def bufferedAmount(self) -> int:
            """Test function bufferedAmount."""
            return 42

    stream_cb = typing.cast("EagerCallableProtocol", WebRTCDataChannelStream)(typing.cast("EagerCallableProtocol", CallableAmount)())
    assert stream_cb.get_buffered_amount() == 42

    # Async sendMessage channel
    class AsyncSendMessageChannel:
        """Test class AsyncSendMessageChannel."""

        def __init__(self) -> None:
            """Test function __init__."""
            self.msgs = typing.cast("list[EagerTensorProtocol]", [])

        async def sendMessage(self, data: bytes) -> None:
            """Test function sendMessage.

            Args:
                data: Test parameter `data`.
            """
            typing.cast("EagerCallableProtocol", self.msgs.append)(data)

    ch_async_sendmsg = AsyncSendMessageChannel()
    st_async_sendmsg = typing.cast("EagerCallableProtocol", WebRTCDataChannelStream)(ch_async_sendmsg)
    await st_async_sendmsg.send_bytes(b"async_sendmsg")
    assert ch_async_sendmsg.msgs == [b"async_sendmsg"]
    getattr(_state, "update")({"AsyncSendMessageChannel": AsyncSendMessageChannel, "CallableAmount": CallableAmount, "ch_async_sendmsg": ch_async_sendmsg, "st_async_sendmsg": st_async_sendmsg})
    getattr(_state, "update")({"stream_cb": stream_cb})
    _state.update(locals())


async def _test_stream_event_handlers_and_messages_part_3(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 3 of test_stream_event_handlers_and_messages.

    Args:
        _state: Shared test state dictionary.
    """

    class SyncSendCoroutineChannel:
        """Test class SyncSendCoroutineChannel."""

        def __init__(self) -> None:
            """Test function __init__."""
            self.msgs = typing.cast("list[EagerTensorProtocol]", [])

        def send(self, data: bytes) -> EagerValue:
            """Test function send.

            Args:
                data: Test parameter `data`.
            """

            async def _coro() -> None:
                """Test function _coro."""
                typing.cast("EagerCallableProtocol", self.msgs.append)(data)

            return typing.cast("EagerTensorProtocol", _coro())

    ch_sync_coro = SyncSendCoroutineChannel()
    st_sync_coro = typing.cast("EagerCallableProtocol", WebRTCDataChannelStream)(ch_sync_coro)
    await st_sync_coro.send_bytes(b"coro_msg")
    assert ch_sync_coro.msgs == [b"coro_msg"]
    getattr(_state, "update")({"SyncSendCoroutineChannel": SyncSendCoroutineChannel, "ch_sync_coro": ch_sync_coro, "st_sync_coro": st_sync_coro})
    _state.update(locals())


async def _test_stream_event_handlers_and_messages_part_4(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 4 of test_stream_event_handlers_and_messages.

    Args:
        _state: Shared test state dictionary.
    """

    class SyncSendNoneChannel:
        """Test class SyncSendNoneChannel."""

        def __init__(self) -> None:
            """Test function __init__."""
            self.msgs = typing.cast("list[EagerTensorProtocol]", [])

        def send(self, data: bytes) -> None:
            """Test function send.

            Args:
                data: Test parameter `data`.
            """
            typing.cast("EagerCallableProtocol", self.msgs.append)(data)

    ch_sync_none = SyncSendNoneChannel()
    st_sync_none = typing.cast("EagerCallableProtocol", WebRTCDataChannelStream)(ch_sync_none)
    await st_sync_none.send_bytes(b"none_msg")
    assert ch_sync_none.msgs == [b"none_msg"]

    # Sync recv returning coroutine
    class SyncRecvChannel:
        """Test class SyncRecvChannel."""

        def recv(self) -> EagerValue:
            """Test function recv."""

            async def _r() -> EagerValue:
                """Test function _r."""
                return b"from_recv"

            return typing.cast("EagerTensorProtocol", _r())

    st_recv = typing.cast("EagerCallableProtocol", WebRTCDataChannelStream)(typing.cast("EagerCallableProtocol", SyncRecvChannel)())
    assert await st_recv.recv_bytes() == b"from_recv"
    getattr(_state, "update")({"SyncRecvChannel": SyncRecvChannel, "SyncSendNoneChannel": SyncSendNoneChannel, "ch_sync_none": ch_sync_none, "st_recv": st_recv})
    getattr(_state, "update")({"st_sync_none": st_sync_none})
    _state.update(locals())


async def _test_stream_event_handlers_and_messages_part_5(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 5 of test_stream_event_handlers_and_messages.

    Args:
        _state: Shared test state dictionary.
    """

    class NonCoroRecvChannel:
        """Test class NonCoroRecvChannel."""

        def recv(self) -> EagerValue:
            """Test function recv."""
            return b"non_coro_bytes"

    st_noncoro = typing.cast("EagerCallableProtocol", WebRTCDataChannelStream)(typing.cast("EagerCallableProtocol", NonCoroRecvChannel)())
    assert await st_noncoro.recv_bytes() == b"non_coro_bytes"

    # Non-coroutine recv returning bytearray
    class BytearrayRecvChannel:
        """Test class BytearrayRecvChannel."""

        def recv(self) -> EagerValue:
            """Test function recv."""
            return bytearray(b"bytearray_res")

    st_ba = typing.cast("EagerCallableProtocol", WebRTCDataChannelStream)(typing.cast("EagerCallableProtocol", BytearrayRecvChannel)())
    assert await st_ba.recv_bytes() == b"bytearray_res"

    # Property buffered_amount
    class PropAmount:
        """Test class PropAmount."""

        buffered_amount = 99

    st_prop = typing.cast("EagerCallableProtocol", WebRTCDataChannelStream)(typing.cast("EagerCallableProtocol", PropAmount)())
    assert st_prop.get_buffered_amount() == 99
    await st_prop.send_bytes(b"ignored")
    getattr(_state, "update")({"BytearrayRecvChannel": BytearrayRecvChannel, "NonCoroRecvChannel": NonCoroRecvChannel, "PropAmount": PropAmount, "st_ba": st_ba})
    getattr(_state, "update")({"st_noncoro": st_noncoro, "st_prop": st_prop})
    _state.update(locals())


async def _test_stream_event_handlers_and_messages_part_6(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 6 of test_stream_event_handlers_and_messages.

    Args:
        _state: Shared test state dictionary.
    """

    class SyncSendMessageCoroChannel:
        """Test class SyncSendMessageCoroChannel."""

        def __init__(self) -> None:
            """Test function __init__."""
            self.msgs = typing.cast("list[EagerTensorProtocol]", [])

        def sendMessage(self, data: bytes) -> EagerValue:
            """Test function sendMessage.

            Args:
                data: Test parameter `data`.
            """

            async def _c() -> None:
                """Test function _c."""
                typing.cast("EagerCallableProtocol", self.msgs.append)(data)

            return typing.cast("EagerTensorProtocol", _c())

    st_sync_sendmsg = typing.cast("EagerCallableProtocol", WebRTCDataChannelStream)(typing.cast("EagerCallableProtocol", SyncSendMessageCoroChannel)())
    await st_sync_sendmsg.send_bytes(b"sync_sendmsg_coro")
    assert len(typing.cast("EagerTensorProtocol", getattr(st_sync_sendmsg.channel, "msgs"))) == 1
    getattr(_state, "update")({"SyncSendMessageCoroChannel": SyncSendMessageCoroChannel, "st_sync_sendmsg": st_sync_sendmsg})


@pytest.mark.asyncio
async def test_stream_event_handlers_and_messages() -> None:
    """Test WebRTCDataChannelStream event listener attachment and message receiving."""
    _state: dict[str, EagerTensorProtocol] = {}
    await _test_stream_event_handlers_and_messages_part_1(_state)
    await _test_stream_event_handlers_and_messages_part_2(_state)
    await _test_stream_event_handlers_and_messages_part_3(_state)
    await _test_stream_event_handlers_and_messages_part_4(_state)
    await _test_stream_event_handlers_and_messages_part_5(_state)
    await _test_stream_event_handlers_and_messages_part_6(_state)
    assert len(_state) > 0


@pytest.mark.asyncio
async def test_stream_backpressure() -> None:
    """Test wait_backpressure handling with drain callbacks and events."""
    mock = MockChannel()
    stream = typing.cast("EagerCallableProtocol", WebRTCDataChannelStream)(mock, high_watermark=50, low_watermark=10)

    mock.bufferedAmount = 20
    await stream.wait_backpressure()

    drain_called = False

    async def mock_wait_drain() -> None:
        """Test function mock_wait_drain."""
        nonlocal drain_called
        drain_called = True
        mock.bufferedAmount = 5

    mock.bufferedAmount = 100
    setattr(mock, "wait_drain", mock_wait_drain)
    await stream.wait_backpressure()
    assert drain_called is True

    mock.bufferedAmount = 100

    def mock_sync_drain() -> None:
        """Test function mock_sync_drain."""
        mock.bufferedAmount = 5

    setattr(mock, "wait_drain", mock_sync_drain)
    await stream.wait_backpressure()

    delattr(mock, "wait_drain")
    mock.bufferedAmount = 100

    async def trigger_drain() -> None:
        """Test function trigger_drain."""
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

    send_task = typing.cast("EagerCallableProtocol", _send_data)(ch_a, data, chunk_size=64)
    recv_task = typing.cast("EagerCallableProtocol", _recv_data)(ch_b)
    _, received = await asyncio.gather(send_task, recv_task)
    typing.cast("EagerCallableProtocol", np.testing.assert_array_equal)(received, data)

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
    res_endian = await typing.cast("EagerCallableProtocol", _recv_data)(ch_end2)
    assert res_endian is not None

    # CRC error test
    ch_bad1, ch_bad2 = make_bipipe_pair()
    await ch_bad1.send(b"META:" + json.dumps(meta).encode("utf-8"))
    bad_chksum = 999999
    header = struct.pack("!6sIII", b"CHUNK:", 0, 1, bad_chksum)
    await ch_bad1.send(header + raw)

    with pytest.raises(ValueError, match="Checksum mismatch"):
        await typing.cast("EagerCallableProtocol", _recv_data)(ch_bad2)

    # Payload length mismatch test
    ch_len1, ch_len2 = make_bipipe_pair()
    meta["total_bytes"] = len(raw) + 10
    chksum = zlib.crc32(raw)
    header = struct.pack("!6sIII", b"CHUNK:", 0, 1, chksum)
    await ch_len1.send(b"META:" + json.dumps(meta).encode("utf-8"))
    await ch_len1.send(header + raw)

    with pytest.raises(ValueError, match="Payload length mismatch"):
        await typing.cast("EagerCallableProtocol", _recv_data)(ch_len2)

    # Non-META header rejection test
    ch_leg1, ch_leg2 = make_bipipe_pair()
    await ch_leg1.send(b"2,2|float32")
    with pytest.raises(ValueError, match="expected META: prefix"):
        await typing.cast("EagerCallableProtocol", _recv_data)(ch_leg2)


@pytest.mark.asyncio
async def test_broadcast_collective() -> None:
    """Test Broadcast collective across ranks."""
    world_size = 3
    mesh = create_interconnected_mesh(world_size)
    tensor = np.array([10.0, 20.0, 30.0], dtype=np.float32)

    async def run_rank(r: int) -> np.ndarray[tuple[int, ...], np.dtype[np.generic]]:
        """Test function run_rank.

        Args:
            r: Test parameter `r`.
        """
        local_t = tensor if r == 0 else np.zeros_like(tensor)
        return typing.cast("np.ndarray[tuple[int, ...], np.dtype[np.generic]]", await typing.cast("EagerCallableProtocol", broadcast)(local_t, root_rank=0, conns=mesh[r], rank=r))

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
    typing.cast("EagerCallableProtocol", np.testing.assert_array_equal)(await typing.cast("EagerCallableProtocol", ring_all_reduce)(single, "SUM", [None], 0), single)

    ch_a, ch_b = make_bipipe_pair()
    with pytest.raises(ValueError, match="Unsupported reduction operator"):
        await typing.cast("EagerCallableProtocol", ring_all_reduce)(single, "INVALID", [ch_a, ch_b], 0)

    # Missing neighbor connection: rank 0 expects rank 1 (next) and rank 2 (prev)
    with pytest.raises(ValueError, match="Ring topology requires connections"):
        await typing.cast("EagerCallableProtocol", ring_all_reduce)(single, "SUM", [None, None, ch_b], 0)

    for op in ("SUM", "PROD", "MAX", "MIN"):
        mesh_op = create_interconnected_mesh(world_size)
        inputs = [np.arange(6, dtype=np.float32) + (r + 1) * 2 for r in range(world_size)]

        async def run_ring(r: int, op_name: str = op, m: list[list[BiPipeChannel | None]] = mesh_op, inp: EagerValue = inputs) -> np.ndarray[tuple[int, ...], np.dtype[np.generic]]:
            """Test function run_ring.

            Args:
                r: Test parameter `r`.
                op_name: Test parameter `op_name`.
                m: Test parameter `m`.
                inp: Test parameter `inp`.
            """
            return typing.cast("np.ndarray[tuple[int, ...], np.dtype[np.generic]]", await typing.cast("EagerCallableProtocol", ring_all_reduce)(typing.cast("EagerTensorProtocol", inp)[r], op_name, m[r], r))

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
            typing.cast("EagerCallableProtocol", np.testing.assert_allclose)(out, expected)


@pytest.mark.asyncio
async def test_all_reduce_direct_and_ring() -> None:
    """Test AllReduce collective for direct algorithm with all ops."""
    world_size = 3
    for op in ("SUM", "PROD", "MAX", "MIN"):
        mesh = create_interconnected_mesh(world_size)
        inputs = [np.arange(6, dtype=np.float32) + (r + 1) for r in range(world_size)]

        async def run_direct(r: int, op_name: str = op, m: list[list[BiPipeChannel | None]] = mesh, inp: EagerValue = inputs) -> np.ndarray[tuple[int, ...], np.dtype[np.generic]]:
            """Test function run_direct.

            Args:
                r: Test parameter `r`.
                op_name: Test parameter `op_name`.
                m: Test parameter `m`.
                inp: Test parameter `inp`.
            """
            return typing.cast("np.ndarray[tuple[int, ...], np.dtype[np.generic]]", await typing.cast("EagerCallableProtocol", all_reduce)(typing.cast("EagerTensorProtocol", inp)[r], op_name, m[r], r, algorithm="direct"))

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
            typing.cast("EagerCallableProtocol", np.testing.assert_allclose)(out, expected)

    # Test ring algorithm dispatch from all_reduce
    mesh_ring = create_interconnected_mesh(world_size)
    inputs_ring = [np.arange(6, dtype=np.float32) + (r + 1) for r in range(world_size)]

    async def run_ring_disp(r: int) -> np.ndarray[tuple[int, ...], np.dtype[np.generic]]:
        """Test function run_ring_disp.

        Args:
            r: Test parameter `r`.
        """
        return typing.cast("np.ndarray[tuple[int, ...], np.dtype[np.generic]]", await typing.cast("EagerCallableProtocol", all_reduce)(inputs_ring[r], "SUM", mesh_ring[r], r, algorithm="ring"))

    outputs_ring = await asyncio.gather(*(run_ring_disp(r) for r in range(world_size)))
    for out in outputs_ring:
        typing.cast("EagerCallableProtocol", np.testing.assert_allclose)(out, typing.cast("EagerCallableProtocol", sum)(inputs_ring))

    # Unsupported op with peers
    mesh_bad = create_interconnected_mesh(2)

    async def run_bad_op() -> None:
        """Test function run_bad_op."""
        task1 = typing.cast("EagerCallableProtocol", all_reduce)(typing.cast("EagerCallableProtocol", np.array)([1.0]), "BAD_OP", mesh_bad[0], 0, algorithm="direct")
        task2 = typing.cast("EagerCallableProtocol", all_reduce)(typing.cast("EagerCallableProtocol", np.array)([1.0]), "BAD_OP", mesh_bad[1], 1, algorithm="direct")
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

    async def run_direct(r: int) -> np.ndarray[tuple[int, ...], np.dtype[np.generic]]:
        """Test function run_direct.

        Args:
            r: Test parameter `r`.
        """
        return typing.cast("np.ndarray[tuple[int, ...], np.dtype[np.generic]]", await typing.cast("EagerCallableProtocol", all_gather)(inputs_3[r], axis=0, conns=mesh_dir[r], rank=r, algorithm="direct"))

    outputs_dir = await asyncio.gather(*(run_direct(r) for r in range(world_size_3)))
    expected_gather_3 = np.concatenate(inputs_3, axis=0)
    for out in outputs_dir:
        np.testing.assert_array_equal(out, expected_gather_3)

    # 2. Recursive Doubling AllGather (power of 2: 2 ranks)
    world_size_2 = 2
    mesh_rd = create_interconnected_mesh(world_size_2)
    inputs_2 = [np.array([[r * 10, r * 10 + 1]], dtype=np.float32) for r in range(world_size_2)]

    async def run_rd(r: int) -> np.ndarray[tuple[int, ...], np.dtype[np.generic]]:
        """Test function run_rd.

        Args:
            r: Test parameter `r`.
        """
        return typing.cast("np.ndarray[tuple[int, ...], np.dtype[np.generic]]", await typing.cast("EagerCallableProtocol", all_gather)(inputs_2[r], axis=0, conns=mesh_rd[r], rank=r, algorithm="recursive_doubling"))

    outputs_rd = await asyncio.gather(*(run_rd(r) for r in range(world_size_2)))
    expected_gather_2 = np.concatenate(inputs_2, axis=0)
    for out in outputs_rd:
        np.testing.assert_array_equal(out, expected_gather_2)

    single = np.array([5.0])
    typing.cast("EagerCallableProtocol", np.testing.assert_array_equal)(await typing.cast("EagerCallableProtocol", recursive_halving_doubling_all_gather)(single, 0, [None], 0), single)


@pytest.mark.asyncio
async def test_reduce_scatter_and_ring() -> None:
    """Test ReduceScatter and Ring ReduceScatter collective operations."""
    world_size = 3
    create_interconnected_mesh(world_size)
    inputs = [np.arange(6, dtype=np.float32) + (r + 1) for r in range(world_size)]

    # 1. Ring ReduceScatter across ops (SUM, PROD, MAX, MIN)
    for op in ("SUM", "PROD", "MAX", "MIN"):
        mesh_op = create_interconnected_mesh(world_size)

        async def run_ring_op(r: int, op_name: str = op, m: list[list[BiPipeChannel | None]] = mesh_op, inp: EagerValue = inputs) -> np.ndarray[tuple[int, ...], np.dtype[np.generic]]:
            """Test function run_ring_op.

            Args:
                r: Test parameter `r`.
                op_name: Test parameter `op_name`.
                m: Test parameter `m`.
                inp: Test parameter `inp`.
            """
            return typing.cast("np.ndarray[tuple[int, ...], np.dtype[np.generic]]", await typing.cast("EagerCallableProtocol", reduce_scatter)(typing.cast("EagerTensorProtocol", inp)[r], op_name, scatter_dim=0, conns=m[r], rank=r, algorithm="ring"))

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
            typing.cast("EagerCallableProtocol", np.testing.assert_allclose)(res_ring[r], exp_chunks[r])

    mesh_dir = create_interconnected_mesh(world_size)

    async def run_dir_rs(r: int) -> np.ndarray[tuple[int, ...], np.dtype[np.generic]]:
        """Test function run_dir_rs.

        Args:
            r: Test parameter `r`.
        """
        return typing.cast("np.ndarray[tuple[int, ...], np.dtype[np.generic]]", await typing.cast("EagerCallableProtocol", reduce_scatter)(inputs[r], "SUM", scatter_dim=0, conns=mesh_dir[r], rank=r, algorithm="direct"))

    res_dir = await asyncio.gather(*(run_dir_rs(r) for r in range(world_size)))
    exp_dir_chunks = np.array_split(sum(inputs), world_size, axis=0)
    for r in range(world_size):
        np.testing.assert_array_equal(res_dir[r], exp_dir_chunks[r])

    single = np.array([1.0, 2.0])
    typing.cast("EagerCallableProtocol", np.testing.assert_array_equal)(await typing.cast("EagerCallableProtocol", ring_reduce_scatter)(single, "SUM", 0, [None], 0), single)

    ch_a, ch_b = make_bipipe_pair()
    with pytest.raises(ValueError, match="Unsupported reduction operator"):
        await typing.cast("EagerCallableProtocol", ring_reduce_scatter)(single, "BAD", 0, [ch_a, ch_b], 0)

    # Missing neighbor connection: rank 0 expects rank 1 (next) and rank 2 (prev)
    with pytest.raises(ValueError, match="Ring topology requires connections"):
        await typing.cast("EagerCallableProtocol", ring_reduce_scatter)(single, "SUM", 0, [None, None, ch_b], 0)


@pytest.mark.asyncio
async def test_all_to_all_collective() -> None:
    """Test AllToAll collective data redistribution."""
    world_size = 3
    mesh = create_interconnected_mesh(world_size)
    inputs = [np.arange(6, dtype=np.float32).reshape(3, 2) * 10 + r for r in range(world_size)]

    async def run_all_to_all(r: int) -> np.ndarray[tuple[int, ...], np.dtype[np.generic]]:
        """Test function run_all_to_all.

        Args:
            r: Test parameter `r`.
        """
        return typing.cast("np.ndarray[tuple[int, ...], np.dtype[np.generic]]", await typing.cast("EagerCallableProtocol", all_to_all)(inputs[r], scatter_dim=0, gather_dim=0, conns=mesh[r], rank=r))

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

    async def run_barrier(r: int) -> None:
        """Test function run_barrier.

        Args:
            r: Test parameter `r`.
        """
        order.append(f"enter_{r}")
        await typing.cast("EagerCallableProtocol", barriers[r].wait)(mesh[r])
        order.append(f"exit_{r}")

    await asyncio.gather(*(run_barrier(r) for r in range(world_size)))
    assert len(order) == 6


def _test_dispatch_collective_and_registry_part_1(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 1 of test_dispatch_collective_and_registry.

    Args:
        _state: Shared test state dictionary.
    """

    def custom_disp(op: str, tensor: np.ndarray[tuple[int, ...], np.dtype[np.generic]], **kwargs: EagerValue) -> np.ndarray[tuple[int, ...], np.dtype[np.generic]]:
        """Test function custom_disp.

        Args:
            op: Test parameter `op`.
            tensor: Test parameter `tensor`.
            kwargs: Test parameter `kwargs`.
        """
        return typing.cast("np.ndarray[tuple[int, ...], np.dtype[np.generic]]", typing.cast("EagerTensorProtocol", tensor) * typing.cast("EagerTensorProtocol", 100.0))

    register_collective_backend("custom_be", custom_disp)
    assert get_collective_backend("custom_be") is custom_disp

    arr = np.array([1.0, 2.0])
    res_custom = dispatch_collective("AllReduce", arr, backend="custom_be")
    typing.cast("EagerCallableProtocol", np.testing.assert_array_equal)(res_custom, arr * 100.0)

    t0 = np.array([1.0, 2.0])
    t1 = np.array([3.0, 4.0])
    all_data = [t0, t1]

    # AllReduce SUM, ADD, PROD, PRODUCT, MAX, MIN
    assert typing.cast("EagerCallableProtocol", np.allclose)(typing.cast("EagerCallableProtocol", dispatch_collective)("AllReduce", t0, backend="numpy", op_type="SUM", all_ranks_data=all_data), t0 + t1)
    assert typing.cast("EagerCallableProtocol", np.allclose)(typing.cast("EagerCallableProtocol", dispatch_collective)("AllReduce", t0, backend="numpy", op_type="ADD", all_ranks_data=all_data), t0 + t1)
    assert typing.cast("EagerCallableProtocol", np.allclose)(typing.cast("EagerCallableProtocol", dispatch_collective)("AllReduce", t0, backend="numpy", op_type="PROD", all_ranks_data=all_data), t0 * t1)
    assert typing.cast("EagerCallableProtocol", np.allclose)(typing.cast("EagerCallableProtocol", dispatch_collective)("AllReduce", t0, backend="numpy", op_type="PRODUCT", all_ranks_data=all_data), t0 * t1)
    assert typing.cast("EagerCallableProtocol", np.allclose)(typing.cast("EagerCallableProtocol", dispatch_collective)("AllReduce", t0, backend="numpy", op_type="MAX", all_ranks_data=all_data), typing.cast("EagerCallableProtocol", np.maximum)(t0, t1))
    assert typing.cast("EagerCallableProtocol", np.allclose)(typing.cast("EagerCallableProtocol", dispatch_collective)("AllReduce", t0, backend="numpy", op_type="MIN", all_ranks_data=all_data), typing.cast("EagerCallableProtocol", np.minimum)(t0, t1))

    gathered = dispatch_collective("AllGather", t0, backend="numpy", axis=0, all_ranks_data=all_data)
    assert typing.cast("EagerCallableProtocol", np.allclose)(gathered, typing.cast("EagerCallableProtocol", np.concatenate)(all_data, axis=0))

    # ReduceScatter ops: SUM, ADD, PROD, PRODUCT, MAX, MIN
    assert typing.cast("EagerCallableProtocol", np.allclose)(typing.cast("EagerCallableProtocol", dispatch_collective)("ReduceScatter", t0, backend="numpy", op_type="SUM", scatter_dim=0, rank=0, all_ranks_data=all_data), typing.cast("EagerCallableProtocol", np.array_split)(t0 + t1, 2, axis=0)[0])
    assert typing.cast("EagerCallableProtocol", np.allclose)(typing.cast("EagerCallableProtocol", dispatch_collective)("ReduceScatter", t0, backend="numpy", op_type="ADD", scatter_dim=0, rank=0, all_ranks_data=all_data), typing.cast("EagerCallableProtocol", np.array_split)(t0 + t1, 2, axis=0)[0])
    assert typing.cast("EagerCallableProtocol", np.allclose)(typing.cast("EagerCallableProtocol", dispatch_collective)("ReduceScatter", t0, backend="numpy", op_type="PROD", scatter_dim=0, rank=0, all_ranks_data=all_data), typing.cast("EagerCallableProtocol", np.array_split)(t0 * t1, 2, axis=0)[0])
    getattr(_state, "update")({"all_data": all_data, "arr": arr, "custom_disp": custom_disp, "gathered": gathered})
    getattr(_state, "update")({"res_custom": res_custom, "t0": t0, "t1": t1})
    _state.update(locals())


def _test_dispatch_collective_and_registry_part_2(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 2 of test_dispatch_collective_and_registry.

    Args:
        _state: Shared test state dictionary.
    """
    all_data, t0, t1 = _state["all_data"], _state["t0"], _state["t1"]
    assert typing.cast("EagerCallableProtocol", np.allclose)(
        typing.cast("EagerCallableProtocol", dispatch_collective)("ReduceScatter", t0, backend="numpy", op_type="PRODUCT", scatter_dim=0, rank=0, all_ranks_data=all_data),
        typing.cast("EagerTensorProtocol", getattr(typing.cast("EagerCallableProtocol", np.array_split)(t0 * t1, 2, axis=0), "__getitem__")(0)),
    )
    assert typing.cast("EagerCallableProtocol", np.allclose)(
        typing.cast("EagerCallableProtocol", dispatch_collective)("ReduceScatter", t0, backend="numpy", op_type="MAX", scatter_dim=0, rank=0, all_ranks_data=all_data),
        typing.cast("EagerTensorProtocol", getattr(typing.cast("EagerCallableProtocol", np.array_split)(typing.cast("EagerCallableProtocol", np.maximum)(t0, t1), 2, axis=0), "__getitem__")(0)),
    )
    assert typing.cast("EagerCallableProtocol", np.allclose)(
        typing.cast("EagerCallableProtocol", dispatch_collective)("ReduceScatter", t0, backend="numpy", op_type="MIN", scatter_dim=0, rank=0, all_ranks_data=all_data),
        typing.cast("EagerTensorProtocol", getattr(typing.cast("EagerCallableProtocol", np.array_split)(typing.cast("EagerCallableProtocol", np.minimum)(t0, t1), 2, axis=0), "__getitem__")(0)),
    )
    a2a = dispatch_collective("AllToAll", t0, backend="numpy", scatter_dim=0, gather_dim=0, rank=0, all_ranks_data=all_data)
    assert a2a is not None

    bcast = dispatch_collective("Broadcast", t1, backend="numpy", root=0, all_ranks_data=all_data)
    assert typing.cast("EagerCallableProtocol", np.allclose)(bcast, t0)

    fallback = dispatch_collective("AllReduce", t0, backend="numpy")
    assert typing.cast("EagerCallableProtocol", np.allclose)(fallback, t0)

    # PyTorch backend import failure fallback
    with patch.dict(sys.modules, {"ml_switcheroo_compiler.backends.pytorch.distributed_collectives": None}):
        fb_pt = dispatch_collective("AllReduce", t0, backend="pytorch", all_ranks_data=all_data)
        assert typing.cast("EagerCallableProtocol", np.allclose)(fb_pt, t0 + t1)

    # JAX backend import failure fallback
    with patch.dict(sys.modules, {"ml_switcheroo_compiler.backends.jax.distributed_collectives": None}):
        fb_jax = dispatch_collective("AllReduce", t0, backend="jax", all_ranks_data=all_data)
        assert typing.cast("EagerCallableProtocol", np.allclose)(fb_jax, t0 + t1)

    # PyTorch backend mock dispatch
    mock_pt = MagicMock()
    getattr(_state, "update")({"a2a": a2a, "bcast": bcast, "fallback": fallback, "fb_jax": fb_jax})
    getattr(_state, "update")({"fb_pt": fb_pt, "mock_pt": mock_pt})
    _state.update(locals())


def _test_dispatch_collective_and_registry_part_3(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 3 of test_dispatch_collective_and_registry.

    Args:
        _state: Shared test state dictionary.
    """
    mock_pt, t0 = _state["mock_pt"], _state["t0"]
    with patch.dict(sys.modules, {"ml_switcheroo_compiler.backends.pytorch.distributed_collectives": mock_pt}):
        setattr(mock_pt.pytorch_all_reduce, "return_value", t0 + 1)
        setattr(mock_pt.pytorch_all_gather, "return_value", t0 + 2)
        setattr(mock_pt.pytorch_reduce_scatter, "return_value", t0 + 3)
        setattr(mock_pt.pytorch_broadcast, "return_value", t0 + 4)

        res_ar = dispatch_collective("AllReduce", t0, backend="pytorch")
        assert typing.cast("EagerCallableProtocol", np.allclose)(res_ar, t0 + 1)

        res_ag = dispatch_collective("AllGather", t0, backend="pytorch")
        assert typing.cast("EagerCallableProtocol", np.allclose)(res_ag, t0 + 2)

        res_rs = dispatch_collective("ReduceScatter", t0, backend="pytorch")
        assert typing.cast("EagerCallableProtocol", np.allclose)(res_rs, t0 + 3)

        res_bc = dispatch_collective("Broadcast", t0, backend="pytorch")
        assert typing.cast("EagerCallableProtocol", np.allclose)(res_bc, t0 + 4)

    # JAX backend mock dispatch
    mock_jax = MagicMock()
    getattr(_state, "update")({"mock_jax": mock_jax, "res_ag": res_ag, "res_ar": res_ar, "res_bc": res_bc})
    getattr(_state, "update")({"res_rs": res_rs})
    _state.update(locals())


def _test_dispatch_collective_and_registry_part_4(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 4 of test_dispatch_collective_and_registry.

    Args:
        _state: Shared test state dictionary.
    """
    mock_jax, t0 = _state["mock_jax"], _state["t0"]
    with patch.dict(sys.modules, {"ml_switcheroo_compiler.backends.jax.distributed_collectives": mock_jax}):
        setattr(mock_jax.jax_all_reduce, "return_value", t0 + 10)
        setattr(mock_jax.jax_all_gather, "return_value", t0 + 20)
        setattr(mock_jax.jax_broadcast, "return_value", t0 + 30)

        res_jar = dispatch_collective("AllReduce", t0, backend="jax")
        assert typing.cast("EagerCallableProtocol", np.allclose)(res_jar, t0 + 10)

        res_jag = dispatch_collective("AllGather", t0, backend="jax")
        assert typing.cast("EagerCallableProtocol", np.allclose)(res_jag, t0 + 20)

        res_jbc = dispatch_collective("Broadcast", t0, backend="jax")
        assert typing.cast("EagerCallableProtocol", np.allclose)(res_jbc, t0 + 30)
    getattr(_state, "update")({"res_jag": res_jag, "res_jar": res_jar, "res_jbc": res_jbc})


def test_dispatch_collective_and_registry() -> None:
    """Test collective dispatcher and backend registry."""
    _state: dict[str, EagerTensorProtocol] = {}
    _test_dispatch_collective_and_registry_part_1(_state)
    _test_dispatch_collective_and_registry_part_2(_state)
    _test_dispatch_collective_and_registry_part_3(_state)
    _test_dispatch_collective_and_registry_part_4(_state)
    assert len(_state) > 0


async def _test_webrtc_collectives_error_handling_part_1(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 1 of test_webrtc_collectives_error_handling.

    Args:
        _state: Shared test state dictionary.
    """
    b_none = DistributedBarrier(world_size=2, rank=1, leader_rank=0)
    await b_none.wait([None, None])

    # Recursive doubling when partner_conn is None and world_size=3 (partner_rank >= world_size)
    res_rd_none = await recursive_halving_doubling_all_gather(np.array([1.0]), 0, [None, None, None], 2)
    assert res_rd_none is not None

    # Byteorder '=' in _send_data
    ch_eq1, ch_eq2 = make_bipipe_pair()
    arr_eq = np.array([1, 2], dtype="=i4")
    send_task = typing.cast("EagerCallableProtocol", _send_data)(ch_eq1, arr_eq)
    recv_task = typing.cast("EagerCallableProtocol", _recv_data)(ch_eq2)
    _, res_eq = await asyncio.gather(send_task, recv_task)
    typing.cast("EagerCallableProtocol", np.testing.assert_array_equal)(res_eq, arr_eq)

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
    res_f = await typing.cast("EagerCallableProtocol", _recv_data)(ch_f2)
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
    getattr(_state, "update")({"_": _, "arr_eq": arr_eq, "b_none": b_none, "ch_eq1": ch_eq1})
    getattr(_state, "update")({"ch_eq2": ch_eq2, "ch_f1": ch_f1, "ch_f2": ch_f2, "ch_raw1": ch_raw1})
    getattr(_state, "update")({"ch_raw2": ch_raw2, "f_meta": f_meta, "foreign_order": foreign_order, "header_f": header_f})
    getattr(_state, "update")({"json": json, "meta_raw": meta_raw, "raw_f": raw_f, "recv_task": recv_task})
    getattr(_state, "update")({"res_eq": res_eq, "res_f": res_f, "res_rd_none": res_rd_none, "send_task": send_task})
    _state.update(locals())


async def _test_webrtc_collectives_error_handling_part_2(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 2 of test_webrtc_collectives_error_handling.

    Args:
        _state: Shared test state dictionary.
    """
    ch_raw1, ch_raw2, json, meta_raw = _state["ch_raw1"], _state["ch_raw2"], _state["json"], _state["meta_raw"]
    await ch_raw1.send(b"META:" + json.dumps(meta_raw).encode("utf-8"))
    await ch_raw1.send(np.array([1, 2], dtype=np.int32).tobytes())
    res_raw = await _recv_data(ch_raw2)
    assert res_raw is not None

    # Empty queue with channel lacking recv method (hits line 166)
    class NoRecvChannel:
        """Test class NoRecvChannel."""

        pass

    st_empty = typing.cast("EagerCallableProtocol", WebRTCDataChannelStream)(typing.cast("EagerCallableProtocol", NoRecvChannel)())

    async def put_msg() -> None:
        """Test function put_msg."""
        await asyncio.sleep(0.01)
        st_empty._recv_queue.put_nowait(b"delayed")

    asyncio.create_task(put_msg())
    assert await st_empty.recv_bytes() == b"delayed"

    # Timeout loop in wait_backpressure (hits line 121 TimeoutError)
    mock_to = MockChannel()
    st_to = typing.cast("EagerCallableProtocol", WebRTCDataChannelStream)(mock_to, high_watermark=10, low_watermark=5)
    mock_to.bufferedAmount = 20

    async def drain_later() -> None:
        """Test function drain_later."""
        await asyncio.sleep(0.02)
        mock_to.bufferedAmount = 2

    mock_ev = MockChannel()
    st_ev = typing.cast("EagerCallableProtocol", WebRTCDataChannelStream)(mock_ev, high_watermark=10, low_watermark=5)
    mock_ev.bufferedAmount = 20

    async def event_later() -> None:
        """Test function event_later."""
        await asyncio.sleep(0.005)
        st_ev._on_buffered_amount_low(None)

    asyncio.create_task(event_later())
    await st_ev.wait_backpressure()

    getattr(_state, "update")({"NoRecvChannel": NoRecvChannel, "drain_later": drain_later, "mock_to": mock_to, "put_msg": put_msg})
    getattr(_state, "update")({"res_raw": res_raw, "st_empty": st_empty, "st_to": st_to})
    _state.update(locals())


async def _test_webrtc_collectives_error_handling_part_3(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 3 of test_webrtc_collectives_error_handling.

    Args:
        _state: Shared test state dictionary.
    """
    drain_later, st_to = _state["drain_later"], _state["st_to"]
    typing.cast("EagerCallableProtocol", asyncio.create_task)(typing.cast("EagerCallableProtocol", drain_later)())
    await st_to.wait_backpressure()

    # Fallthrough for unknown op in pytorch / jax
    t = np.array([1.0, 2.0])
    all_data = [t, t]
    with patch.dict(sys.modules, {"ml_switcheroo_compiler.backends.pytorch.distributed_collectives": MagicMock()}):
        res_pt_un = dispatch_collective("UnknownOp", t, backend="pytorch", all_ranks_data=all_data)
        typing.cast("EagerCallableProtocol", np.testing.assert_array_equal)(res_pt_un, t)

    with patch.dict(sys.modules, {"ml_switcheroo_compiler.backends.jax.distributed_collectives": MagicMock()}):
        res_jax_un = dispatch_collective("UnknownOp", t, backend="jax", all_ranks_data=all_data)
        typing.cast("EagerCallableProtocol", np.testing.assert_array_equal)(res_jax_un, t)

    # Fallthrough when all_ranks_data is None for Broadcast
    res_bcast_none = dispatch_collective("Broadcast", t, backend="numpy")
    typing.cast("EagerCallableProtocol", np.testing.assert_array_equal)(res_bcast_none, t)

    # Unknown op_type in AllReduce and ReduceScatter with all_ranks_data
    assert typing.cast("EagerCallableProtocol", np.allclose)(typing.cast("EagerCallableProtocol", dispatch_collective)("AllReduce", t, backend="numpy", op_type="UNKNOWN", all_ranks_data=all_data), t)
    assert typing.cast("EagerCallableProtocol", np.allclose)(typing.cast("EagerCallableProtocol", dispatch_collective)("ReduceScatter", t, backend="numpy", op_type="UNKNOWN", scatter_dim=0, rank=0, all_ranks_data=all_data), typing.cast("EagerCallableProtocol", np.array_split)(t, 2, axis=0)[0])
    getattr(_state, "update")({"all_data": all_data, "res_bcast_none": res_bcast_none, "res_jax_un": res_jax_un, "res_pt_un": res_pt_un})
    getattr(_state, "update")({"t": t})


@pytest.mark.asyncio
async def test_webrtc_collectives_error_handling() -> None:
    """Test fallback branches including None partners, endianness conversion, and dispatch fallthroughs."""
    _state: dict[str, EagerTensorProtocol] = {}
    await _test_webrtc_collectives_error_handling_part_1(_state)
    await _test_webrtc_collectives_error_handling_part_2(_state)
    await _test_webrtc_collectives_error_handling_part_3(_state)
    assert len(_state) > 0
