"""Unit tests for WebRTC distributed collective operations."""

from __future__ import annotations

import asyncio
import sys
import typing

import numpy as np
import pytest

from ml_switcheroo_compiler.distributed.collectives import (
    DistributedBarrier,
    WebRTCDataChannelStream,
    _recv_data,
    _send_data,
    all_gather,
    all_reduce,
    all_to_all,
    broadcast,
    reduce_scatter,
    ring_reduce_scatter,
)

if typing.TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import EagerCallableProtocol, EagerTensorProtocol, EagerValue


class MockDataChannel:
    """Mock RTCDataChannel simulating binary transmission and event triggers."""

    sent: list[bytes]
    recv_queue: list[bytes]
    buffered_amount: int
    buffered_amount_low_threshold: int
    onmessage: EagerValue | None
    onbufferedamountlow: EagerValue | None

    def __init__(self) -> None:
        """Initialize MockDataChannel."""
        self.sent = []
        self.recv_queue = []
        self.buffered_amount = 0
        self.buffered_amount_low_threshold = 16384
        self.onmessage = None
        self.onbufferedamountlow = None

    async def send(self, data: bytes) -> None:
        """Record sent binary data.

        Args:
            data: Test parameter `data`.
        """
        self.sent.append(data)

    async def recv(self) -> bytes:
        """Pop queued binary data or return empty bytes."""
        if not self.recv_queue:
            return b""
        return self.recv_queue.pop(0)

    def trigger_message(self, data: bytes) -> None:
        """Simulate an incoming onmessage event.

        Args:
            data: Test parameter `data`.
        """
        if self.onmessage is not None and callable(self.onmessage):
            self.onmessage(data)

    def trigger_low_buffer(self) -> None:
        """Simulate a bufferedamountlow event."""
        self.buffered_amount = 0
        if self.onbufferedamountlow is not None and callable(self.onbufferedamountlow):
            self.onbufferedamountlow(None)


@pytest.mark.asyncio
async def test_collectives_send_recv_basic() -> None:
    """Test basic tensor send and recv."""
    data = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    conn = MockDataChannel()

    await typing.cast("EagerCallableProtocol", _send_data)(conn, data)
    assert len(conn.sent) >= 2  # Meta header + chunk

    conn.recv_queue = list(conn.sent)
    received = await typing.cast("EagerCallableProtocol", _recv_data)(conn)
    assert typing.cast("EagerCallableProtocol", np.array_equal)(data, received)


@pytest.mark.asyncio
async def test_collectives_chunked_large_tensor() -> None:
    """Test chunked binary transfer for tensors exceeding default chunk size."""
    # 20,000 float32s = 80,000 bytes > 16,384 byte chunk size (5 chunks + 1 header = 6 messages)
    data = np.arange(20000, dtype=np.float32)
    conn = MockDataChannel()

    await typing.cast("EagerCallableProtocol", _send_data)(conn, data, chunk_size=16384)
    assert len(conn.sent) == 6  # 1 metadata + 5 chunks

    conn.recv_queue = list(conn.sent)
    received = await typing.cast("EagerCallableProtocol", _recv_data)(conn)
    assert typing.cast("EagerCallableProtocol", np.array_equal)(data, received)


@pytest.mark.asyncio
async def test_collectives_datatypes() -> None:
    """Test serialization and deserialization across varied numpy dtypes."""
    dtypes = [
        np.float16,
        np.float32,
        np.float64,
        np.int8,
        np.int16,
        np.int32,
        np.int64,
        np.uint8,
        np.uint16,
        np.uint32,
        np.uint64,
        bool,
        np.complex64,
        np.complex128,
    ]

    for dt in dtypes:
        conn = MockDataChannel()
        arr = np.array([0, 1, 2, 3], dtype=dt)
        await typing.cast("EagerCallableProtocol", _send_data)(conn, arr)
        conn.recv_queue = list(conn.sent)
        rec = await typing.cast("EagerCallableProtocol", _recv_data)(conn)
        assert typing.cast("EagerTensorProtocol", getattr(rec, "dtype")) == typing.cast("EagerTensorProtocol", getattr(np, "dtype"))(dt)
        assert typing.cast("EagerCallableProtocol", np.array_equal)(arr, rec)


@pytest.mark.asyncio
async def test_collectives_byte_order_handling() -> None:
    """Test serialization and deserialization with explicit endianness."""
    opposite_order = ">" if sys.byteorder == "little" else "<"
    data = np.array([1.5, 2.5, 3.5], dtype=f"{opposite_order}f4")

    conn = MockDataChannel()
    await typing.cast("EagerCallableProtocol", _send_data)(conn, data)

    conn.recv_queue = list(conn.sent)
    received = await typing.cast("EagerCallableProtocol", _recv_data)(conn)
    assert typing.cast("EagerCallableProtocol", np.allclose)(data, received)


@pytest.mark.asyncio
async def test_collectives_backpressure() -> None:
    """Test backpressure flow control pauses sending until drain."""
    conn = MockDataChannel()
    stream = typing.cast("EagerCallableProtocol", WebRTCDataChannelStream)(conn, high_watermark=100, low_watermark=50)

    conn.buffered_amount = 200

    drain_task = typing.cast("EagerCallableProtocol", asyncio.create_task)(typing.cast("EagerCallableProtocol", stream.wait_backpressure)())
    await asyncio.sleep(0.01)
    assert not drain_task.done()

    conn.trigger_low_buffer()
    await drain_task
    assert drain_task.done()


@pytest.mark.asyncio
async def test_collectives_event_listener_stream() -> None:
    """Test WebRTCDataChannelStream event-driven message dispatch."""
    conn = MockDataChannel()
    stream = typing.cast("EagerCallableProtocol", WebRTCDataChannelStream)(conn)

    conn.trigger_message(b"test_payload")
    rec = await stream.recv_bytes()
    assert rec == b"test_payload"


@pytest.mark.asyncio
async def test_collectives_broadcast() -> None:
    """Test Broadcast collective operation."""
    data = np.array([1.0, 2.0], dtype=np.float32)
    conn0, conn1 = MockDataChannel(), MockDataChannel()

    # Leader broadcast
    res = await typing.cast("EagerCallableProtocol", broadcast)(data, 0, [conn0, conn1], 0)
    assert typing.cast("EagerCallableProtocol", np.array_equal)(res, data)
    assert len(conn1.sent) >= 2

    # Follower receive from root (rank 0)
    conn0.recv_queue = list(conn1.sent)
    res2 = await typing.cast("EagerCallableProtocol", broadcast)(typing.cast("EagerCallableProtocol", np.zeros)(2, dtype=np.float32), 0, [conn0, conn1], 1)
    assert typing.cast("EagerCallableProtocol", np.array_equal)(res2, data)

    with pytest.raises(ValueError):
        await typing.cast("EagerCallableProtocol", broadcast)(data, 0, [None, conn1], 1)


@pytest.mark.asyncio
async def test_collectives_all_reduce() -> None:
    """Test AllReduce with SUM, PROD, MAX, MIN."""
    data0 = np.array([1.0, 2.0], dtype=np.float32)
    data1 = np.array([3.0, 4.0], dtype=np.float32)

    # SUM
    c1_to_0 = MockDataChannel()
    await typing.cast("EagerCallableProtocol", _send_data)(c1_to_0, data1)
    c1_to_0.recv_queue = list(c1_to_0.sent)

    res_sum = await typing.cast("EagerCallableProtocol", all_reduce)(data0, "SUM", [None, c1_to_0], 0)
    assert typing.cast("EagerCallableProtocol", np.array_equal)(res_sum, typing.cast("EagerCallableProtocol", np.array)([4.0, 6.0], dtype=np.float32))

    # PROD
    c1_to_0.sent.clear()
    await typing.cast("EagerCallableProtocol", _send_data)(c1_to_0, data1)
    c1_to_0.recv_queue = list(c1_to_0.sent)
    res_prod = await typing.cast("EagerCallableProtocol", all_reduce)(data0, "PROD", [None, c1_to_0], 0)
    assert typing.cast("EagerCallableProtocol", np.array_equal)(res_prod, typing.cast("EagerCallableProtocol", np.array)([3.0, 8.0], dtype=np.float32))

    # MAX
    c1_to_0.sent.clear()
    await typing.cast("EagerCallableProtocol", _send_data)(c1_to_0, data1)
    c1_to_0.recv_queue = list(c1_to_0.sent)
    res_max = await typing.cast("EagerCallableProtocol", all_reduce)(data0, "MAX", [None, c1_to_0], 0)
    assert typing.cast("EagerCallableProtocol", np.array_equal)(res_max, typing.cast("EagerCallableProtocol", np.array)([3.0, 4.0], dtype=np.float32))

    # MIN
    c1_to_0.sent.clear()
    await typing.cast("EagerCallableProtocol", _send_data)(c1_to_0, data1)
    c1_to_0.recv_queue = list(c1_to_0.sent)
    res_min = await typing.cast("EagerCallableProtocol", all_reduce)(data0, "MIN", [None, c1_to_0], 0)
    assert typing.cast("EagerCallableProtocol", np.array_equal)(res_min, typing.cast("EagerCallableProtocol", np.array)([1.0, 2.0], dtype=np.float32))

    c1_to_0.sent.clear()
    await typing.cast("EagerCallableProtocol", _send_data)(c1_to_0, data1)
    c1_to_0.recv_queue = list(c1_to_0.sent)
    with pytest.raises(ValueError, match="Unsupported reduction operator"):
        await typing.cast("EagerCallableProtocol", all_reduce)(data0, "UNKNOWN_OP", [None, c1_to_0], 0)


@pytest.mark.asyncio
async def test_collectives_all_gather() -> None:
    """Test AllGather collective operation."""
    data0 = np.array([1.0, 2.0], dtype=np.float32)
    data1 = np.array([3.0, 4.0], dtype=np.float32)

    c1 = MockDataChannel()
    await typing.cast("EagerCallableProtocol", _send_data)(c1, data1)
    c1.recv_queue = list(c1.sent)

    gathered = await typing.cast("EagerCallableProtocol", all_gather)(data0, 0, [None, c1], 0)
    assert typing.cast("EagerCallableProtocol", np.array_equal)(gathered, typing.cast("EagerCallableProtocol", np.array)([1.0, 2.0, 3.0, 4.0], dtype=np.float32))


@pytest.mark.asyncio
async def test_collectives_reduce_scatter() -> None:
    """Test ReduceScatter collective operation."""
    data0 = np.array([1.0, 2.0], dtype=np.float32)
    data1 = np.array([3.0, 4.0], dtype=np.float32)

    c1 = MockDataChannel()
    await typing.cast("EagerCallableProtocol", _send_data)(c1, data1)
    c1.recv_queue = list(c1.sent)

    res0 = await typing.cast("EagerCallableProtocol", reduce_scatter)(data0, "SUM", 0, [None, c1], 0)
    assert typing.cast("EagerCallableProtocol", np.array_equal)(res0, typing.cast("EagerCallableProtocol", np.array)([4.0], dtype=np.float32))


@pytest.mark.asyncio
async def test_collectives_all_to_all() -> None:
    """Test AllToAll collective operation across peers."""
    data0 = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)

    c1_on_0 = MockDataChannel()

    chunk_from_1 = np.array([[5.0, 6.0]], dtype=np.float32)
    await typing.cast("EagerCallableProtocol", _send_data)(c1_on_0, chunk_from_1)
    c1_on_0.recv_queue = list(c1_on_0.sent)

    res = await typing.cast("EagerCallableProtocol", all_to_all)(data0, scatter_dim=0, gather_dim=0, conns=[None, c1_on_0], rank=0)
    assert typing.cast("EagerCallableProtocol", np.array_equal)(res, typing.cast("EagerCallableProtocol", np.array)([[1.0, 2.0], [5.0, 6.0]], dtype=np.float32))


@pytest.mark.asyncio
async def test_collectives_barrier() -> None:
    """Test DistributedBarrier leader-follower coordination."""
    b_leader = DistributedBarrier(world_size=2, rank=0, leader_rank=0)
    b_follower = DistributedBarrier(world_size=2, rank=1, leader_rank=0)

    conn0 = MockDataChannel()
    conn1 = MockDataChannel()

    # Follower signals ready
    conn0.recv_queue = [b"g"]
    await typing.cast("EagerCallableProtocol", b_follower.wait)([conn0, None])
    assert conn0.sent[0] == b"r"

    # Leader receives ready, signals go
    conn1.recv_queue = [b"r"]
    await typing.cast("EagerCallableProtocol", b_leader.wait)([None, conn1])
    assert conn1.sent[0] == b"g"

    # Missing connection graceful pass
    await b_leader.wait([None, None])
    await b_follower.wait([None, None])


@pytest.mark.asyncio
async def test_stream_event_handlers_snake_case_and_payload_types() -> None:
    """Test snake_case attribute event handler binding and diverse message payload types."""

    class SnakeCaseChannel:
        """Test class SnakeCaseChannel."""

        def __init__(self) -> None:
            """Test function __init__."""
            self.on_message = None
            self.buffered_amount_low_threshold = 0
            self.on_buffered_amount_low = None
            self.buffered_amount = lambda: 50

    ch = SnakeCaseChannel()
    stream = typing.cast("EagerCallableProtocol", WebRTCDataChannelStream)(ch, low_watermark=100)
    assert ch.buffered_amount_low_threshold == 100
    assert ch.on_message == stream._on_message
    assert ch.on_buffered_amount_low == stream._on_buffered_amount_low
    assert stream.get_buffered_amount() == 50

    # Test payload types in _on_message:
    # 1. str
    typing.cast("EagerCallableProtocol", ch.on_message)("str_message")
    assert await stream.recv_bytes() == b"str_message"

    # 2. tobytes
    mem = memoryview(b"memoryview_bytes")
    typing.cast("EagerCallableProtocol", ch.on_message)(mem)
    assert await stream.recv_bytes() == b"memoryview_bytes"

    # 3. other iterable/convertible to bytes
    typing.cast("EagerCallableProtocol", ch.on_message)([65, 66, 67])
    assert await stream.recv_bytes() == b"ABC"

    # Channel with no recognized attributes
    bare_stream = typing.cast("EagerCallableProtocol", WebRTCDataChannelStream)(typing.cast("EagerCallableProtocol", object)())
    assert bare_stream.get_buffered_amount() == 0


@pytest.mark.asyncio
async def test_stream_backpressure_branches() -> None:
    """Test wait_backpressure with sync and async wait_drain, and polling timeout."""

    # 1. Async wait_drain
    class AsyncDrainChannel:
        """Test class AsyncDrainChannel."""

        def __init__(self) -> None:
            """Test function __init__."""
            self.bufferedAmount = 200
            self.drained = False

        async def wait_drain(self) -> None:
            """Test function wait_drain."""
            self.drained = True
            self.bufferedAmount = 0

    ch_async = AsyncDrainChannel()
    s1 = typing.cast("EagerCallableProtocol", WebRTCDataChannelStream)(ch_async, high_watermark=100, low_watermark=50)
    await s1.wait_backpressure()
    assert ch_async.drained

    # 2. Sync wait_drain
    class SyncDrainChannel:
        """Test class SyncDrainChannel."""

        def __init__(self) -> None:
            """Test function __init__."""
            self.bufferedAmount = 200
            self.drained = False

        def wait_drain(self) -> None:
            """Test function wait_drain."""
            self.drained = True
            self.bufferedAmount = 0

    ch_sync = SyncDrainChannel()
    s2 = typing.cast("EagerCallableProtocol", WebRTCDataChannelStream)(ch_sync, high_watermark=100, low_watermark=50)
    await s2.wait_backpressure()
    assert ch_sync.drained

    # 3. Timeout in loop before draining
    class LoopDrainChannel:
        """Test class LoopDrainChannel."""

        def __init__(self) -> None:
            """Test function __init__."""
            self.checks = 0

        def bufferedAmount(self) -> EagerValue:
            """Test function bufferedAmount."""
            self.checks += 1
            if self.checks > 2:
                return 0
            return 200

    ch_loop = LoopDrainChannel()
    s3 = typing.cast("EagerCallableProtocol", WebRTCDataChannelStream)(ch_loop, high_watermark=100, low_watermark=50)
    await s3.wait_backpressure()
    assert ch_loop.checks > 2


async def _test_stream_send_recv_message_variants_part_1(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 1 of test_stream_send_recv_message_variants.

    Args:
        _state: Shared test state dictionary.
    """

    class CoroSendChannel:
        """Test class CoroSendChannel."""

        def __init__(self) -> None:
            """Test function __init__."""
            self.sent = typing.cast("list[EagerTensorProtocol]", [])

        def send(self, data: EagerValue) -> EagerValue:
            """Test function send.

            Args:
                data: Test parameter `data`.
            """

            async def _inner() -> None:
                """Test function _inner."""
                typing.cast("EagerCallableProtocol", self.sent.append)(data)

            return typing.cast("EagerTensorProtocol", _inner())

    ch_coro = CoroSendChannel()
    s1 = typing.cast("EagerCallableProtocol", WebRTCDataChannelStream)(ch_coro)
    await s1.send_bytes(b"coro_send")
    assert ch_coro.sent == [b"coro_send"]
    getattr(_state, "update")({"CoroSendChannel": CoroSendChannel, "ch_coro": ch_coro, "s1": s1})
    _state.update(locals())


async def _test_stream_send_recv_message_variants_part_2(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 2 of test_stream_send_recv_message_variants.

    Args:
        _state: Shared test state dictionary.
    """

    class AsyncSendMessageChannel:
        """Test class AsyncSendMessageChannel."""

        def __init__(self) -> None:
            """Test function __init__."""
            self.sent = typing.cast("list[EagerTensorProtocol]", [])

        async def sendMessage(self, data: EagerValue) -> None:
            """Test function sendMessage.

            Args:
                data: Test parameter `data`.
            """
            typing.cast("EagerCallableProtocol", self.sent.append)(data)

    ch_msg_async = AsyncSendMessageChannel()
    s2 = typing.cast("EagerCallableProtocol", WebRTCDataChannelStream)(ch_msg_async)
    await s2.send_bytes(b"msg_async")
    assert ch_msg_async.sent == [b"msg_async"]

    # 3. sendMessage sync returning coroutine and sync returning non-coroutine
    class SyncSendMessageChannel:
        """Test class SyncSendMessageChannel."""

        def __init__(self) -> None:
            """Test function __init__."""
            self.sent = typing.cast("list[EagerTensorProtocol]", [])

        def sendMessage(self, data: EagerValue) -> None:
            """Test function sendMessage.

            Args:
                data: Test parameter `data`.
            """
            typing.cast("EagerCallableProtocol", self.sent.append)(data)

    ch_msg_sync = SyncSendMessageChannel()
    s3 = typing.cast("EagerCallableProtocol", WebRTCDataChannelStream)(ch_msg_sync)
    getattr(_state, "update")({"AsyncSendMessageChannel": AsyncSendMessageChannel, "SyncSendMessageChannel": SyncSendMessageChannel, "ch_msg_async": ch_msg_async, "ch_msg_sync": ch_msg_sync})
    getattr(_state, "update")({"s2": s2, "s3": s3})
    _state.update(locals())


async def _test_stream_send_recv_message_variants_part_3(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 3 of test_stream_send_recv_message_variants.

    Args:
        _state: Shared test state dictionary.
    """
    ch_msg_sync, s3 = _state["ch_msg_sync"], _state["s3"]
    await s3.send_bytes(b"msg_sync")
    assert ch_msg_sync.sent == [b"msg_sync"]

    class CoroMsgChannel:
        """Test class CoroMsgChannel."""

        def __init__(self) -> None:
            """Test function __init__."""
            self.sent = typing.cast("list[EagerTensorProtocol]", [])

        def sendMessage(self, data: EagerValue) -> EagerValue:
            """Test function sendMessage.

            Args:
                data: Test parameter `data`.
            """

            async def _inner() -> None:
                """Test function _inner."""
                typing.cast("EagerCallableProtocol", self.sent.append)(data)

            return typing.cast("EagerTensorProtocol", _inner())

    ch_msg_coro = CoroMsgChannel()
    s4 = typing.cast("EagerCallableProtocol", WebRTCDataChannelStream)(ch_msg_coro)
    await s4.send_bytes(b"msg_coro")
    assert ch_msg_coro.sent == [b"msg_coro"]
    getattr(_state, "update")({"CoroMsgChannel": CoroMsgChannel, "ch_msg_coro": ch_msg_coro, "s4": s4})
    _state.update(locals())


async def _test_stream_send_recv_message_variants_part_4(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 4 of test_stream_send_recv_message_variants.

    Args:
        _state: Shared test state dictionary.
    """

    class RecvChannel:
        """Test class RecvChannel."""

        def __init__(self, mode: EagerValue) -> None:
            """Test function __init__.

            Args:
                mode: Test parameter `mode`.
            """
            self.mode = mode

        def recv(self) -> EagerValue:
            """Test function recv."""
            if self.mode == "sync_coro":

                async def _inner() -> EagerValue:
                    """Test function _inner."""
                    return bytearray(b"bytearray_data")

                return typing.cast("EagerTensorProtocol", _inner())
            return b"sync_bytes"

    s_sync = typing.cast("EagerCallableProtocol", WebRTCDataChannelStream)(typing.cast("EagerCallableProtocol", RecvChannel)("sync"))
    assert await s_sync.recv_bytes() == b"sync_bytes"

    s_coro = typing.cast("EagerCallableProtocol", WebRTCDataChannelStream)(typing.cast("EagerCallableProtocol", RecvChannel)("sync_coro"))
    assert await s_coro.recv_bytes() == b"bytearray_data"
    getattr(_state, "update")({"RecvChannel": RecvChannel, "s_coro": s_coro, "s_sync": s_sync})


@pytest.mark.asyncio
async def test_stream_send_recv_message_variants() -> None:
    """Test send and recv variants: sendMessage, sync/async coroutines, non-bytes recv."""
    _state: dict[str, EagerTensorProtocol] = {}
    await _test_stream_send_recv_message_variants_part_1(_state)
    await _test_stream_send_recv_message_variants_part_2(_state)
    await _test_stream_send_recv_message_variants_part_3(_state)
    await _test_stream_send_recv_message_variants_part_4(_state)
    assert len(_state) > 0


@pytest.mark.asyncio
async def test_recv_data_mismatch_and_invalid_header() -> None:
    """Test payload length mismatch and invalid non-META header rejection in _recv_data."""
    # 1. Payload length mismatch
    ch_mismatch = MockDataChannel()
    import json

    meta = {
        "protocol": "webrtc_stream_v1",
        "shape": [2],
        "dtype": "float32",
        "total_bytes": 100,  # Expect 100 bytes
        "num_chunks": 1,
    }
    header = b"META:" + json.dumps(meta).encode("utf-8")
    ch_mismatch.recv_queue = [header, b"too_short"]
    with pytest.raises(ValueError, match="Payload length mismatch"):
        await typing.cast("EagerCallableProtocol", _recv_data)(ch_mismatch)

    # 2. Non-META header rejected
    ch_invalid = MockDataChannel()
    ch_invalid.recv_queue = [b"2,2|float32"]
    with pytest.raises(ValueError, match="expected META: prefix"):
        await typing.cast("EagerCallableProtocol", _recv_data)(ch_invalid)


@pytest.mark.asyncio
async def test_stream_remaining_missing_branches() -> None:
    """Test lines 58, 118, 134->exit, 136->exit, 162."""

    # Line 58: CamelCase bufferedAmountLowThreshold
    class CamelCaseChannel:
        """Test class CamelCaseChannel."""

        def __init__(self) -> None:
            """Test function __init__."""
            self.bufferedAmountLowThreshold = 0

    camel_ch = CamelCaseChannel()
    typing.cast("EagerCallableProtocol", WebRTCDataChannelStream)(camel_ch, low_watermark=2048)
    assert camel_ch.bufferedAmountLowThreshold == 2048

    # Line 118: wait_backpressure break
    class DrainEventChannel:
        """Test class DrainEventChannel."""

        def __init__(self) -> None:
            """Test function __init__."""
            self.bufferedAmount = 200

    drain_ch = DrainEventChannel()
    stream_drain = typing.cast("EagerCallableProtocol", WebRTCDataChannelStream)(drain_ch, high_watermark=100, low_watermark=50)

    async def _trigger_drain() -> None:
        """Test function _trigger_drain."""
        await asyncio.sleep(0.001)
        stream_drain._drain_event.set()

    asyncio.create_task(_trigger_drain())
    await stream_drain.wait_backpressure()

    # Line 134->exit: sync send returning non-coroutine
    class SyncSendNonCoroChannel:
        """Test class SyncSendNonCoroChannel."""

        def __init__(self) -> None:
            """Test function __init__."""
            self.sent = typing.cast("list[EagerTensorProtocol]", [])

        def send(self, data: EagerValue) -> None:
            """Test function send.

            Args:
                data: Test parameter `data`.
            """
            typing.cast("EagerCallableProtocol", self.sent.append)(data)
            return None

    non_coro_ch = SyncSendNonCoroChannel()
    s_non_coro = typing.cast("EagerCallableProtocol", WebRTCDataChannelStream)(non_coro_ch)
    await s_non_coro.send_bytes(b"non_coro")
    assert non_coro_ch.sent == [b"non_coro"]

    # Line 136->exit: channel with neither send nor sendMessage
    class NoSendMethodChannel:
        """Test class NoSendMethodChannel."""

        pass

    s_no_send = typing.cast("EagerCallableProtocol", WebRTCDataChannelStream)(typing.cast("EagerCallableProtocol", NoSendMethodChannel)())
    await s_no_send.send_bytes(b"will_not_send")

    # Line 162: recv_bytes with empty queue and channel without recv
    class NoRecvMethodChannel:
        """Test class NoRecvMethodChannel."""

        pass

    s_no_recv = typing.cast("EagerCallableProtocol", WebRTCDataChannelStream)(typing.cast("EagerCallableProtocol", NoRecvMethodChannel)())

    async def _put_later() -> None:
        """Test function _put_later."""
        await asyncio.sleep(0.001)
        s_no_recv._recv_queue.put_nowait(b"delayed_data")

    asyncio.create_task(_put_later())
    res = await s_no_recv.recv_bytes()
    assert res == b"delayed_data"


def _test_dispatch_collective_exhaustive_part_1(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 1 of test_dispatch_collective_exhaustive.

    Args:
        _state: Shared test state dictionary.
    """
    from unittest.mock import patch

    from ml_switcheroo_compiler.distributed.collectives import dispatch_collective

    t = np.array([1.0, 2.0], dtype=np.float32)
    ranks_data = [
        np.array([1.0, 2.0], dtype=np.float32),
        np.array([3.0, 4.0], dtype=np.float32),
    ]

    # 1. Fallback when all_ranks_data is None
    res_fallback = dispatch_collective("UnknownOp", t)
    assert typing.cast("EagerCallableProtocol", np.array_equal)(res_fallback, t)

    # 2. AllReduce with different reduction ops
    res_sum = dispatch_collective("AllReduce", t, all_ranks_data=ranks_data, op_type="SUM")
    assert typing.cast("EagerCallableProtocol", np.array_equal)(res_sum, typing.cast("EagerCallableProtocol", np.array)([4.0, 6.0], dtype=np.float32))

    res_prod = dispatch_collective("AllReduce", t, all_ranks_data=ranks_data, op_type="PROD")
    assert typing.cast("EagerCallableProtocol", np.array_equal)(res_prod, typing.cast("EagerCallableProtocol", np.array)([3.0, 8.0], dtype=np.float32))

    res_max = dispatch_collective("AllReduce", t, all_ranks_data=ranks_data, op_type="MAX")
    assert typing.cast("EagerCallableProtocol", np.array_equal)(res_max, typing.cast("EagerCallableProtocol", np.array)([3.0, 4.0], dtype=np.float32))

    res_min = dispatch_collective("AllReduce", t, all_ranks_data=ranks_data, op_type="MIN")
    assert typing.cast("EagerCallableProtocol", np.array_equal)(res_min, typing.cast("EagerCallableProtocol", np.array)([1.0, 2.0], dtype=np.float32))

    res_other = dispatch_collective("AllReduce", t, all_ranks_data=ranks_data, op_type="UNKNOWN")
    assert typing.cast("EagerCallableProtocol", np.array_equal)(res_other, ranks_data[0])

    # 3. AllGather
    res_gather = dispatch_collective("AllGather", t, all_ranks_data=ranks_data, axis=0)
    assert typing.cast("EagerCallableProtocol", np.array_equal)(res_gather, typing.cast("EagerCallableProtocol", np.array)([1.0, 2.0, 3.0, 4.0], dtype=np.float32))

    # 4. ReduceScatter with various op_types
    ranks_data_3 = [
        np.array([1.0, 2.0, 3.0], dtype=np.float32),
        np.array([4.0, 5.0, 6.0], dtype=np.float32),
        np.array([7.0, 8.0, 9.0], dtype=np.float32),
    ]
    res_rs_multi = dispatch_collective("ReduceScatter", t, all_ranks_data=ranks_data_3, rank=0, op_type="MIN", scatter_dim=0)
    assert typing.cast("EagerCallableProtocol", np.array_equal)(res_rs_multi, typing.cast("EagerCallableProtocol", np.array)([1.0], dtype=np.float32))

    res_rs_sum = dispatch_collective("ReduceScatter", t, all_ranks_data=ranks_data, rank=0, op_type="SUM", scatter_dim=0)
    getattr(_state, "update")({"dispatch_collective": dispatch_collective, "patch": patch, "ranks_data": ranks_data, "ranks_data_3": ranks_data_3})
    getattr(_state, "update")({"res_fallback": res_fallback, "res_gather": res_gather, "res_max": res_max, "res_min": res_min})
    getattr(_state, "update")({"res_other": res_other, "res_prod": res_prod, "res_rs_multi": res_rs_multi, "res_rs_sum": res_rs_sum})
    getattr(_state, "update")({"res_sum": res_sum, "t": t})
    _state.update(locals())


def _test_dispatch_collective_exhaustive_part_2(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 2 of test_dispatch_collective_exhaustive.

    Args:
        _state: Shared test state dictionary.
    """
    dispatch_collective, patch, ranks_data, ranks_data_3 = _state["dispatch_collective"], _state["patch"], _state["ranks_data"], _state["ranks_data_3"]
    res_rs_sum, t = _state["res_rs_sum"], _state["t"]
    assert typing.cast("EagerCallableProtocol", np.array_equal)(res_rs_sum, typing.cast("EagerCallableProtocol", np.array)([4.0], dtype=np.float32))

    res_rs_prod = dispatch_collective("ReduceScatter", t, all_ranks_data=ranks_data, rank=1, op_type="PROD", scatter_dim=0)
    assert typing.cast("EagerCallableProtocol", np.array_equal)(res_rs_prod, typing.cast("EagerCallableProtocol", np.array)([8.0], dtype=np.float32))
    res_rs_max = dispatch_collective("ReduceScatter", t, all_ranks_data=ranks_data, rank=0, op_type="MAX", scatter_dim=0)
    assert typing.cast("EagerCallableProtocol", np.array_equal)(res_rs_max, typing.cast("EagerCallableProtocol", np.array)([3.0], dtype=np.float32))

    res_rs_min = dispatch_collective("ReduceScatter", t, all_ranks_data=ranks_data, rank=1, op_type="MIN", scatter_dim=0)
    assert typing.cast("EagerCallableProtocol", np.array_equal)(res_rs_min, typing.cast("EagerCallableProtocol", np.array)([2.0], dtype=np.float32))

    res_rs_unknown = dispatch_collective("ReduceScatter", t, all_ranks_data=ranks_data_3, rank=0, op_type="UNKNOWN", scatter_dim=0)
    assert typing.cast("EagerCallableProtocol", np.array_equal)(res_rs_unknown, typing.cast("EagerCallableProtocol", np.array)([1.0], dtype=np.float32))

    # 5. AllToAll
    data_2x2_0 = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    data_2x2_1 = np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32)
    res_a2a = dispatch_collective("AllToAll", data_2x2_0, all_ranks_data=[data_2x2_0, data_2x2_1], rank=0, scatter_dim=0, gather_dim=0)
    assert res_a2a.shape == (2, 2)

    # 6. Broadcast and unhandled op with all_ranks_data
    res_bcast = dispatch_collective("Broadcast", t, all_ranks_data=ranks_data, root=1)
    assert typing.cast("EagerCallableProtocol", np.array_equal)(res_bcast, ranks_data[1])

    res_unhandled_with_ranks = dispatch_collective("UnhandledOp", t, all_ranks_data=ranks_data)
    assert typing.cast("EagerCallableProtocol", np.array_equal)(res_unhandled_with_ranks, t)

    # 7. PyTorch backend branch and exception handling
    dispatch_collective("AllToAll", t, backend="pytorch")
    with patch("ml_switcheroo_compiler.backends.pytorch.distributed_collectives.pytorch_all_reduce", return_value=t) as mock_pt_ar:
        dispatch_collective("AllReduce", t, backend="pytorch")
        assert mock_pt_ar.called
    getattr(_state, "update")({"data_2x2_0": data_2x2_0, "data_2x2_1": data_2x2_1, "mock_pt_ar": mock_pt_ar, "res_a2a": res_a2a})
    getattr(_state, "update")({"res_bcast": res_bcast, "res_rs_max": res_rs_max, "res_rs_min": res_rs_min, "res_rs_prod": res_rs_prod})
    getattr(_state, "update")({"res_rs_unknown": res_rs_unknown, "res_unhandled_with_ranks": res_unhandled_with_ranks})
    _state.update(locals())


def _test_dispatch_collective_exhaustive_part_3(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 3 of test_dispatch_collective_exhaustive.

    Args:
        _state: Shared test state dictionary.
    """
    dispatch_collective, patch, t = _state["dispatch_collective"], _state["patch"], _state["t"]
    with patch("ml_switcheroo_compiler.backends.pytorch.distributed_collectives.pytorch_all_gather", return_value=t) as mock_pt_ag:
        dispatch_collective("AllGather", t, backend="pytorch")
        assert mock_pt_ag.called
    with patch("ml_switcheroo_compiler.backends.pytorch.distributed_collectives.pytorch_reduce_scatter", return_value=t) as mock_pt_rs:
        dispatch_collective("ReduceScatter", t, backend="pytorch")
        assert mock_pt_rs.called
    with patch("ml_switcheroo_compiler.backends.pytorch.distributed_collectives.pytorch_broadcast", return_value=t) as mock_pt_bc:
        dispatch_collective("Broadcast", t, backend="pytorch")
        assert mock_pt_bc.called
    with typing.cast("EagerCallableProtocol", patch)("ml_switcheroo_compiler.backends.pytorch.distributed_collectives.pytorch_all_reduce", side_effect=typing.cast("EagerCallableProtocol", RuntimeError)("pt_err")):
        res_pt_err = dispatch_collective("AllReduce", t, backend="pytorch")
        assert typing.cast("EagerCallableProtocol", np.array_equal)(res_pt_err, t)

    # 8. JAX backend branch and exception handling
    dispatch_collective("AllToAll", t, backend="jax")
    with patch("ml_switcheroo_compiler.backends.jax.distributed_collectives.jax_all_reduce", return_value=t) as mock_jax_ar:
        dispatch_collective("AllReduce", t, backend="jax")
        assert mock_jax_ar.called
    getattr(_state, "update")({"mock_jax_ar": mock_jax_ar, "mock_pt_ag": mock_pt_ag, "mock_pt_bc": mock_pt_bc, "mock_pt_rs": mock_pt_rs})
    getattr(_state, "update")({"res_pt_err": res_pt_err})
    _state.update(locals())


def _test_dispatch_collective_exhaustive_part_4(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 4 of test_dispatch_collective_exhaustive.

    Args:
        _state: Shared test state dictionary.
    """
    dispatch_collective, patch, t = _state["dispatch_collective"], _state["patch"], _state["t"]
    with patch("ml_switcheroo_compiler.backends.jax.distributed_collectives.jax_all_gather", return_value=t) as mock_jax_ag:
        dispatch_collective("AllGather", t, backend="jax")
        assert mock_jax_ag.called
    with patch("ml_switcheroo_compiler.backends.jax.distributed_collectives.jax_broadcast", return_value=t) as mock_jax_bc:
        dispatch_collective("Broadcast", t, backend="jax")
        assert mock_jax_bc.called
    with typing.cast("EagerCallableProtocol", patch)("ml_switcheroo_compiler.backends.jax.distributed_collectives.jax_all_reduce", side_effect=typing.cast("EagerCallableProtocol", RuntimeError)("jax_err")):
        res_jax_err = dispatch_collective("AllReduce", t, backend="jax")
        assert typing.cast("EagerCallableProtocol", np.array_equal)(res_jax_err, t)
    getattr(_state, "update")({"mock_jax_ag": mock_jax_ag, "mock_jax_bc": mock_jax_bc, "res_jax_err": res_jax_err})


def test_dispatch_collective_exhaustive() -> None:
    """Verify all branches, op types, and backends in dispatch_collective."""
    _state: dict[str, EagerTensorProtocol] = {}
    _test_dispatch_collective_exhaustive_part_1(_state)
    _test_dispatch_collective_exhaustive_part_2(_state)
    _test_dispatch_collective_exhaustive_part_3(_state)
    _test_dispatch_collective_exhaustive_part_4(_state)
    assert len(_state) > 0


class SimulatedPeerChannel:
    """Channel connecting peer A to peer B using asyncio queues."""

    def __init__(self, in_q: asyncio.Queue[bytes], out_q: asyncio.Queue[bytes]) -> None:
        """Initialize channel with input and output queues.

        Args:
            in_q: Test parameter `in_q`.
            out_q: Test parameter `out_q`.
        """
        self.in_q = in_q
        self.out_q = out_q
        self.buffered_amount = 0

    async def send(self, data: bytes) -> None:
        """Send byte message to recipient queue.

        Args:
            data: Test parameter `data`.
        """
        await self.out_q.put(data)

    async def recv(self) -> bytes:
        """Receive byte message from incoming queue."""
        return await self.in_q.get()


def _create_mesh_channels(n: int) -> list[list[SimulatedPeerChannel | None]]:
    """Create a fully connected mesh of simulated peer channels.

    Args:
        n: Test parameter `n`.
    """
    # queues[src][dst]
    queues = typing.cast("EagerTensorProtocol", [[asyncio.Queue() for _ in range(n)] for _ in range(n)])
    mesh: list[list[SimulatedPeerChannel | None]] = []
    for r in range(n):
        row: list[SimulatedPeerChannel | None] = []
        for peer in range(n):
            if r == peer:
                row.append(None)
            else:
                typing.cast("EagerCallableProtocol", row.append)(typing.cast("EagerCallableProtocol", SimulatedPeerChannel)(in_q=queues[peer][r], out_q=queues[r][peer]))
        mesh.append(row)
    return mesh


@pytest.mark.asyncio
async def test_multi_peer_ring_all_reduce() -> None:
    """Test multi-peer Ring AllReduce float array synchronization across 4 concurrent peers."""
    from ml_switcheroo_compiler.distributed.collectives import ring_all_reduce

    world_size = 4
    mesh = _create_mesh_channels(world_size)

    # SUM test: each rank has 4 elements: [1, 2, 3, 4] * (rank + 1)
    inputs = [np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32) * (r + 1) for r in range(world_size)]
    # Sum: 1* + 2* + 3* + 4* = 10 * [1, 2, 3, 4] = [10, 20, 30, 40]
    expected_sum = np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float32)

    async def run_rank_sum(rank: int) -> np.ndarray[tuple[int, ...], np.dtype[np.generic]]:
        """Test function run_rank_sum.

        Args:
            rank: Test parameter `rank`.
        """
        return typing.cast("np.ndarray[tuple[int, ...], np.dtype[np.generic]]", await typing.cast("EagerCallableProtocol", ring_all_reduce)(inputs[rank], "SUM", mesh[rank], rank))

    results = await asyncio.gather(*[run_rank_sum(r) for r in range(world_size)])
    for res in results:
        assert np.allclose(res, expected_sum)

    # PROD, MAX, MIN tests
    mesh_prod = _create_mesh_channels(world_size)
    inputs_prod = [np.array([2.0, 3.0, 4.0, 5.0], dtype=np.float32) for _ in range(world_size)]

    async def run_rank_prod(rank: int) -> np.ndarray[tuple[int, ...], np.dtype[np.generic]]:
        """Test function run_rank_prod.

        Args:
            rank: Test parameter `rank`.
        """
        return typing.cast("np.ndarray[tuple[int, ...], np.dtype[np.generic]]", await typing.cast("EagerCallableProtocol", all_reduce)(inputs_prod[rank], "PROD", mesh_prod[rank], rank, algorithm="ring"))

    results_prod = await asyncio.gather(*[run_rank_prod(r) for r in range(world_size)])
    expected_prod = np.array([2.0**4, 3.0**4, 4.0**4, 5.0**4], dtype=np.float32)
    for res in results_prod:
        assert np.allclose(res, expected_prod)

    # MAX
    mesh_max = _create_mesh_channels(world_size)
    inputs_max = [np.array([float(r), float(r * 2), float(r * 3), float(r * 4)], dtype=np.float32) for r in range(world_size)]

    async def run_rank_max(rank: int) -> np.ndarray[tuple[int, ...], np.dtype[np.generic]]:
        """Test function run_rank_max.

        Args:
            rank: Test parameter `rank`.
        """
        return typing.cast("np.ndarray[tuple[int, ...], np.dtype[np.generic]]", await typing.cast("EagerCallableProtocol", ring_all_reduce)(inputs_max[rank], "MAX", mesh_max[rank], rank))

    results_max = await asyncio.gather(*[run_rank_max(r) for r in range(world_size)])
    expected_max = np.array([3.0, 6.0, 9.0, 12.0], dtype=np.float32)
    for res in results_max:
        assert np.allclose(res, expected_max)

    # MIN
    mesh_min = _create_mesh_channels(world_size)

    async def run_rank_min(rank: int) -> np.ndarray[tuple[int, ...], np.dtype[np.generic]]:
        """Test function run_rank_min.

        Args:
            rank: Test parameter `rank`.
        """
        return typing.cast("np.ndarray[tuple[int, ...], np.dtype[np.generic]]", await typing.cast("EagerCallableProtocol", ring_all_reduce)(inputs_max[rank], "MIN", mesh_min[rank], rank))

    results_min = await asyncio.gather(*[run_rank_min(r) for r in range(world_size)])
    expected_min = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    for res_min in results_min:
        assert np.allclose(res_min, expected_min)

    # Edge cases
    # world_size <= 1
    single = np.array([1.0, 2.0], dtype=np.float32)
    res_single = await ring_all_reduce(single, "SUM", [None], 0)
    assert typing.cast("EagerCallableProtocol", np.allclose)(res_single, single)

    # Unsupported op
    with pytest.raises(ValueError, match="Unsupported reduction operator"):
        await ring_all_reduce(single, "INVALID", [None, None], 0)

    # Missing neighbor
    with pytest.raises(ValueError, match="Ring topology requires connections"):
        await ring_all_reduce(single, "SUM", [None, None], 0)


@pytest.mark.asyncio
async def test_multi_peer_recursive_doubling_all_gather() -> None:
    """Test multi-peer Recursive Doubling AllGather across 4 concurrent peers."""
    from ml_switcheroo_compiler.distributed.collectives import recursive_halving_doubling_all_gather

    world_size = 4
    mesh = _create_mesh_channels(world_size)

    inputs = [np.array([[float(r)]], dtype=np.float32) for r in range(world_size)]
    expected = np.array([[0.0], [1.0], [2.0], [3.0]], dtype=np.float32)

    async def run_rank(rank: int) -> np.ndarray[tuple[int, ...], np.dtype[np.generic]]:
        """Test function run_rank.

        Args:
            rank: Test parameter `rank`.
        """
        return typing.cast("np.ndarray[tuple[int, ...], np.dtype[np.generic]]", await typing.cast("EagerCallableProtocol", recursive_halving_doubling_all_gather)(inputs[rank], 0, mesh[rank], rank))

    results = await asyncio.gather(*[run_rank(r) for r in range(world_size)])
    for res in results:
        assert np.allclose(res, expected)

    # Test through all_gather with algorithm="recursive_doubling"
    mesh2 = _create_mesh_channels(world_size)

    async def run_rank2(rank: int) -> np.ndarray[tuple[int, ...], np.dtype[np.generic]]:
        """Test function run_rank2.

        Args:
            rank: Test parameter `rank`.
        """
        return typing.cast("np.ndarray[tuple[int, ...], np.dtype[np.generic]]", await typing.cast("EagerCallableProtocol", all_gather)(inputs[rank], 0, mesh2[rank], rank, algorithm="recursive_doubling"))

    results2 = await asyncio.gather(*[run_rank2(r) for r in range(world_size)])
    for res in results2:
        assert np.allclose(res, expected)

    # Edge case: world_size <= 1
    res_single = await recursive_halving_doubling_all_gather(inputs[0], 0, [None], 0)
    assert typing.cast("EagerCallableProtocol", np.allclose)(res_single, inputs[0])

    # Non-power-of-two world_size (e.g. 3) to exercise partner_rank >= world_size branches
    world_size_3 = 3
    mesh_3 = _create_mesh_channels(world_size_3)
    inputs_3 = [np.array([[float(r)]], dtype=np.float32) for r in range(world_size_3)]

    async def run_rank_3(rank: int) -> np.ndarray[tuple[int, ...], np.dtype[np.generic]]:
        """Test function run_rank_3.

        Args:
            rank: Test parameter `rank`.
        """
        return typing.cast("np.ndarray[tuple[int, ...], np.dtype[np.generic]]", await typing.cast("EagerCallableProtocol", recursive_halving_doubling_all_gather)(inputs_3[rank], 0, mesh_3[rank], rank))

    results_3 = await asyncio.gather(*[run_rank_3(r) for r in range(world_size_3)])
    assert len(results_3) == 3

    # Partial connection mesh with None partner connection
    mesh_partial = [[None, None], [None, None]]
    res_partial = await recursive_halving_doubling_all_gather(inputs[0], 0, mesh_partial[0], 0)
    assert typing.cast("EagerCallableProtocol", np.allclose)(res_partial, inputs[0])


@pytest.mark.asyncio
async def test_recv_data_crc_mismatch() -> None:
    """Test ValueError raised on CRC32 checksum mismatch in chunk reception."""
    import json
    import struct

    conn = MockDataChannel()
    header = b"META:" + json.dumps({"shape": [4], "dtype": "float32", "total_bytes": 16, "num_chunks": 1}).encode("utf-8")
    corrupt_crc = 12345
    chunk_raw = b"CHUNK:" + struct.pack("!6sIII", b"CHUNK:", 0, 1, corrupt_crc) + b"1234567812345678"
    conn.recv_queue = [header, chunk_raw]

    with pytest.raises(ValueError, match="Checksum mismatch on received WebRTC chunk"):
        await typing.cast("EagerCallableProtocol", _recv_data)(conn)


async def _test_multi_peer_ring_reduce_scatter_part_1(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 1 of test_multi_peer_ring_reduce_scatter.

    Args:
        _state: Shared test state dictionary.
    """
    world_size = 4
    mesh = _create_mesh_channels(world_size)

    # SUM test: each rank has 4 elements: [1, 2, 3, 4] * (rank + 1)
    # Total sum: [10, 20, 30, 40]
    # Scatter dim 0 -> rank r gets [10 * (r + 1)]
    inputs = [np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32) * (r + 1) for r in range(world_size)]

    async def run_rank_sum(rank: int) -> np.ndarray[tuple[int, ...], np.dtype[np.generic]]:
        """Test function run_rank_sum.

        Args:
            rank: Test parameter `rank`.
        """
        return typing.cast("np.ndarray[tuple[int, ...], np.dtype[np.generic]]", await typing.cast("EagerCallableProtocol", ring_reduce_scatter)(inputs[rank], "SUM", 0, mesh[rank], rank))

    results = await asyncio.gather(*[run_rank_sum(r) for r in range(world_size)])
    for r, res in enumerate(results):
        assert np.allclose(res, np.array([(r + 1) * 10.0], dtype=np.float32))

    # Test through reduce_scatter with algorithm="ring"
    mesh_ring = _create_mesh_channels(world_size)

    async def run_rank_ring(rank: int) -> np.ndarray[tuple[int, ...], np.dtype[np.generic]]:
        """Test function run_rank_ring.

        Args:
            rank: Test parameter `rank`.
        """
        return typing.cast("np.ndarray[tuple[int, ...], np.dtype[np.generic]]", await typing.cast("EagerCallableProtocol", reduce_scatter)(inputs[rank], "SUM", 0, mesh_ring[rank], rank, algorithm="ring"))

    results_ring = await asyncio.gather(*[run_rank_ring(r) for r in range(world_size)])
    for r, res in enumerate(results_ring):
        assert np.allclose(res, np.array([(r + 1) * 10.0], dtype=np.float32))

    # PROD test
    mesh_prod = _create_mesh_channels(world_size)
    getattr(_state, "update")({"inputs": inputs, "mesh": mesh, "mesh_prod": mesh_prod, "mesh_ring": mesh_ring})
    getattr(_state, "update")({"results": results, "results_ring": results_ring, "run_rank_ring": run_rank_ring, "run_rank_sum": run_rank_sum})
    getattr(_state, "update")({"world_size": world_size})
    _state.update(locals())


async def _test_multi_peer_ring_reduce_scatter_part_2(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 2 of test_multi_peer_ring_reduce_scatter.

    Args:
        _state: Shared test state dictionary.
    """
    mesh_prod, world_size = _state["mesh_prod"], _state["world_size"]
    inputs_prod = [np.array([2.0, 3.0, 4.0, 5.0], dtype=np.float32) for _ in range(world_size)]

    async def run_rank_prod(rank: int) -> np.ndarray[tuple[int, ...], np.dtype[np.generic]]:
        """Test function run_rank_prod.

        Args:
            rank: Test parameter `rank`.
        """
        return typing.cast("np.ndarray[tuple[int, ...], np.dtype[np.generic]]", await typing.cast("EagerCallableProtocol", ring_reduce_scatter)(inputs_prod[rank], "PROD", 0, mesh_prod[rank], rank))

    results_prod = await asyncio.gather(*[run_rank_prod(r) for r in range(world_size)])
    expected_prod = [2.0**4, 3.0**4, 4.0**4, 5.0**4]
    for r, res in enumerate(results_prod):
        assert np.allclose(res, np.array([expected_prod[r]], dtype=np.float32))

    # MAX test
    mesh_max = typing.cast("EagerCallableProtocol", _create_mesh_channels)(world_size)
    inputs_max = [np.array([float(r), float(r * 2), float(r * 3), float(r * 4)], dtype=np.float32) for r in range(world_size)]

    async def run_rank_max(rank: int) -> np.ndarray[tuple[int, ...], np.dtype[np.generic]]:
        """Test function run_rank_max.

        Args:
            rank: Test parameter `rank`.
        """
        return typing.cast("np.ndarray[tuple[int, ...], np.dtype[np.generic]]", await typing.cast("EagerCallableProtocol", ring_reduce_scatter)(inputs_max[rank], "MAX", 0, mesh_max[rank], rank))

    results_max = await asyncio.gather(*[run_rank_max(r) for r in range(world_size)])
    expected_max = [3.0, 6.0, 9.0, 12.0]
    getattr(_state, "update")({"expected_max": expected_max, "expected_prod": expected_prod, "inputs_max": inputs_max, "inputs_prod": inputs_prod})
    getattr(_state, "update")({"mesh_max": mesh_max, "results_max": results_max, "results_prod": results_prod, "run_rank_max": run_rank_max})
    getattr(_state, "update")({"run_rank_prod": run_rank_prod})
    _state.update(locals())


async def _test_multi_peer_ring_reduce_scatter_part_3(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 3 of test_multi_peer_ring_reduce_scatter.

    Args:
        _state: Shared test state dictionary.
    """
    expected_max, inputs_max, results_max, world_size = _state["expected_max"], _state["inputs_max"], _state["results_max"], _state["world_size"]
    for r, res in enumerate(results_max):
        assert typing.cast("EagerCallableProtocol", np.allclose)(res, typing.cast("EagerCallableProtocol", np.array)([expected_max[r]], dtype=np.float32))

    # MIN test
    mesh_min = typing.cast("EagerCallableProtocol", _create_mesh_channels)(world_size)

    async def run_rank_min(rank: int) -> np.ndarray[tuple[int, ...], np.dtype[np.generic]]:
        """Test function run_rank_min.

        Args:
            rank: Test parameter `rank`.
        """
        return typing.cast("np.ndarray[tuple[int, ...], np.dtype[np.generic]]", await typing.cast("EagerCallableProtocol", ring_reduce_scatter)(inputs_max[rank], "MIN", 0, mesh_min[rank], rank))

    results_min = await asyncio.gather(*[run_rank_min(r) for r in range(world_size)])
    for res_min in results_min:
        assert typing.cast("EagerCallableProtocol", np.allclose)(res_min, typing.cast("EagerCallableProtocol", np.array)([0.0], dtype=np.float32))

    # Edge cases:
    # 1. world_size <= 1
    single = np.array([1.0, 2.0], dtype=np.float32)
    res_single = await ring_reduce_scatter(single, "SUM", 0, [None], 0)
    assert typing.cast("EagerCallableProtocol", np.allclose)(res_single, single)

    # 2. Unsupported op
    with pytest.raises(ValueError, match="Unsupported reduction operator"):
        await ring_reduce_scatter(single, "INVALID", 0, [None, None], 0)

    # 3. Missing neighbor
    with pytest.raises(ValueError, match="Ring topology requires connections"):
        await ring_reduce_scatter(single, "SUM", 0, [None, None], 0)
    getattr(_state, "update")({"mesh_min": mesh_min, "res_single": res_single, "results_min": results_min, "run_rank_min": run_rank_min})
    getattr(_state, "update")({"single": single})


@pytest.mark.asyncio
async def test_multi_peer_ring_reduce_scatter() -> None:
    """Test multi-peer Ring ReduceScatter across concurrent peers and edge branches."""
    _state: dict[str, EagerTensorProtocol] = {}
    await _test_multi_peer_ring_reduce_scatter_part_1(_state)
    await _test_multi_peer_ring_reduce_scatter_part_2(_state)
    await _test_multi_peer_ring_reduce_scatter_part_3(_state)
    assert len(_state) > 0
