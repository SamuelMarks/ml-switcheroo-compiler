"""Zero-copy collective operations for distributed execution over WebRTC."""

from __future__ import annotations

import asyncio
import json
import logging
import struct
import sys
import zlib
from collections.abc import Sequence
from typing import Callable, cast

from ml_switcheroo_compiler.backends.eager_registry import EagerTensorProtocol, EagerValue
from ml_switcheroo_compiler.backends.registry import get_active_backend

DEFAULT_CHUNK_SIZE: int = 16384
HIGH_WATERMARK_BYTES: int = 65536
LOW_WATERMARK_BYTES: int = 16384


class WebRTCDataChannelStream:
    """Stream manager wrapping WebRTC RTCDataChannel event-driven communication."""

    channel: EagerValue
    chunk_size: int
    high_watermark: int
    low_watermark: int
    _recv_queue: asyncio.Queue[bytes]
    _drain_event: asyncio.Event

    def __init__(
        self,
        channel: EagerValue,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        high_watermark: int = HIGH_WATERMARK_BYTES,
        low_watermark: int = LOW_WATERMARK_BYTES,
    ) -> None:
        """Initialize WebRTCDataChannelStream.

        Args:
            channel (EagerValue): Underlying WebRTC RTCDataChannel or compatible mock.
            chunk_size (int): Size of individual binary chunks in bytes.
            high_watermark (int): Backpressure threshold in bytes.
            low_watermark (int): Drain threshold in bytes to resume transmission.
        """
        self.channel = channel
        self.chunk_size = chunk_size
        self.high_watermark = high_watermark
        self.low_watermark = low_watermark
        self._recv_queue = asyncio.Queue()
        self._drain_event = asyncio.Event()
        self._drain_event.set()
        self._setup_event_handlers()

    def _setup_event_handlers(self) -> None:
        """Attach onmessage and onbufferedamountlow handlers if supported by channel."""
        if hasattr(self.channel, "onmessage"):
            setattr(self.channel, "onmessage", self._on_message)
        elif hasattr(self.channel, "on_message"):
            setattr(self.channel, "on_message", self._on_message)

        if hasattr(self.channel, "bufferedAmountLowThreshold"):
            setattr(self.channel, "bufferedAmountLowThreshold", self.low_watermark)
        elif hasattr(self.channel, "buffered_amount_low_threshold"):
            setattr(self.channel, "buffered_amount_low_threshold", self.low_watermark)

        if hasattr(self.channel, "onbufferedamountlow"):
            setattr(self.channel, "onbufferedamountlow", self._on_buffered_amount_low)
        elif hasattr(self.channel, "on_buffered_amount_low"):
            setattr(self.channel, "on_buffered_amount_low", self._on_buffered_amount_low)

    def _on_message(self, event_or_data: EagerValue) -> None:
        """Event listener for incoming RTCDataChannel binary and text messages.

        Args:
            event_or_data (EagerValue): RTCDataChannel message event or raw byte payload.
        """
        payload = getattr(event_or_data, "data", event_or_data)
        if isinstance(payload, bytes):
            self._recv_queue.put_nowait(payload)
        elif isinstance(payload, str):
            self._recv_queue.put_nowait(payload.encode("utf-8"))
        elif hasattr(payload, "tobytes"):
            self._recv_queue.put_nowait(bytes(getattr(payload, "tobytes")()))
        else:
            self._recv_queue.put_nowait(cast("Callable[..., bytes]", bytes)(payload))

    def _on_buffered_amount_low(self, event: EagerValue | None = None) -> None:
        """Event listener for RTCDataChannel bufferedamountlow event.

        Args:
            event (Optional[object]): Low watermark drain event.
        """
        self._drain_event.set()

    def get_buffered_amount(self) -> int:
        """Retrieve the number of queued bytes in the send buffer.

        Returns:
            int: Number of queued bytes.
        """
        amt = getattr(self.channel, "bufferedAmount", getattr(self.channel, "buffered_amount", 0))
        if callable(amt):
            return int(amt())
        return int(amt)

    async def wait_backpressure(self) -> None:
        """Pause execution when bufferedAmount exceeds high watermark until drained."""
        if self.get_buffered_amount() > self.high_watermark:
            self._drain_event.clear()
            drain_fn = getattr(self.channel, "wait_drain", None)
            if callable(drain_fn):
                if asyncio.iscoroutinefunction(drain_fn):
                    await drain_fn()
                else:
                    drain_fn()
                self._drain_event.set()
                return

            while self.get_buffered_amount() > self.low_watermark:
                try:
                    await asyncio.wait_for(self._drain_event.wait(), timeout=0.01)
                    break
                except asyncio.TimeoutError:
                    logging.getLogger(__name__).debug("Ignored exception in %s", __name__)

    async def send_bytes(self, data: bytes) -> None:
        """Send raw bytes over the underlying channel.

        Args:
            data (bytes): Byte payload to transmit.
        """
        if hasattr(self.channel, "send"):
            send_fn = getattr(self.channel, "send")
            if asyncio.iscoroutinefunction(send_fn):
                await send_fn(data)
            else:
                res = send_fn(data)
                if asyncio.iscoroutine(res):
                    await res
        elif hasattr(self.channel, "sendMessage"):
            send_fn = getattr(self.channel, "sendMessage")
            if asyncio.iscoroutinefunction(send_fn):
                await send_fn(data)
            else:
                res = send_fn(data)
                if asyncio.iscoroutine(res):
                    await res

    async def recv_bytes(self) -> bytes:
        """Receive the next chunk from the incoming queue or channel.

        Returns:
            bytes: Received chunk payload.
        """
        if not self._recv_queue.empty():
            return await self._recv_queue.get()
        recv_fn = getattr(self.channel, "recv", None)
        if callable(recv_fn):
            if asyncio.iscoroutinefunction(recv_fn):
                res = await recv_fn()
            else:
                res = recv_fn()
                if asyncio.iscoroutine(res):
                    res = await res
            return bytes(res) if not isinstance(res, bytes) else res
        return await self._recv_queue.get()


async def _send_data(conn: EagerValue, data: EagerValue, chunk_size: int = DEFAULT_CHUNK_SIZE) -> None:
    """Send tensor over WebRTC DataChannel using chunked binary streaming protocol.

    Args:
        conn (EagerValue): WebRTC DataChannel, stream adapter, or async connection.
        data (EagerValue): Tensor to serialize and stream.
        chunk_size (int): Size of individual binary chunks in bytes.
    """
    stream = conn if isinstance(conn, WebRTCDataChannelStream) else WebRTCDataChannelStream(conn, chunk_size=chunk_size)
    arr = cast(EagerTensorProtocol, data)
    dtype_obj = arr.dtype
    arr_byteorder = str(getattr(dtype_obj, "byteorder", "="))

    native_order = "<" if sys.byteorder == "little" else ">"
    byteorder = native_order if arr_byteorder in ("=", "|") else arr_byteorder

    raw_bytes = bytes(arr.tobytes())
    total_bytes = len(raw_bytes)
    num_chunks = max(1, (total_bytes + chunk_size - 1) // chunk_size)

    meta = {
        "protocol": "webrtc_stream_v1",
        "shape": list(arr.shape),
        "dtype": str(getattr(dtype_obj, "str", dtype_obj)),
        "byteorder": byteorder,
        "total_bytes": total_bytes,
        "num_chunks": num_chunks,
        "chunk_size": chunk_size,
    }
    meta_bytes = b"META:" + json.dumps(meta).encode("utf-8")

    await stream.wait_backpressure()
    await stream.send_bytes(meta_bytes)

    for i in range(num_chunks):
        start = i * chunk_size
        end = min(start + chunk_size, total_bytes)
        chunk = raw_bytes[start:end]
        chksum: int = zlib.crc32(chunk)
        frame_header: bytes = struct.pack("!6sIII", b"CHUNK:", i, num_chunks, chksum)
        chunk_frame: bytes = frame_header + chunk
        await stream.wait_backpressure()
        await stream.send_bytes(chunk_frame)


async def _recv_data(conn: EagerValue) -> EagerValue:
    """Receive tensor from WebRTC DataChannel with chunked reassembly and endianness handling.

    Args:
        conn (EagerValue): WebRTC DataChannel, stream adapter, or async connection.

    Returns:
        EagerValue: Reconstructed numpy ndarray.

    Raises:
        ValueError: If received metadata header is corrupt or payload size mismatches.
    """
    stream = conn if isinstance(conn, WebRTCDataChannelStream) else WebRTCDataChannelStream(conn)

    header_bytes = await stream.recv_bytes()
    if header_bytes.startswith(b"META:"):
        meta = json.loads(header_bytes[5:].decode("utf-8"))
        shape = tuple(int(s) for s in meta["shape"])
        dtype_str = str(meta["dtype"])
        num_chunks = int(meta["num_chunks"])
        total_bytes = int(meta["total_bytes"])
        expected_byteorder = str(meta.get("byteorder", "|"))

        chunks_map: dict[int, bytes] = {}
        for _ in range(num_chunks):
            chunk_raw = await stream.recv_bytes()
            if chunk_raw.startswith(b"CHUNK:") and len(chunk_raw) >= 18:
                _, chunk_idx, expected_num, chunk_crc = struct.unpack("!6sIII", chunk_raw[:18])
                chunk_data = chunk_raw[18:]
                actual_crc = zlib.crc32(chunk_data)
                if actual_crc != chunk_crc:
                    raise ValueError(f"Checksum mismatch on received WebRTC chunk {chunk_idx}: expected {chunk_crc}, got {actual_crc}")
                chunks_map[chunk_idx] = chunk_data
            else:
                chunks_map[len(chunks_map)] = chunk_raw

        chunks: list[bytes] = [chunks_map[i] for i in range(num_chunks)]
        payload = b"".join(chunks)
        if len(payload) != total_bytes:
            raise ValueError(f"Payload length mismatch: expected {total_bytes}, got {len(payload)}")

        arr = cast(EagerTensorProtocol, get_active_backend().execute_op("frombuffer", payload, dtype=dtype_str)).reshape(shape)

        native_order = "<" if sys.byteorder == "little" else ">"
        if expected_byteorder in ("<", ">") and expected_byteorder != native_order:
            arr = arr.astype(getattr(arr.dtype, "newbyteorder")("="))

        return cast(EagerValue, arr.copy())
    else:
        meta_str = header_bytes.decode("utf-8")
        shape_str, dtype_str = meta_str.split("|")
        shape = tuple(int(s) for s in shape_str.split(",")) if shape_str else ()
        data_bytes = await stream.recv_bytes()
        arr_raw = cast(EagerTensorProtocol, get_active_backend().execute_op("frombuffer", data_bytes, dtype=dtype_str))
        return cast(EagerValue, arr_raw.reshape(shape).copy())


async def broadcast(
    tensor: EagerValue,
    root_rank: int,
    conns: Sequence[EagerValue | None],
    rank: int,
) -> EagerValue:
    """Execute Broadcast collective operation across WebRTC DataChannels.

    Args:
        tensor (EagerValue): Tensor to broadcast from root or receive buffer.
        root_rank (int): Rank of broadcasting leader node.
        conns (Sequence[Optional[object]]): Connection list indexed by peer rank.
        rank (int): Global rank of current node.

    Returns:
        EagerValue: Received or copied tensor data.

    Raises:
        ValueError: If root connection is missing.
    """
    if rank == root_rank:
        tasks = []
        for i, conn in enumerate(conns):
            if i != rank and conn is not None:
                tasks.append(_send_data(conn, tensor))
        await asyncio.gather(*tasks)
        return cast(EagerTensorProtocol, tensor).copy()
    else:
        conn = conns[root_rank]
        if conn is None:
            raise ValueError(f"Connection to root rank {root_rank} is not established.")
        return await _recv_data(conn)


async def ring_all_reduce(
    tensor: EagerValue,
    op_type: str,
    conns: Sequence[EagerValue | None],
    rank: int,
) -> EagerValue:
    """Execute Ring AllReduce collective operation across cyclic peer DataChannels.

    Args:
        tensor (EagerValue): Local input tensor to reduce across all ranks.
        op_type (str): Reduction operator type ("SUM", "PROD", "MAX", "MIN").
        conns (Sequence[Optional[object]]): Connection list indexed by peer rank.
        rank (int): Global rank of current node.

    Returns:
        EagerValue: Globally reduced result tensor on local rank.

    Raises:
        ValueError: If op_type is unrecognized or ring neighbor connection is missing.
    """
    world_size = len(conns)
    if world_size <= 1:
        return cast(EagerTensorProtocol, tensor).copy()

    op_upper = op_type.upper()
    if op_upper not in ("SUM", "PROD", "MAX", "MIN"):
        raise ValueError(f"Unsupported reduction operator '{op_type}'")

    next_rank = (rank + 1) % world_size
    prev_rank = (rank - 1) % world_size
    next_conn = conns[next_rank]
    prev_conn = conns[prev_rank]
    if next_conn is None or prev_conn is None:
        raise ValueError(f"Ring topology requires connections to neighbors ({prev_rank}, {next_rank})")

    backend = get_active_backend()
    split_chunks = cast("Sequence[EagerTensorProtocol]", backend.execute_op("array_split", tensor, world_size, axis=0))
    chunks: list[EagerValue] = [c.copy() for c in split_chunks]

    def _apply_op(a: EagerValue, b: EagerValue) -> EagerValue:
        """Apply elementwise reduction operation between two chunks.

        Args:
            a (EagerValue): First operand chunk.
            b (EagerValue): Second operand chunk.

        Returns:
            EagerValue: Result of the reduction.
        """
        a_t = cast(EagerTensorProtocol, a)
        b_t = cast(EagerTensorProtocol, b)
        if op_upper == "SUM":
            return a_t + b_t
        elif op_upper == "PROD":
            return a_t * b_t
        elif op_upper == "MAX":
            return backend.execute_op("maximum", a_t, b_t)
        else:
            return backend.execute_op("minimum", a_t, b_t)

    # Phase 1: Scatter-Reduce
    for step in range(world_size - 1):
        send_idx = (rank - step) % world_size
        recv_idx = (rank - step - 1) % world_size

        send_task = _send_data(next_conn, chunks[send_idx])
        recv_task = _recv_data(prev_conn)
        _, received_chunk = await asyncio.gather(send_task, recv_task)
        chunks[recv_idx] = _apply_op(chunks[recv_idx], received_chunk)

    # Phase 2: AllGather
    for step in range(world_size - 1):
        send_idx = (rank - step + 1) % world_size
        recv_idx = (rank - step) % world_size

        send_task = _send_data(next_conn, chunks[send_idx])
        recv_task = _recv_data(prev_conn)
        _, received_chunk = await asyncio.gather(send_task, recv_task)
        chunks[recv_idx] = received_chunk

    return backend.execute_op("concatenate", chunks, axis=0)


async def recursive_halving_doubling_all_gather(
    tensor: EagerValue,
    axis: int,
    conns: Sequence[EagerValue | None],
    rank: int,
) -> EagerValue:
    """Execute Recursive Halving/Doubling AllGather collective across WebRTC DataChannels.

    Args:
        tensor (EagerValue): Local input tensor to contribute to gather.
        axis (int): Dimension along which to concatenate gathered slices.
        conns (Sequence[Optional[object]]): Connection list indexed by peer rank.
        rank (int): Global rank of current node.

    Returns:
        EagerValue: Concatenated global tensor containing all slices in rank order.
    """
    world_size = len(conns)
    if world_size <= 1:
        return cast(EagerTensorProtocol, tensor).copy()

    backend = get_active_backend()
    gathered: dict[int, EagerValue] = {rank: cast(EagerTensorProtocol, tensor).copy()}
    num_steps = (world_size - 1).bit_length()

    for step in range(num_steps):
        distance = 1 << step
        partner_rank = rank ^ distance
        if partner_rank < world_size:
            partner_conn = conns[partner_rank]
            if partner_conn is not None:
                current_ranks = sorted(gathered.keys())
                current_payload = backend.execute_op("concatenate", [gathered[r] for r in current_ranks], axis=axis)

                send_task = _send_data(partner_conn, current_payload)
                recv_task = _recv_data(partner_conn)
                _, partner_payload = await asyncio.gather(send_task, recv_task)

                partner_ranks = [r ^ distance for r in current_ranks if (r ^ distance) < world_size]
                received_slices = cast("Sequence[EagerValue]", backend.execute_op("array_split", partner_payload, len(partner_ranks), axis=axis))
                for pr, slc in zip(partner_ranks, received_slices):
                    gathered[pr] = slc

    valid_slices = [gathered[r] for r in range(world_size) if r in gathered]
    return backend.execute_op("concatenate", valid_slices, axis=axis)


def _apply_reduction_op(a: EagerValue, b: EagerValue, op_type: str) -> EagerValue:
    """Apply an elementwise reduction operator between two tensors.

    Args:
        a (EagerValue): First operand tensor.
        b (EagerValue): Second operand tensor.
        op_type (str): Reduction operator name ("SUM", "PROD", "MAX", "MIN").

    Returns:
        EagerValue: Reduced result tensor.

    Raises:
        ValueError: If op_type is not supported.
    """
    a_t = cast(EagerTensorProtocol, a)
    b_t = cast(EagerTensorProtocol, b)
    if op_type == "SUM":
        return a_t + b_t
    if op_type == "PROD":
        return a_t * b_t
    if op_type == "MAX":
        return get_active_backend().execute_op("maximum", a_t, b_t)
    if op_type == "MIN":
        return get_active_backend().execute_op("minimum", a_t, b_t)
    raise ValueError(f"Unsupported reduction operator '{op_type}'")


async def all_reduce(
    tensor: EagerValue,
    op_type: str,
    conns: Sequence[EagerValue | None],
    rank: int,
    algorithm: str = "direct",
) -> EagerValue:
    """Execute AllReduce collective operation across WebRTC DataChannels.

    Args:
        tensor (EagerValue): Local input tensor to reduce.
        op_type (str): Reduction operator type ("SUM", "PROD", "MAX", "MIN").
        conns (Sequence[Optional[object]]): Connection list indexed by peer rank.
        rank (int): Global rank of current node.
        algorithm (str): AllReduce algorithm choice ("direct", "ring").

    Returns:
        EagerValue: Fully reduced result tensor on local rank.

    Raises:
        ValueError: If op_type is unrecognized.
    """
    if algorithm == "ring":
        return await ring_all_reduce(tensor, op_type, conns, rank)

    res: EagerValue = cast(EagerTensorProtocol, tensor).copy()
    peer_conns = [conn for i, conn in enumerate(conns) if i != rank and conn is not None]
    await asyncio.gather(*[_send_data(conn, tensor) for conn in peer_conns])
    received_tensors = await asyncio.gather(*[_recv_data(conn) for conn in peer_conns])

    for other_tensor in received_tensors:
        res = _apply_reduction_op(res, other_tensor, op_type)

    return res


async def all_gather(
    tensor: EagerValue,
    axis: int,
    conns: Sequence[EagerValue | None],
    rank: int,
    algorithm: str = "direct",
) -> EagerValue:
    """Execute AllGather collective operation across WebRTC DataChannels.

    Args:
        tensor (EagerValue): Local input tensor to contribute to gather.
        axis (int): Dimension along which to concatenate gathered slices.
        conns (Sequence[Optional[object]]): Connection list indexed by peer rank.
        rank (int): Global rank of current node.
        algorithm (str): AllGather algorithm choice ("direct", "recursive_doubling").

    Returns:
        EagerValue: Concatenated global tensor containing all slices in rank order.
    """
    if algorithm == "recursive_doubling":
        return await recursive_halving_doubling_all_gather(tensor, axis, conns, rank)

    world_size = len(conns)
    tensors: list[EagerValue | None] = [None] * world_size
    tensors[rank] = cast(EagerTensorProtocol, tensor).copy()

    send_tasks = []
    for i, conn in enumerate(conns):
        if i != rank and conn is not None:
            send_tasks.append(_send_data(conn, tensor))
    await asyncio.gather(*send_tasks)

    recv_tasks = []
    for i, conn in enumerate(conns):
        if i != rank and conn is not None:
            recv_tasks.append(_recv_data(conn))

    received_tensors = await asyncio.gather(*recv_tasks)

    idx = 0
    for i in range(world_size):
        if i != rank:
            tensors[i] = received_tensors[idx]
            idx += 1

    valid_tensors = [t for t in tensors if t is not None]
    return get_active_backend().execute_op("concatenate", valid_tensors, axis=axis)


async def ring_reduce_scatter(
    tensor: EagerValue,
    op_type: str,
    scatter_dim: int,
    conns: Sequence[EagerValue | None],
    rank: int,
) -> EagerValue:
    """Execute Ring ReduceScatter collective operation across cyclic peer DataChannels.

    Args:
        tensor (EagerValue): Local input tensor to reduce and scatter.
        op_type (str): Reduction operator type ("SUM", "PROD", "MAX", "MIN").
        scatter_dim (int): Axis along which to partition the reduced tensor.
        conns (Sequence[Optional[object]]): Connection list indexed by peer rank.
        rank (int): Global rank of current node.

    Returns:
        EagerValue: The local slice of the globally reduced tensor.

    Raises:
        ValueError: If op_type is unrecognized or ring neighbor connection is missing.
    """
    world_size: int = len(conns)
    if world_size <= 1:
        return cast(EagerTensorProtocol, tensor).copy()

    op_upper: str = op_type.upper()
    if op_upper not in ("SUM", "PROD", "MAX", "MIN"):
        raise ValueError(f"Unsupported reduction operator '{op_type}'")

    next_rank: int = (rank + 1) % world_size
    prev_rank: int = (rank - 1) % world_size
    next_conn = conns[next_rank]
    prev_conn = conns[prev_rank]
    if next_conn is None or prev_conn is None:
        raise ValueError(f"Ring topology requires connections to neighbors ({prev_rank}, {next_rank})")

    backend = get_active_backend()
    split_chunks = cast("Sequence[EagerTensorProtocol]", backend.execute_op("array_split", tensor, world_size, axis=scatter_dim))
    chunks: list[EagerValue] = [c.copy() for c in split_chunks]

    def _apply_op(a: EagerValue, b: EagerValue) -> EagerValue:
        """Apply elementwise reduction between two chunks.

        Args:
            a: The a parameter.
            b: The b parameter.

        Returns:
            EagerValue: Computed result.
        """
        a_t = cast(EagerTensorProtocol, a)
        b_t = cast(EagerTensorProtocol, b)
        if op_upper == "SUM":
            return a_t + b_t
        elif op_upper == "PROD":
            return a_t * b_t
        elif op_upper == "MAX":
            return backend.execute_op("maximum", a_t, b_t)
        else:
            return backend.execute_op("minimum", a_t, b_t)

    for step in range(world_size - 1):
        send_idx: int = (rank - 1 - step) % world_size
        recv_idx: int = (rank - 2 - step) % world_size

        send_task = _send_data(next_conn, chunks[send_idx])
        recv_task = _recv_data(prev_conn)
        _, received_chunk = await asyncio.gather(send_task, recv_task)
        chunks[recv_idx] = _apply_op(chunks[recv_idx], received_chunk)

    return chunks[rank]


async def reduce_scatter(
    tensor: EagerValue,
    op_type: str,
    scatter_dim: int,
    conns: Sequence[EagerValue | None],
    rank: int,
    algorithm: str = "direct",
) -> EagerValue:
    """Execute ReduceScatter collective operation across WebRTC DataChannels.

    Args:
        tensor (EagerValue): Local tensor to reduce and scatter.
        op_type (str): Reduction operator type ("SUM", "PROD", "MAX", "MIN").
        scatter_dim (int): Axis along which to partition the reduced tensor.
        conns (Sequence[Optional[object]]): Connection list indexed by peer rank.
        rank (int): Global rank of current node.
        algorithm (str): Algorithm choice ("direct", "ring").

    Returns:
        EagerValue: The local slice of the globally reduced tensor.
    """
    if algorithm == "ring":
        return await ring_reduce_scatter(tensor, op_type, scatter_dim, conns, rank)

    reduced = await all_reduce(tensor, op_type, conns, rank)
    world_size = len(conns)
    chunks = cast("Sequence[EagerValue]", get_active_backend().execute_op("array_split", reduced, world_size, axis=scatter_dim))
    return chunks[rank]


async def all_to_all(
    tensor: EagerValue,
    scatter_dim: int,
    gather_dim: int,
    conns: Sequence[EagerValue | None],
    rank: int,
) -> EagerValue:
    """Execute AllToAll collective operation across WebRTC DataChannels.

    Each rank splits its input tensor along scatter_dim into world_size equal parts,
    sends part j to rank j, receives part i from rank i, and concatenates
    the received parts along gather_dim.

    Args:
        tensor (EagerValue): Local input tensor to distribute.
        scatter_dim (int): Axis along which to split local tensor.
        gather_dim (int): Axis along which to concatenate received tensors.
        conns (Sequence[Optional[object]]): Connection list indexed by peer rank.
        rank (int): Global rank of current node.

    Returns:
        EagerValue: Assembled tensor after all-to-all exchange.
    """
    world_size = len(conns)
    backend = get_active_backend()
    chunks = cast("Sequence[EagerValue]", backend.execute_op("array_split", tensor, world_size, axis=scatter_dim))

    send_tasks = []
    for j, conn in enumerate(conns):
        if j != rank and conn is not None:
            send_tasks.append(_send_data(conn, chunks[j]))
    await asyncio.gather(*send_tasks)

    recv_tasks = []
    for i, conn in enumerate(conns):
        if i != rank and conn is not None:
            recv_tasks.append(_recv_data(conn))
    received_chunks = await asyncio.gather(*recv_tasks)

    all_chunks: list[EagerValue | None] = [None] * world_size
    all_chunks[rank] = chunks[rank]
    idx = 0
    for i in range(world_size):
        if i != rank:
            all_chunks[i] = received_chunks[idx]
            idx += 1

    valid_chunks = [c for c in all_chunks if c is not None]
    return backend.execute_op("concatenate", valid_chunks, axis=gather_dim)


class DistributedBarrier:
    """A distributed barrier for synchronizing ranks via WebRTC DataChannels."""

    world_size: int
    rank: int
    leader_rank: int

    def __init__(self, world_size: int, rank: int, leader_rank: int = 0) -> None:
        """Initialize DistributedBarrier.

        Args:
            world_size (int): Total number of participating ranks.
            rank (int): Global rank of current node.
            leader_rank (int): Rank designated as synchronization coordinator.
        """
        self.world_size = world_size
        self.rank = rank
        self.leader_rank = leader_rank

    async def wait(self, conns: Sequence[EagerValue | None]) -> None:
        """Wait for all ranks to reach the barrier.

        Args:
            conns (Sequence[Optional[object]]): Peer connections indexed by rank.
        """
        if self.rank != self.leader_rank:
            conn = conns[self.leader_rank]
            if conn is not None:
                stream = conn if isinstance(conn, WebRTCDataChannelStream) else WebRTCDataChannelStream(conn)
                await stream.send_bytes(b"r")
                await stream.recv_bytes()
        else:
            recv_tasks = []
            for i, conn in enumerate(conns):
                if i != self.rank and conn is not None:
                    stream = conn if isinstance(conn, WebRTCDataChannelStream) else WebRTCDataChannelStream(conn)
                    recv_tasks.append(stream.recv_bytes())
            await asyncio.gather(*recv_tasks)

            send_tasks = []
            for i, conn in enumerate(conns):
                if i != self.rank and conn is not None:
                    stream = conn if isinstance(conn, WebRTCDataChannelStream) else WebRTCDataChannelStream(conn)
                    send_tasks.append(stream.send_bytes(b"g"))
            await asyncio.gather(*send_tasks)


_BACKEND_COLLECTIVE_REGISTRY: dict[str, EagerValue] = {}


def register_collective_backend(backend_name: str, dispatcher: EagerValue) -> None:
    """Register a backend-specific collective dispatcher.

    Args:
        backend_name (str): The name of the backend (e.g. 'pytorch', 'jax', 'cuda').
        dispatcher (EagerValue): Dispatcher callable or instance.
    """
    _BACKEND_COLLECTIVE_REGISTRY[backend_name.lower()] = dispatcher


def get_collective_backend(backend_name: str) -> EagerValue | None:
    """Retrieve a registered collective dispatcher for a backend.

    Args:
        backend_name (str): Target backend name.

    Returns:
        Optional[object]: Registered collective dispatcher or None.
    """
    return _BACKEND_COLLECTIVE_REGISTRY.get(backend_name.lower())


def _dispatch_pytorch_collective(op: str, tensor: EagerValue, kwargs: dict[str, EagerValue]) -> EagerValue | None:
    """Attempt to dispatch collective operation to PyTorch distributed backend.

    Args:
        op (str): Collective operation name.
        tensor (EagerValue): Local input tensor.
        kwargs (dict[str, EagerValue]): Additional collective parameters.

    Returns:
        EagerValue | None: Computed tensor if handled, else None.
    """
    try:
        from ml_switcheroo_compiler.backends.pytorch.distributed_collectives import (
            pytorch_all_gather,
            pytorch_all_reduce,
            pytorch_broadcast,
            pytorch_reduce_scatter,
        )

        group = kwargs.get("group")
        if op == "AllReduce":
            return pytorch_all_reduce(tensor, op=str(kwargs.get("op_type", "SUM")), group=group)
        if op == "AllGather":
            return pytorch_all_gather(tensor, axis=int(cast(int, kwargs.get("axis", 0))), group=group)
        if op == "ReduceScatter":
            return pytorch_reduce_scatter(tensor, op=str(kwargs.get("op_type", "SUM")), scatter_dim=int(cast(int, kwargs.get("scatter_dim", 0))), group=group)
        if op == "Broadcast":
            return pytorch_broadcast(tensor, src=int(cast(int, kwargs.get("root", 0))), group=group)
    except (ImportError, AttributeError, TypeError, ValueError, KeyError, RuntimeError, OSError):
        logging.getLogger(__name__).debug("Ignored exception in %s", __name__)
    return None


def _dispatch_jax_collective(op: str, tensor: EagerValue, kwargs: dict[str, EagerValue]) -> EagerValue | None:
    """Attempt to dispatch collective operation to JAX distributed backend.

    Args:
        op (str): Collective operation name.
        tensor (EagerValue): Local input tensor.
        kwargs (dict[str, EagerValue]): Additional collective parameters.

    Returns:
        EagerValue | None: Computed tensor if handled, else None.
    """
    try:
        from ml_switcheroo_compiler.backends.jax.distributed_collectives import (
            jax_all_gather,
            jax_all_reduce,
            jax_broadcast,
        )

        axis_name = str(kwargs.get("axis_name", "data"))
        if op == "AllReduce":
            return jax_all_reduce(tensor, op=str(kwargs.get("op_type", "SUM")), axis_name=axis_name)
        if op == "AllGather":
            return jax_all_gather(tensor, axis_name=axis_name, axis=int(cast(int, kwargs.get("axis", 0))))
        if op == "Broadcast":
            return jax_broadcast(tensor, src=int(cast(int, kwargs.get("root", 0))))
    except (ImportError, AttributeError, TypeError, ValueError, KeyError, RuntimeError, OSError):
        logging.getLogger(__name__).debug("Ignored exception in %s", __name__)
    return None


def _reduce_rank_sequence(all_ranks_data: Sequence[EagerValue], red_op: str) -> EagerValue:
    """Reduce a sequence of tensors across all ranks using the specified operator.

    Args:
        all_ranks_data (Sequence[EagerValue]): Tensors from all participating ranks.
        red_op (str): Uppercase reduction operator string.

    Returns:
        EagerValue: Reduced result tensor.
    """
    backend = get_active_backend()
    accum = cast(EagerTensorProtocol, all_ranks_data[0]).copy()
    for other in all_ranks_data[1:]:
        other_t = cast(EagerTensorProtocol, other)
        if red_op in ("SUM", "ADD"):
            accum = accum + other_t
        elif red_op in ("PROD", "PRODUCT"):
            accum = accum * other_t
        elif red_op == "MAX":
            accum = cast(EagerTensorProtocol, backend.execute_op("maximum", accum, other_t))
        elif red_op == "MIN":
            accum = cast(EagerTensorProtocol, backend.execute_op("minimum", accum, other_t))
    return accum


def _simulate_multi_rank_collective(
    op: str,
    all_ranks_data: Sequence[EagerValue],
    kwargs: dict[str, EagerValue],
) -> EagerValue | None:
    """Simulate a collective operation across multi-rank data on the host.

    Args:
        op (str): Collective operation name.
        all_ranks_data (Sequence[EagerValue]): List or tuple of tensors for all ranks.
        kwargs (dict[str, EagerValue]): Additional collective parameters.

    Returns:
        EagerValue | None: Simulated collective result if op is recognized, else None.
    """
    backend = get_active_backend()
    world_size = len(all_ranks_data)
    rank = int(cast(int, kwargs.get("rank", 0)))
    red_op = str(kwargs.get("op_type", "SUM")).upper()

    if op == "AllReduce":
        return _reduce_rank_sequence(all_ranks_data, red_op)
    if op == "AllGather":
        return backend.execute_op("concatenate", all_ranks_data, axis=int(cast(int, kwargs.get("axis", 0))))
    if op == "ReduceScatter":
        accum = _reduce_rank_sequence(all_ranks_data, red_op)
        chunks = cast("Sequence[EagerValue]", backend.execute_op("array_split", accum, world_size, axis=int(cast(int, kwargs.get("scatter_dim", 0)))))
        return chunks[rank]
    if op == "AllToAll":
        scatter_dim = int(cast(int, kwargs.get("scatter_dim", 0)))
        gather_dim = int(cast(int, kwargs.get("gather_dim", 0)))
        gathered = [cast("Sequence[EagerValue]", backend.execute_op("array_split", all_ranks_data[src], world_size, axis=scatter_dim))[rank] for src in range(world_size)]
        return backend.execute_op("concatenate", gathered, axis=gather_dim)
    if op == "Broadcast":
        return cast(EagerTensorProtocol, all_ranks_data[int(cast(int, kwargs.get("root", 0)))]).copy()
    return None


def dispatch_collective(
    op: str,
    tensor: EagerValue,
    backend: str = "numpy",
    **kwargs: EagerValue,
) -> EagerValue:
    """Dispatch a collective operation to the designated backend runtime.

    Args:
        op (str): Collective operation name ("AllReduce", "AllGather", "ReduceScatter", "AllToAll", "Broadcast").
        tensor (EagerValue): Local tensor data.
        backend (str): Target execution backend ("pytorch", "jax", "cuda", "numpy").
        **kwargs (EagerValue): Additional collective parameters (op_type, axis, rank, world_size, all_ranks_data).

    Returns:
        EagerValue: Evaluated collective result.
    """
    backend_key = backend.lower()
    custom_disp = get_collective_backend(backend_key)
    if custom_disp is not None and callable(custom_disp):
        return custom_disp(op, tensor, **kwargs)

    res: EagerValue | None = None
    if backend_key in ("pytorch", "torch"):
        res = _dispatch_pytorch_collective(op, tensor, kwargs)
    elif backend_key == "jax":
        res = _dispatch_jax_collective(op, tensor, kwargs)

    if res is not None:
        return res

    all_ranks_data = kwargs.get("all_ranks_data")
    if isinstance(all_ranks_data, (list, tuple)) and len(all_ranks_data) > 0:
        sim_res = _simulate_multi_rank_collective(op, all_ranks_data, kwargs)
        if sim_res is not None:
            return sim_res

    return cast(EagerTensorProtocol, tensor).copy()
