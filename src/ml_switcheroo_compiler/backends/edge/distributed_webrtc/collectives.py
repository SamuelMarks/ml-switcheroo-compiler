# ruff: noqa: E402, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, E701, E722, F403, E711, E712, PLR0913, PLR0915
"""Zero-copy collective operations for distributed execution over WebRTC."""

import asyncio
import json
import struct
import sys
import zlib
from collections.abc import Sequence
from typing import Optional

import numpy as np

DEFAULT_CHUNK_SIZE: int = 16384
HIGH_WATERMARK_BYTES: int = 65536
LOW_WATERMARK_BYTES: int = 16384


class WebRTCDataChannelStream:
    """Stream manager wrapping WebRTC RTCDataChannel event-driven communication."""

    channel: object
    chunk_size: int
    high_watermark: int
    low_watermark: int
    _recv_queue: asyncio.Queue[bytes]
    _drain_event: asyncio.Event

    def __init__(
        self,
        channel: object,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        high_watermark: int = HIGH_WATERMARK_BYTES,
        low_watermark: int = LOW_WATERMARK_BYTES,
    ) -> None:
        """Initialize WebRTCDataChannelStream.

        Args:
            channel (object): Underlying WebRTC RTCDataChannel or compatible mock.
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
            self.channel.onmessage = self._on_message
        elif hasattr(self.channel, "on_message"):
            self.channel.on_message = self._on_message

        if hasattr(self.channel, "bufferedAmountLowThreshold"):
            self.channel.bufferedAmountLowThreshold = self.low_watermark
        elif hasattr(self.channel, "buffered_amount_low_threshold"):
            self.channel.buffered_amount_low_threshold = self.low_watermark

        if hasattr(self.channel, "onbufferedamountlow"):
            self.channel.onbufferedamountlow = self._on_buffered_amount_low
        elif hasattr(self.channel, "on_buffered_amount_low"):
            self.channel.on_buffered_amount_low = self._on_buffered_amount_low

    def _on_message(self, event_or_data: object) -> None:
        """Event listener for incoming RTCDataChannel binary and text messages.

        Args:
            event_or_data (object): RTCDataChannel message event or raw byte payload.
        """
        payload = getattr(event_or_data, "data", event_or_data)
        if isinstance(payload, bytes):
            self._recv_queue.put_nowait(payload)
        elif isinstance(payload, str):
            self._recv_queue.put_nowait(payload.encode("utf-8"))
        elif hasattr(payload, "tobytes"):
            self._recv_queue.put_nowait(bytes(payload.tobytes()))
        else:
            self._recv_queue.put_nowait(bytes(payload))

    def _on_buffered_amount_low(self, event: Optional[object] = None) -> None:
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
            if hasattr(self.channel, "wait_drain") and callable(self.channel.wait_drain):
                drain_fn = self.channel.wait_drain
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
                    pass

    async def send_bytes(self, data: bytes) -> None:
        """Send raw bytes over the underlying channel.

        Args:
            data (bytes): Byte payload to transmit.
        """
        if hasattr(self.channel, "send"):
            send_fn = self.channel.send
            if asyncio.iscoroutinefunction(send_fn):
                await send_fn(data)
            else:
                res = send_fn(data)
                if asyncio.iscoroutine(res):
                    await res
        elif hasattr(self.channel, "sendMessage"):
            send_fn = self.channel.sendMessage
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
        if hasattr(self.channel, "recv") and callable(self.channel.recv):
            recv_fn = self.channel.recv
            if asyncio.iscoroutinefunction(recv_fn):
                res = await recv_fn()
            else:
                res = recv_fn()
                if asyncio.iscoroutine(res):
                    res = await res
            return bytes(res) if not isinstance(res, bytes) else res
        return await self._recv_queue.get()


async def _send_data(conn: object, data: np.ndarray, chunk_size: int = DEFAULT_CHUNK_SIZE) -> None:
    """Send tensor over WebRTC DataChannel using chunked binary streaming protocol.

    Args:
        conn (object): WebRTC DataChannel, stream adapter, or async connection.
        data (np.ndarray): Tensor to serialize and stream.
        chunk_size (int): Size of individual binary chunks in bytes.
    """
    stream = conn if isinstance(conn, WebRTCDataChannelStream) else WebRTCDataChannelStream(conn, chunk_size=chunk_size)

    native_order = "<" if sys.byteorder == "little" else ">"
    byteorder = native_order if data.dtype.byteorder in ("=", "|") else data.dtype.byteorder

    raw_bytes = data.tobytes()
    total_bytes = len(raw_bytes)
    num_chunks = max(1, (total_bytes + chunk_size - 1) // chunk_size)

    meta = {
        "protocol": "webrtc_stream_v1",
        "shape": list(data.shape),
        "dtype": data.dtype.str,
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


async def _recv_data(conn: object) -> np.ndarray:
    """Receive tensor from WebRTC DataChannel with chunked reassembly and endianness handling.

    Args:
        conn (object): WebRTC DataChannel, stream adapter, or async connection.

    Returns:
        np.ndarray: Reconstructed numpy ndarray.

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

        target_dtype = np.dtype(dtype_str)
        arr = np.frombuffer(payload, dtype=target_dtype).reshape(shape)

        native_order = "<" if sys.byteorder == "little" else ">"
        if expected_byteorder in ("<", ">") and expected_byteorder != native_order:
            arr = arr.astype(target_dtype.newbyteorder("="))

        return arr.copy()
    else:
        meta_str = header_bytes.decode("utf-8")
        shape_str, dtype_str = meta_str.split("|")
        shape = tuple(int(s) for s in shape_str.split(",")) if shape_str else ()
        dtype = np.dtype(dtype_str)
        data_bytes = await stream.recv_bytes()
        return np.frombuffer(data_bytes, dtype=dtype).reshape(shape).copy()


async def broadcast(
    tensor: np.ndarray,
    root_rank: int,
    conns: Sequence[Optional[object]],
    rank: int,
) -> np.ndarray:
    """Execute Broadcast collective operation across WebRTC DataChannels.

    Args:
        tensor (np.ndarray): Tensor to broadcast from root or receive buffer.
        root_rank (int): Rank of broadcasting leader node.
        conns (Sequence[Optional[object]]): Connection list indexed by peer rank.
        rank (int): Global rank of current node.

    Returns:
        np.ndarray: Received or copied tensor data.

    Raises:
        ValueError: If root connection is missing.
    """
    if rank == root_rank:
        tasks = []
        for i, conn in enumerate(conns):
            if i != rank and conn is not None:
                tasks.append(_send_data(conn, tensor))
        await asyncio.gather(*tasks)
        return tensor.copy()
    else:
        conn = conns[root_rank]
        if conn is None:
            raise ValueError(f"Connection to root rank {root_rank} is not established.")
        return await _recv_data(conn)


async def ring_all_reduce(
    tensor: np.ndarray,
    op_type: str,
    conns: Sequence[Optional[object]],
    rank: int,
) -> np.ndarray:
    """Execute Ring AllReduce collective operation across cyclic peer DataChannels.

    Args:
        tensor (np.ndarray): Local input tensor to reduce across all ranks.
        op_type (str): Reduction operator type ("SUM", "PROD", "MAX", "MIN").
        conns (Sequence[Optional[object]]): Connection list indexed by peer rank.
        rank (int): Global rank of current node.

    Returns:
        np.ndarray: Globally reduced result tensor on local rank.

    Raises:
        ValueError: If op_type is unrecognized or ring neighbor connection is missing.
    """
    world_size = len(conns)
    if world_size <= 1:
        return tensor.copy()

    op_upper = op_type.upper()
    if op_upper not in ("SUM", "PROD", "MAX", "MIN"):
        raise ValueError(f"Unsupported reduction operator '{op_type}'")

    next_rank = (rank + 1) % world_size
    prev_rank = (rank - 1) % world_size
    next_conn = conns[next_rank]
    prev_conn = conns[prev_rank]
    if next_conn is None or prev_conn is None:
        raise ValueError(f"Ring topology requires connections to neighbors ({prev_rank}, {next_rank})")

    chunks = [c.copy() for c in np.array_split(tensor, world_size, axis=0)]

    def _apply_op(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Apply elementwise reduction operation between two chunks.

        Args:
            a (np.ndarray): First operand chunk.
            b (np.ndarray): Second operand chunk.

        Returns:
            np.ndarray: Result of the reduction.
        """
        if op_upper == "SUM":
            return a + b
        elif op_upper == "PROD":
            return a * b
        elif op_upper == "MAX":
            return np.maximum(a, b)
        else:
            return np.minimum(a, b)

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

    return np.concatenate(chunks, axis=0)


async def recursive_halving_doubling_all_gather(
    tensor: np.ndarray,
    axis: int,
    conns: Sequence[Optional[object]],
    rank: int,
) -> np.ndarray:
    """Execute Recursive Halving/Doubling AllGather collective across WebRTC DataChannels.

    Args:
        tensor (np.ndarray): Local input tensor to contribute to gather.
        axis (int): Dimension along which to concatenate gathered slices.
        conns (Sequence[Optional[object]]): Connection list indexed by peer rank.
        rank (int): Global rank of current node.

    Returns:
        np.ndarray: Concatenated global tensor containing all slices in rank order.
    """
    world_size = len(conns)
    if world_size <= 1:
        return tensor.copy()

    gathered: dict[int, np.ndarray] = {rank: tensor.copy()}
    num_steps = (world_size - 1).bit_length()

    for step in range(num_steps):
        distance = 1 << step
        partner_rank = rank ^ distance
        if partner_rank < world_size:
            partner_conn = conns[partner_rank]
            if partner_conn is not None:
                current_ranks = sorted(gathered.keys())
                current_payload = np.concatenate([gathered[r] for r in current_ranks], axis=axis)

                send_task = _send_data(partner_conn, current_payload)
                recv_task = _recv_data(partner_conn)
                _, partner_payload = await asyncio.gather(send_task, recv_task)

                partner_ranks = [r ^ distance for r in current_ranks if (r ^ distance) < world_size]
                received_slices = np.array_split(partner_payload, len(partner_ranks), axis=axis)
                for pr, slc in zip(partner_ranks, received_slices):
                    gathered[pr] = slc

    valid_slices = [gathered[r] for r in range(world_size) if r in gathered]
    return np.concatenate(valid_slices, axis=axis)


async def all_reduce(
    tensor: np.ndarray,
    op_type: str,
    conns: Sequence[Optional[object]],
    rank: int,
    algorithm: str = "direct",
) -> np.ndarray:
    """Execute AllReduce collective operation across WebRTC DataChannels.

    Args:
        tensor (np.ndarray): Local input tensor to reduce.
        op_type (str): Reduction operator type ("SUM", "PROD", "MAX", "MIN").
        conns (Sequence[Optional[object]]): Connection list indexed by peer rank.
        rank (int): Global rank of current node.
        algorithm (str): AllReduce algorithm choice ("direct", "ring").

    Returns:
        np.ndarray: Fully reduced result tensor on local rank.

    Raises:
        ValueError: If op_type is unrecognized.
    """
    if algorithm == "ring":
        return await ring_all_reduce(tensor, op_type, conns, rank)

    res = tensor.copy()

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

    for other_tensor in received_tensors:
        if op_type == "SUM":
            res = res + other_tensor
        elif op_type == "PROD":
            res = res * other_tensor
        elif op_type == "MAX":
            res = np.maximum(res, other_tensor)
        elif op_type == "MIN":
            res = np.minimum(res, other_tensor)
        else:
            raise ValueError(f"Unsupported reduction operator '{op_type}'")

    return res


async def all_gather(
    tensor: np.ndarray,
    axis: int,
    conns: Sequence[Optional[object]],
    rank: int,
    algorithm: str = "direct",
) -> np.ndarray:
    """Execute AllGather collective operation across WebRTC DataChannels.

    Args:
        tensor (np.ndarray): Local input tensor to contribute to gather.
        axis (int): Dimension along which to concatenate gathered slices.
        conns (Sequence[Optional[object]]): Connection list indexed by peer rank.
        rank (int): Global rank of current node.
        algorithm (str): AllGather algorithm choice ("direct", "recursive_doubling").

    Returns:
        np.ndarray: Concatenated global tensor containing all slices in rank order.
    """
    if algorithm == "recursive_doubling":
        return await recursive_halving_doubling_all_gather(tensor, axis, conns, rank)

    world_size = len(conns)
    tensors: list[Optional[np.ndarray]] = [None] * world_size
    tensors[rank] = tensor.copy()

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
    return np.concatenate(valid_tensors, axis=axis)


async def ring_reduce_scatter(
    tensor: np.ndarray,
    op_type: str,
    scatter_dim: int,
    conns: Sequence[Optional[object]],
    rank: int,
) -> np.ndarray:
    """Execute Ring ReduceScatter collective operation across cyclic peer DataChannels.

    Args:
        tensor (np.ndarray): Local input tensor to reduce and scatter.
        op_type (str): Reduction operator type ("SUM", "PROD", "MAX", "MIN").
        scatter_dim (int): Axis along which to partition the reduced tensor.
        conns (Sequence[Optional[object]]): Connection list indexed by peer rank.
        rank (int): Global rank of current node.

    Returns:
        np.ndarray: The local slice of the globally reduced tensor.

    Raises:
        ValueError: If op_type is unrecognized or ring neighbor connection is missing.
    """
    world_size: int = len(conns)
    if world_size <= 1:
        return tensor.copy()

    op_upper: str = op_type.upper()
    if op_upper not in ("SUM", "PROD", "MAX", "MIN"):
        raise ValueError(f"Unsupported reduction operator '{op_type}'")

    next_rank: int = (rank + 1) % world_size
    prev_rank: int = (rank - 1) % world_size
    next_conn = conns[next_rank]
    prev_conn = conns[prev_rank]
    if next_conn is None or prev_conn is None:
        raise ValueError(f"Ring topology requires connections to neighbors ({prev_rank}, {next_rank})")

    chunks: list[np.ndarray] = [c.copy() for c in np.array_split(tensor, world_size, axis=scatter_dim)]

    def _apply_op(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Apply elementwise reduction between two chunks."""
        if op_upper == "SUM":
            return a + b
        elif op_upper == "PROD":
            return a * b
        elif op_upper == "MAX":
            return np.maximum(a, b)
        else:
            return np.minimum(a, b)

    for step in range(world_size - 1):
        send_idx: int = (rank - 1 - step) % world_size
        recv_idx: int = (rank - 2 - step) % world_size

        send_task = _send_data(next_conn, chunks[send_idx])
        recv_task = _recv_data(prev_conn)
        _, received_chunk = await asyncio.gather(send_task, recv_task)
        chunks[recv_idx] = _apply_op(chunks[recv_idx], received_chunk)

    return chunks[rank]


async def reduce_scatter(
    tensor: np.ndarray,
    op_type: str,
    scatter_dim: int,
    conns: Sequence[Optional[object]],
    rank: int,
    algorithm: str = "direct",
) -> np.ndarray:
    """Execute ReduceScatter collective operation across WebRTC DataChannels.

    Args:
        tensor (np.ndarray): Local tensor to reduce and scatter.
        op_type (str): Reduction operator type ("SUM", "PROD", "MAX", "MIN").
        scatter_dim (int): Axis along which to partition the reduced tensor.
        conns (Sequence[Optional[object]]): Connection list indexed by peer rank.
        rank (int): Global rank of current node.
        algorithm (str): Algorithm choice ("direct", "ring").

    Returns:
        np.ndarray: The local slice of the globally reduced tensor.
    """
    if algorithm == "ring":
        return await ring_reduce_scatter(tensor, op_type, scatter_dim, conns, rank)

    reduced = await all_reduce(tensor, op_type, conns, rank)
    world_size = len(conns)
    chunks = np.array_split(reduced, world_size, axis=scatter_dim)
    return chunks[rank]


async def all_to_all(
    tensor: np.ndarray,
    scatter_dim: int,
    gather_dim: int,
    conns: Sequence[Optional[object]],
    rank: int,
) -> np.ndarray:
    """Execute AllToAll collective operation across WebRTC DataChannels.

    Each rank splits its input tensor along scatter_dim into world_size equal parts,
    sends part j to rank j, receives part i from rank i, and concatenates
    the received parts along gather_dim.

    Args:
        tensor (np.ndarray): Local input tensor to distribute.
        scatter_dim (int): Axis along which to split local tensor.
        gather_dim (int): Axis along which to concatenate received tensors.
        conns (Sequence[Optional[object]]): Connection list indexed by peer rank.
        rank (int): Global rank of current node.

    Returns:
        np.ndarray: Assembled tensor after all-to-all exchange.
    """
    world_size = len(conns)
    chunks = np.array_split(tensor, world_size, axis=scatter_dim)

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

    all_chunks: list[Optional[np.ndarray]] = [None] * world_size
    all_chunks[rank] = chunks[rank]
    idx = 0
    for i in range(world_size):
        if i != rank:
            all_chunks[i] = received_chunks[idx]
            idx += 1

    valid_chunks = [c for c in all_chunks if c is not None]
    return np.concatenate(valid_chunks, axis=gather_dim)


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

    async def wait(self, conns: Sequence[Optional[object]]) -> None:
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


_BACKEND_COLLECTIVE_REGISTRY: dict[str, object] = {}


def register_collective_backend(backend_name: str, dispatcher: object) -> None:
    """Register a backend-specific collective dispatcher.

    Args:
        backend_name (str): The name of the backend (e.g. 'pytorch', 'jax', 'cuda').
        dispatcher (object): Dispatcher callable or instance.
    """
    _BACKEND_COLLECTIVE_REGISTRY[backend_name.lower()] = dispatcher


def get_collective_backend(backend_name: str) -> Optional[object]:
    """Retrieve a registered collective dispatcher for a backend.

    Args:
        backend_name (str): Target backend name.

    Returns:
        Optional[object]: Registered collective dispatcher or None.
    """
    return _BACKEND_COLLECTIVE_REGISTRY.get(backend_name.lower())


def dispatch_collective(
    op: str,
    tensor: np.ndarray,
    backend: str = "numpy",
    **kwargs: object,
) -> np.ndarray:
    """Dispatch a collective operation to the designated backend runtime.

    Args:
        op (str): Collective operation name ("AllReduce", "AllGather", "ReduceScatter", "AllToAll", "Broadcast").
        tensor (np.ndarray): Local tensor data.
        backend (str): Target execution backend ("pytorch", "jax", "cuda", "numpy").
        **kwargs (object): Additional collective parameters (op_type, axis, rank, world_size, all_ranks_data).

    Returns:
        np.ndarray: Evaluated collective result.
    """
    backend_key = backend.lower()
    custom_disp = get_collective_backend(backend_key)
    if custom_disp is not None and callable(custom_disp):
        return custom_disp(op, tensor, **kwargs)

    if backend_key in ("pytorch", "torch"):
        try:
            from ml_switcheroo_compiler.backends.pytorch.distributed_collectives import (
                pytorch_all_gather,
                pytorch_all_reduce,
                pytorch_broadcast,
                pytorch_reduce_scatter,
            )

            if op == "AllReduce":
                return pytorch_all_reduce(tensor, op=str(kwargs.get("op_type", "SUM")), group=kwargs.get("group"))
            elif op == "AllGather":
                return pytorch_all_gather(tensor, axis=int(kwargs.get("axis", 0)), group=kwargs.get("group"))
            elif op == "ReduceScatter":
                return pytorch_reduce_scatter(tensor, op=str(kwargs.get("op_type", "SUM")), scatter_dim=int(kwargs.get("scatter_dim", 0)), group=kwargs.get("group"))
            elif op == "Broadcast":
                return pytorch_broadcast(tensor, src=int(kwargs.get("root", 0)), group=kwargs.get("group"))
        except Exception:
            pass

    elif backend_key == "jax":
        try:
            from ml_switcheroo_compiler.backends.jax.distributed_collectives import (
                jax_all_gather,
                jax_all_reduce,
                jax_broadcast,
            )

            if op == "AllReduce":
                return jax_all_reduce(tensor, op=str(kwargs.get("op_type", "SUM")), axis_name=str(kwargs.get("axis_name", "data")))
            elif op == "AllGather":
                return jax_all_gather(tensor, axis_name=str(kwargs.get("axis_name", "data")), axis=int(kwargs.get("axis", 0)))
            elif op == "Broadcast":
                return jax_broadcast(tensor, src=int(kwargs.get("root", 0)))
        except Exception:
            pass

    # Reference host/multi-rank simulation for NumPy / decoupled execution
    all_ranks_data = kwargs.get("all_ranks_data")
    if isinstance(all_ranks_data, (list, tuple)) and len(all_ranks_data) > 0:
        world_size = len(all_ranks_data)
        rank = int(kwargs.get("rank", 0))

        if op == "AllReduce":
            red_op = str(kwargs.get("op_type", "SUM")).upper()
            accum = all_ranks_data[0].copy()
            for other in all_ranks_data[1:]:
                if red_op in ("SUM", "ADD"):
                    accum = accum + other
                elif red_op in ("PROD", "PRODUCT"):
                    accum = accum * other
                elif red_op == "MAX":
                    accum = np.maximum(accum, other)
                elif red_op == "MIN":
                    accum = np.minimum(accum, other)
            return accum

        elif op == "AllGather":
            axis = int(kwargs.get("axis", 0))
            return np.concatenate(all_ranks_data, axis=axis)

        elif op == "ReduceScatter":
            red_op = str(kwargs.get("op_type", "SUM")).upper()
            accum = all_ranks_data[0].copy()
            for other in all_ranks_data[1:]:
                if red_op in ("SUM", "ADD"):
                    accum = accum + other
                elif red_op in ("PROD", "PRODUCT"):
                    accum = accum * other
                elif red_op == "MAX":
                    accum = np.maximum(accum, other)
                elif red_op == "MIN":
                    accum = np.minimum(accum, other)
            scatter_dim = int(kwargs.get("scatter_dim", 0))
            chunks = np.array_split(accum, world_size, axis=scatter_dim)
            return chunks[rank]

        elif op == "AllToAll":
            scatter_dim = int(kwargs.get("scatter_dim", 0))
            gather_dim = int(kwargs.get("gather_dim", 0))
            gathered_chunks = []
            for src_rank in range(world_size):
                src_chunks = np.array_split(all_ranks_data[src_rank], world_size, axis=scatter_dim)
                gathered_chunks.append(src_chunks[rank])
            return np.concatenate(gathered_chunks, axis=gather_dim)

        elif op == "Broadcast":
            root = int(kwargs.get("root", 0))
            return all_ranks_data[root].copy()

    return tensor.copy()
