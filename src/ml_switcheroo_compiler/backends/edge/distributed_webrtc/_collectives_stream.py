"""Chunked WebRTC DataChannel stream framing, compression, and backpressure."""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
import struct
import sys
import typing
import zlib

from ml_switcheroo_compiler.backends.adapter_bindings import read_static_attr
from ml_switcheroo_compiler.backends.eager_registry import (
    BackendArg,
    coerce_int,
    coerce_operand_list,
    require_callable,
)
from ml_switcheroo_compiler.backends.registry import get_active_backend

DEFAULT_CHUNK_SIZE: int = 16384
HIGH_WATERMARK_BYTES: int = 65536
LOW_WATERMARK_BYTES: int = 16384


@typing.runtime_checkable
class _OnMessageJsProto(typing.Protocol):
    """Protocol for JS-style WebRTC data channels exposing `onmessage`."""

    onmessage: BackendArg


@typing.runtime_checkable
class _OnMessagePyProto(typing.Protocol):
    """Protocol for Python-style WebRTC data channels exposing `on_message`."""

    on_message: BackendArg


@typing.runtime_checkable
class _BufferedLowJsProto(typing.Protocol):
    """Protocol for JS-style WebRTC data channels exposing `bufferedAmountLowThreshold`."""

    bufferedAmountLowThreshold: int


@typing.runtime_checkable
class _BufferedLowPyProto(typing.Protocol):
    """Protocol for Python-style WebRTC data channels exposing `buffered_amount_low_threshold`."""

    buffered_amount_low_threshold: int


@typing.runtime_checkable
class _OnBufferedLowJsProto(typing.Protocol):
    """Protocol for JS-style WebRTC data channels exposing `onbufferedamountlow`."""

    onbufferedamountlow: BackendArg


@typing.runtime_checkable
class _OnBufferedLowPyProto(typing.Protocol):
    """Protocol for Python-style WebRTC data channels exposing `on_buffered_amount_low`."""

    on_buffered_amount_low: BackendArg


def _as_tensor_seq(val: BackendArg) -> list[BackendArg]:
    """Convert `val` to a list of tensor chunks.

    Args:
        val (BackendArg): Split tensor chunks sequence.

    Returns:
        list[BackendArg]: List of tensor chunks.
    """
    return list(coerce_operand_list(val))


class WebRTCDataChannelStream:
    """Stream manager wrapping WebRTC RTCDataChannel event-driven communication."""

    channel: BackendArg
    chunk_size: int
    high_watermark: int
    low_watermark: int
    _recv_queue: asyncio.Queue[bytes]
    _drain_event: asyncio.Event

    def __init__(
        self,
        channel: BackendArg,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        high_watermark: int = HIGH_WATERMARK_BYTES,
        low_watermark: int = LOW_WATERMARK_BYTES,
    ) -> None:
        """Initialize WebRTCDataChannelStream.

        Args:
            channel (BackendArg): Underlying WebRTC RTCDataChannel or compatible mock.
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
        cand_msg: BackendArg | _OnMessageJsProto | _OnMessagePyProto = self.channel
        if isinstance(cand_msg, _OnMessageJsProto):
            cand_msg.onmessage = self._on_message
        elif isinstance(cand_msg, _OnMessagePyProto):
            cand_msg.on_message = self._on_message

        cand_low: BackendArg | _BufferedLowJsProto | _BufferedLowPyProto = self.channel
        if isinstance(cand_low, _BufferedLowJsProto):
            cand_low.bufferedAmountLowThreshold = self.low_watermark
        elif isinstance(cand_low, _BufferedLowPyProto):
            cand_low.buffered_amount_low_threshold = self.low_watermark

        cand_cb: BackendArg | _OnBufferedLowJsProto | _OnBufferedLowPyProto = self.channel
        if isinstance(cand_cb, _OnBufferedLowJsProto):
            cand_cb.onbufferedamountlow = self._on_buffered_amount_low
        elif isinstance(cand_cb, _OnBufferedLowPyProto):
            cand_cb.on_buffered_amount_low = self._on_buffered_amount_low

    def _on_message(self, event_or_data: BackendArg) -> None:
        """Event listener for incoming RTCDataChannel binary and text messages.

        Args:
            event_or_data (BackendArg): RTCDataChannel message event or raw byte payload.
        """
        has_data, extracted = read_static_attr(event_or_data, "data")
        payload: BackendArg = extracted if has_data else event_or_data
        has_tb, tb_fn = read_static_attr(payload, "tobytes")
        if isinstance(payload, bytes):
            self._recv_queue.put_nowait(payload)
        elif isinstance(payload, str):
            self._recv_queue.put_nowait(payload.encode("utf-8"))
        elif has_tb and tb_fn is not None:
            from ml_switcheroo_compiler.backends.eager_registry import coerce_bytes

            self._recv_queue.put_nowait(coerce_bytes(tb_fn()))
        else:
            from ml_switcheroo_compiler.backends.eager_registry import coerce_bytes

            self._recv_queue.put_nowait(coerce_bytes(payload))

    def _on_buffered_amount_low(self, event: BackendArg | None = None) -> None:
        """Event listener for RTCDataChannel bufferedamountlow event.

        Args:
            event (BackendArg | None): Low watermark drain event.
        """
        self._drain_event.set()

    def get_buffered_amount(self) -> int:
        """Retrieve the number of queued bytes in the send buffer.

        Returns:
            int: Number of queued bytes.
        """
        has_js, js_amt = read_static_attr(self.channel, "bufferedAmount")
        has_py, py_amt = read_static_attr(self.channel, "buffered_amount")
        amt = js_amt if has_js else (py_amt if has_py else 0)
        if callable(amt):
            return int(coerce_int(amt()))
        return int(coerce_int(amt))

    async def wait_backpressure(self) -> None:
        """Pause execution when bufferedAmount exceeds high watermark until drained."""
        if self.get_buffered_amount() > self.high_watermark:
            self._drain_event.clear()
            has_drain, drain_fn = read_static_attr(self.channel, "wait_drain")
            if has_drain and callable(drain_fn):
                drain_call = drain_fn
                if asyncio.iscoroutinefunction(drain_fn):
                    await drain_call()
                else:
                    drain_call()
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
        has_send, raw_send = read_static_attr(self.channel, "send")
        has_send_msg, raw_send_msg = read_static_attr(self.channel, "sendMessage")
        if has_send:
            send_fn = raw_send
            if asyncio.iscoroutinefunction(raw_send):
                await require_callable(send_fn)(data)
            else:
                res = require_callable(send_fn)(data)
                if inspect.isawaitable(res):
                    await res
        elif has_send_msg:
            send_fn = raw_send_msg
            if asyncio.iscoroutinefunction(raw_send_msg):
                await require_callable(send_fn)(data)
            else:
                res = require_callable(send_fn)(data)
                if inspect.isawaitable(res):
                    await res

    async def recv_bytes(self) -> bytes:
        """Receive the next chunk from the incoming queue or channel.

        Returns:
            bytes: Received chunk payload.
        """
        if not self._recv_queue.empty():
            return await self._recv_queue.get()
        has_recv, recv_fn = read_static_attr(self.channel, "recv")
        if has_recv and callable(recv_fn):
            recv_call = recv_fn
            if asyncio.iscoroutinefunction(recv_fn):
                res = await recv_call()
            else:
                res = recv_call()
                if inspect.isawaitable(res):
                    res = await res
            from ml_switcheroo_compiler.backends.eager_registry import coerce_bytes

            return coerce_bytes(res)
        return await self._recv_queue.get()


async def _send_data(
    conn: BackendArg | WebRTCDataChannelStream, data: BackendArg, chunk_size: int = DEFAULT_CHUNK_SIZE
) -> None:
    """Send tensor over WebRTC DataChannel using chunked binary streaming protocol.

    Args:
        conn (BackendArg | WebRTCDataChannelStream): WebRTC DataChannel, stream adapter, or async connection.
        data (BackendArg): Tensor to serialize and stream.
        chunk_size (int): Size of individual binary chunks in bytes.
    """
    from ml_switcheroo_compiler.backends.eager_registry import coerce_bytes, coerce_shape
    from ml_switcheroo_compiler.backends.protocols import adapt_backend_module

    stream = conn if isinstance(conn, WebRTCDataChannelStream) else WebRTCDataChannelStream(conn, chunk_size=chunk_size)
    arr_adp = adapt_backend_module(data)
    dtype_obj = arr_adp.require_symbol("dtype")
    has_bo, bo_val = read_static_attr(dtype_obj, "byteorder")
    arr_byteorder = str(bo_val) if has_bo else "="

    native_order = "<" if sys.byteorder == "little" else ">"
    byteorder = native_order if arr_byteorder in ("=", "|") else arr_byteorder

    raw_bytes = coerce_bytes(arr_adp.require_op("tobytes")())
    total_bytes = len(raw_bytes)
    num_chunks = max(1, (total_bytes + chunk_size - 1) // chunk_size)

    has_dt_str, dt_str_val = read_static_attr(dtype_obj, "str")
    meta = {
        "protocol": "webrtc_stream_v1",
        "shape": list(coerce_shape(arr_adp.require_symbol("shape"))),
        "dtype": str(dt_str_val if has_dt_str else dtype_obj),
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


async def _recv_data(conn: BackendArg | WebRTCDataChannelStream) -> BackendArg:
    """Receive tensor from WebRTC DataChannel with chunked reassembly and endianness handling.

    Args:
        conn (BackendArg | WebRTCDataChannelStream): WebRTC DataChannel, stream adapter, or async connection.

    Returns:
        BackendArg: Reconstructed numpy ndarray.

    Raises:
        ValueError: If received metadata header is corrupt or payload size mismatches.
    """
    stream = conn if isinstance(conn, WebRTCDataChannelStream) else WebRTCDataChannelStream(conn)

    header_bytes = await stream.recv_bytes()
    if not header_bytes.startswith(b"META:"):
        raise ValueError("Invalid WebRTC stream header: expected META: prefix")

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
                raise ValueError(
                    f"Checksum mismatch on received WebRTC chunk {chunk_idx}: expected {chunk_crc}, got {actual_crc}"
                )
            chunks_map[chunk_idx] = chunk_data
        else:
            chunks_map[len(chunks_map)] = chunk_raw

    chunks: list[bytes] = [chunks_map[i] for i in range(num_chunks)]
    payload = b"".join(chunks)
    if len(payload) != total_bytes:
        raise ValueError(f"Payload length mismatch: expected {total_bytes}, got {len(payload)}")

    from ml_switcheroo_compiler.backends.protocols import adapt_backend_module

    raw_buf = get_active_backend().execute_op("frombuffer", payload, dtype=dtype_str)
    arr = adapt_backend_module(raw_buf).require_op("reshape")(shape)

    native_order = "<" if sys.byteorder == "little" else ">"
    if expected_byteorder in ("<", ">") and expected_byteorder != native_order:
        arr = arr.astype(arr.dtype.newbyteorder("="))

    return adapt_backend_module(arr).require_op("copy")()
