"""WebRTC distributed communication, signaling, and stream managers for browser edge backends."""

from __future__ import annotations

from ml_switcheroo_compiler.backends.edge.distributed_webrtc.collectives import WebRTCDataChannelStream
from ml_switcheroo_compiler.distributed.webrtc_signaling import (
    SignalingConfig,
    SignalingHandler,
    SignalingServer,
    SignalingStore,
)

__all__ = [
    "SignalingConfig",
    "SignalingHandler",
    "SignalingServer",
    "SignalingStore",
    "WebRTCDataChannelStream",
]
