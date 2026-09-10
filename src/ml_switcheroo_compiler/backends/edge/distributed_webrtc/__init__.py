"""WebRTC distributed communication, signaling, and stream managers for browser edge backends."""

from ml_switcheroo_compiler.backends.edge.distributed_webrtc.collectives import WebRTCDataChannelStream
from ml_switcheroo_compiler.backends.edge.distributed_webrtc.signaling import SignalingHandler, SignalingServer

__all__ = [
    "SignalingHandler",
    "SignalingServer",
    "WebRTCDataChannelStream",
]
