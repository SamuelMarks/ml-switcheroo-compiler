"""Modular WebRTC Signaling Server and Protocol Contract for Edge Distributed Execution."""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import parse_qs, urlparse

from pydantic import BaseModel

# Shared in-memory store for WebRTC offers, answers, and ICE candidates
_offers: dict[str, str] = {}
_answers: dict[str, str] = {}
_candidates: dict[str, list[str]] = {}


class SignalingConfig(BaseModel):
    """Configuration contract for WebRTC signaling server.

    Attributes:
        host (str): Hostname or IP address to bind.
        port (int): Port number to listen on.
        websocket_path (str): Endpoint path for WebSocket signaling upgrade.
        cors_origin (str): Allowed CORS origin header value.
    """

    host: str = "0.0.0.0"
    port: int = 8080
    websocket_path: str = "/ws"
    cors_origin: str = "*"


class SignalingStore:
    """Thread-safe state container for WebRTC peer signaling session records."""

    def __init__(self) -> None:
        """Initialize empty signaling store."""
        self._lock: threading.Lock = threading.Lock()
        self.offers: dict[str, str] = _offers
        self.answers: dict[str, str] = _answers
        self.candidates: dict[str, list[str]] = _candidates

    def clear(self) -> None:
        """Clear all active session states."""
        with self._lock:
            self.offers.clear()
            self.answers.clear()
            self.candidates.clear()

    def set_offer(self, peer_id: str, offer: str) -> None:
        """Store session description offer for a peer.

        Args:
            peer_id (str): Unique peer identifier.
            offer (str): Serialized SDP offer string.
        """
        with self._lock:
            self.offers[peer_id] = offer

    def get_offer(self, peer_id: str) -> str | None:
        """Retrieve SDP offer for a peer.

        Args:
            peer_id (str): Unique peer identifier.

        Returns:
            Optional[str]: SDP offer string or None.
        """
        with self._lock:
            return self.offers.get(peer_id)

    def set_answer(self, peer_id: str, answer: str) -> None:
        """Store session description answer for a peer.

        Args:
            peer_id (str): Unique peer identifier.
            answer (str): Serialized SDP answer string.
        """
        with self._lock:
            self.answers[peer_id] = answer

    def get_answer(self, peer_id: str) -> str | None:
        """Retrieve SDP answer for a peer.

        Args:
            peer_id (str): Unique peer identifier.

        Returns:
            Optional[str]: SDP answer string or None.
        """
        with self._lock:
            return self.answers.get(peer_id)

    def add_candidate(self, peer_id: str, candidate: str) -> None:
        """Register an ICE candidate for a peer.

        Args:
            peer_id (str): Unique peer identifier.
            candidate (str): ICE candidate payload.
        """
        with self._lock:
            if peer_id not in self.candidates:
                self.candidates[peer_id] = []
            self.candidates[peer_id].append(candidate)

    def get_candidates(self, peer_id: str) -> list[str]:
        """Retrieve all accumulated ICE candidates for a peer.

        Args:
            peer_id (str): Unique peer identifier.

        Returns:
            list[str]: List of ICE candidate payloads.
        """
        with self._lock:
            return list(self.candidates.get(peer_id, []))


_GLOBAL_STORE = SignalingStore()


class SignalingHandler(BaseHTTPRequestHandler):
    """Modular HTTP/WebSocket signaling request handler for WebRTC peers."""

    def _send_json(self, status: int, data: object) -> None:
        """Send a JSON formatted HTTP response.

        Args:
            status (int): HTTP status code.
            data (object): Serializable JSON payload.
        """
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Upgrade, Connection")
        self.end_headers()
        self.wfile.write(json.dumps(data).encode("utf-8"))

    def do_OPTIONS(self) -> None:
        """Handle CORS preflight requests."""
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Upgrade, Connection")
        self.end_headers()

    def do_POST(self) -> None:
        """Process POST requests for SDP offers, answers, and ICE candidates."""
        content_length = int(self.headers.get("Content-Length", 0))
        post_data = self.rfile.read(content_length)
        try:
            data = json.loads(post_data.decode("utf-8"))
        except Exception:
            self._send_json(400, {"error": "Invalid JSON"})
            return

        path = urlparse(self.path).path
        peer_id = data.get("peer_id")
        if not peer_id and path in ("/offer", "/answer", "/candidate"):
            self._send_json(400, {"error": "Missing peer_id"})
            return

        if path == "/offer":
            _GLOBAL_STORE.set_offer(str(peer_id), data.get("offer", ""))
            self._send_json(200, {"status": "ok"})
        elif path == "/answer":
            _GLOBAL_STORE.set_answer(str(peer_id), data.get("answer", ""))
            self._send_json(200, {"status": "ok"})
        elif path == "/candidate":
            _GLOBAL_STORE.add_candidate(str(peer_id), data.get("candidate", ""))
            self._send_json(200, {"status": "ok"})
        elif path == "/reset":
            _GLOBAL_STORE.clear()
            self._send_json(200, {"status": "reset"})
        else:
            self._send_json(404, {"error": "Not Found"})

    def do_GET(self) -> None:
        """Process GET requests for SDP offers, answers, and ICE candidates."""
        parsed = urlparse(self.path)
        path = parsed.path
        qs = parse_qs(parsed.query)
        peer_id = qs.get("peer_id", [None])[0]

        if path == "/health":
            self._send_json(200, {"status": "healthy"})
            return

        if path == "/ws":
            # WebSocket handshake upgrade response
            self.send_response(101, "Switching Protocols")
            self.send_header("Upgrade", "websocket")
            self.send_header("Connection", "Upgrade")
            self.end_headers()
            return

        if not peer_id:
            self._send_json(400, {"error": "Missing peer_id"})
            return

        if path == "/offer":
            offer = _GLOBAL_STORE.get_offer(peer_id)
            self._send_json(200, {"offer": offer} if offer is not None else {})
        elif path == "/answer":
            answer = _GLOBAL_STORE.get_answer(peer_id)
            self._send_json(200, {"answer": answer} if answer is not None else {})
        elif path == "/candidate":
            candidates = _GLOBAL_STORE.get_candidates(peer_id)
            self._send_json(200, {"candidates": candidates})
        else:
            self._send_json(404, {"error": "Not Found"})


class SignalingServer:
    """Modular WebRTC Signaling Server supporting customizable host and port bindings."""

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 8080,
        config: SignalingConfig | None = None,
    ) -> None:
        """Initialize SignalingServer instance.

        Args:
            host (str): Binding host IP address or hostname.
            port (int): Port number to bind.
            config (Optional[SignalingConfig]): Optional custom configuration specification.
        """
        self.config: SignalingConfig = config or SignalingConfig(host=host, port=port)
        self.host: str = self.config.host
        self.port: int = self.config.port
        self.server: HTTPServer = HTTPServer((self.host, self.port), SignalingHandler)
        self.thread: threading.Thread | None = None

    def get_url(self) -> str:
        """Retrieve root HTTP endpoint URL for signaling.

        Returns:
            str: Root signaling server URL.
        """
        return f"http://{self.host}:{self.port}"

    def get_websocket_url(self) -> str:
        """Retrieve WebSocket endpoint URL for signaling.

        Returns:
            str: WebSocket signaling URL.
        """
        return f"ws://{self.host}:{self.port}{self.config.websocket_path}"

    def start(self) -> None:
        """Start the background signaling server daemon thread."""
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def stop(self) -> None:
        """Shutdown the signaling server and release socket resources."""
        if self.thread:
            self.server.shutdown()
            self.thread.join(timeout=1.0)
        self.server.server_close()


__all__ = [
    "SignalingConfig",
    "SignalingHandler",
    "SignalingServer",
    "SignalingStore",
    "_answers",
    "_candidates",
    "_offers",
]
