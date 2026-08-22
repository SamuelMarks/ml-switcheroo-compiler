"""WebRTC Signaling Server using standard library http.server for in-browser peer-to-peer."""

import json
import threading
import typing
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import parse_qs, urlparse

# In-memory store for WebRTC offers, answers, and ICE candidates
_offers: dict[str, str] = {}
_answers: dict[str, str] = {}
_candidates: dict[str, list[str]] = {}


class SignalingHandler(BaseHTTPRequestHandler):
    """Simple HTTP signaling handler for WebRTC."""

    def _send_json(self, status: int, data: dict) -> None:
        """Send a JSON response."""
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(json.dumps(data).encode("utf-8"))

    def do_OPTIONS(self) -> None:
        """Handle CORS preflight."""
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_POST(self) -> None:
        """Handle POST requests for offers, answers, and candidates."""
        content_length = int(self.headers.get("Content-Length", 0))
        post_data = self.rfile.read(content_length)
        data = json.loads(post_data.decode("utf-8"))

        path = urlparse(self.path).path
        peer_id = data.get("peer_id")

        if path == "/offer":
            _offers[peer_id] = data["offer"]
            self._send_json(200, {"status": "ok"})
        elif path == "/answer":
            _answers[peer_id] = data["answer"]
            self._send_json(200, {"status": "ok"})
        elif path == "/candidate":
            if peer_id not in _candidates:
                _candidates[peer_id] = []
            _candidates[peer_id].append(data["candidate"])
            self._send_json(200, {"status": "ok"})
        else:
            self._send_json(404, {"error": "Not Found"})

    def do_GET(self) -> None:
        """Handle GET requests to retrieve offers, answers, and candidates."""
        parsed = urlparse(self.path)
        path = parsed.path
        qs = parse_qs(parsed.query)
        peer_id = qs.get("peer_id", [None])[0]

        if not peer_id:
            self._send_json(400, {"error": "Missing peer_id"})
            return

        if path == "/offer":
            offer = _offers.get(peer_id)
            self._send_json(200, {"offer": offer} if offer else {})
        elif path == "/answer":
            answer = _answers.get(peer_id)
            self._send_json(200, {"answer": answer} if answer else {})
        elif path == "/candidate":
            candidates = _candidates.get(peer_id, [])
            self._send_json(200, {"candidates": candidates})
        else:
            self._send_json(404, {"error": "Not Found"})


class SignalingServer:
    """WebRTC Signaling Server orchestrator."""

    def __init__(self, host: str = "127.0.0.1", port: int = 8080) -> None:
        """Initialize the signaling server."""
        self.host = host
        self.port = port
        self.server = HTTPServer((self.host, self.port), SignalingHandler)
        self.thread: typing.Optional[threading.Thread] = None

    def start(self) -> None:
        """Start the signaling server in a background thread."""
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def stop(self) -> None:
        """Stop the signaling server."""
        if self.thread:
            self.server.shutdown()
            self.thread.join()
        self.server.server_close()
