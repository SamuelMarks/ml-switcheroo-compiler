"""Tests for Edge WebRTC HTTP signaling server and handler."""

import http.client
import json

from ml_switcheroo_compiler.backends.edge.distributed_webrtc.signaling import SignalingServer


def test_edge_signaling_server_full_workflow() -> None:
    """Verify complete HTTP signaling workflow: OPTIONS, offer, answer, candidate, and 404/400 errors."""
    server = SignalingServer(host="127.0.0.1", port=0)
    port = server.server.server_port
    server.start()

    try:
        conn = http.client.HTTPConnection("127.0.0.1", port)

        # 1. OPTIONS preflight
        conn.request("OPTIONS", "/")
        resp = conn.getresponse()
        assert resp.status == 200
        assert resp.getheader("Access-Control-Allow-Origin") == "*"
        resp.read()

        # 2. POST /offer
        conn.request(
            "POST",
            "/offer",
            json.dumps({"peer_id": "peer_a", "offer": "sdp_offer_data"}),
            headers={"Content-Type": "application/json"},
        )
        resp = conn.getresponse()
        assert resp.status == 200
        assert json.loads(resp.read().decode("utf-8")) == {"status": "ok"}

        # 3. GET /offer existing and missing
        conn.request("GET", "/offer?peer_id=peer_a")
        resp = conn.getresponse()
        assert resp.status == 200
        assert json.loads(resp.read().decode("utf-8")) == {"offer": "sdp_offer_data"}

        conn.request("GET", "/offer?peer_id=unknown_peer")
        resp = conn.getresponse()
        assert resp.status == 200
        assert json.loads(resp.read().decode("utf-8")) == {}

        # 4. POST /answer
        conn.request(
            "POST",
            "/answer",
            json.dumps({"peer_id": "peer_a", "answer": "sdp_answer_data"}),
            headers={"Content-Type": "application/json"},
        )
        resp = conn.getresponse()
        assert resp.status == 200
        assert json.loads(resp.read().decode("utf-8")) == {"status": "ok"}

        # 5. GET /answer existing and missing
        conn.request("GET", "/answer?peer_id=peer_a")
        resp = conn.getresponse()
        assert resp.status == 200
        assert json.loads(resp.read().decode("utf-8")) == {"answer": "sdp_answer_data"}

        conn.request("GET", "/answer?peer_id=unknown_peer")
        resp = conn.getresponse()
        assert resp.status == 200
        assert json.loads(resp.read().decode("utf-8")) == {}

        # 6. POST /candidate (first time: creates list; second time: appends)
        conn.request(
            "POST",
            "/candidate",
            json.dumps({"peer_id": "peer_a", "candidate": "cand_1"}),
            headers={"Content-Type": "application/json"},
        )
        resp = conn.getresponse()
        assert resp.status == 200
        resp.read()

        conn.request(
            "POST",
            "/candidate",
            json.dumps({"peer_id": "peer_a", "candidate": "cand_2"}),
            headers={"Content-Type": "application/json"},
        )
        resp = conn.getresponse()
        assert resp.status == 200
        resp.read()

        # 7. GET /candidate existing and missing
        conn.request("GET", "/candidate?peer_id=peer_a")
        resp = conn.getresponse()
        assert resp.status == 200
        assert json.loads(resp.read().decode("utf-8")) == {"candidates": ["cand_1", "cand_2"]}

        conn.request("GET", "/candidate?peer_id=unknown_peer")
        resp = conn.getresponse()
        assert resp.status == 200
        assert json.loads(resp.read().decode("utf-8")) == {"candidates": []}

        # 8. POST /unknown -> 404
        conn.request(
            "POST",
            "/unknown_path",
            json.dumps({"peer_id": "peer_a"}),
            headers={"Content-Type": "application/json"},
        )
        resp = conn.getresponse()
        assert resp.status == 404
        assert json.loads(resp.read().decode("utf-8")) == {"error": "Not Found"}

        # 9. GET /unknown -> 404
        conn.request("GET", "/unknown_path?peer_id=peer_a")
        resp = conn.getresponse()
        assert resp.status == 404
        assert json.loads(resp.read().decode("utf-8")) == {"error": "Not Found"}

        # 10. GET without peer_id -> 400
        conn.request("GET", "/offer")
        resp = conn.getresponse()
        assert resp.status == 400
        assert json.loads(resp.read().decode("utf-8")) == {"error": "Missing peer_id"}

        conn.close()
    finally:
        server.stop()

    # Verify stop when server thread was not started
    unstarted = SignalingServer(host="127.0.0.1", port=0)
    unstarted.stop()
