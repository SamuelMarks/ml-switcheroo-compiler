"""Tests for distributed WebRTC signaling server and store."""

import json
import time
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from ml_switcheroo_compiler.distributed.webrtc_signaling import SignalingServer, _answers, _candidates, _offers


def test_webrtc_signaling() -> None:
    """Test WebRTC signaling server lifecycle, routes, HTTP methods, and edge cases."""
    _offers.clear()
    _answers.clear()
    _candidates.clear()

    server = SignalingServer(port=18080)
    server.start()

    time.sleep(0.1)  # Wait for server to start

    base_url = "http://127.0.0.1:18080"

    try:
        # Test OPTIONS
        req = Request(base_url, method="OPTIONS")
        with urlopen(req) as resp:
            assert resp.status == 200
            assert resp.headers.get("Access-Control-Allow-Origin") == "*"

        # Test POST /offer
        data = json.dumps({"peer_id": "p1", "offer": "offer1"}).encode("utf-8")
        req = Request(f"{base_url}/offer", data=data, method="POST")
        req.add_header("Content-Length", str(len(data)))
        with urlopen(req) as resp:
            assert resp.status == 200

        # Test POST /answer
        data = json.dumps({"peer_id": "p1", "answer": "answer1"}).encode("utf-8")
        req = Request(f"{base_url}/answer", data=data, method="POST")
        req.add_header("Content-Length", str(len(data)))
        with urlopen(req) as resp:
            assert resp.status == 200

        # Test POST /candidate
        data = json.dumps({"peer_id": "p1", "candidate": "cand1"}).encode("utf-8")
        req = Request(f"{base_url}/candidate", data=data, method="POST")
        req.add_header("Content-Length", str(len(data)))
        with urlopen(req) as resp:
            assert resp.status == 200

        # Test POST second candidate
        data = json.dumps({"peer_id": "p1", "candidate": "cand2"}).encode("utf-8")
        req = Request(f"{base_url}/candidate", data=data, method="POST")
        req.add_header("Content-Length", str(len(data)))
        with urlopen(req) as resp:
            assert resp.status == 200

        # Test GET /offer
        req = Request(f"{base_url}/offer?peer_id=p1")
        with urlopen(req) as resp:
            assert resp.status == 200
            res = json.loads(resp.read().decode())
            assert res["offer"] == "offer1"

        # Test GET /answer
        req = Request(f"{base_url}/answer?peer_id=p1")
        with urlopen(req) as resp:
            assert resp.status == 200
            res = json.loads(resp.read().decode())
            assert res["answer"] == "answer1"

        # Test GET /candidate
        req = Request(f"{base_url}/candidate?peer_id=p1")
        with urlopen(req) as resp:
            assert resp.status == 200
            res = json.loads(resp.read().decode())
            assert res["candidates"] == ["cand1", "cand2"]

        # Test GET without peer_id
        req = Request(f"{base_url}/offer")
        try:
            urlopen(req)
        except HTTPError as e:
            assert e.code == 400

        # Test GET not found path
        req = Request(f"{base_url}/invalid?peer_id=p1")
        try:
            urlopen(req)
        except HTTPError as e:
            assert e.code == 404

        # Test POST not found path
        data = json.dumps({"peer_id": "p1"}).encode("utf-8")
        req = Request(f"{base_url}/invalid", data=data, method="POST")
        req.add_header("Content-Length", str(len(data)))
        try:
            urlopen(req)
        except HTTPError as e:
            assert e.code == 404

        # Test empty GET offer
        req = Request(f"{base_url}/offer?peer_id=p2")
        with urlopen(req) as resp:
            res = json.loads(resp.read().decode())
            assert "offer" not in res

        # Test empty GET answer
        req = Request(f"{base_url}/answer?peer_id=p2")
        with urlopen(req) as resp:
            res = json.loads(resp.read().decode())
            assert "answer" not in res

        # Test /health
        req = Request(f"{base_url}/health")
        with urlopen(req) as resp:
            assert resp.status == 200
            res = json.loads(resp.read().decode())
            assert res["status"] == "healthy"

        # Test /reset
        req = Request(f"{base_url}/reset", data=b"{}", method="POST")
        req.add_header("Content-Length", "2")
        with urlopen(req) as resp:
            assert resp.status == 200

        # Test invalid JSON POST
        req = Request(f"{base_url}/offer", data=b"not-json", method="POST")
        req.add_header("Content-Length", "8")
        try:
            urlopen(req)
        except HTTPError as e:
            assert e.code == 400

        # Test POST without peer_id
        req = Request(f"{base_url}/offer", data=b'{"offer": "no_peer"}', method="POST")
        req.add_header("Content-Length", "20")
        try:
            urlopen(req)
        except HTTPError as e:
            assert e.code == 400
            err_data = json.loads(e.read().decode())
            assert err_data["error"] == "Missing peer_id"

        # Test OPTIONS
        req = Request(f"{base_url}/offer", method="OPTIONS")
        with urlopen(req) as resp:
            assert resp.status == 200

        # Test /ws upgrade path
        req = Request(f"{base_url}/ws")
        try:
            with urlopen(req) as resp:
                assert resp.status in (101, 200)
        except HTTPError as e:
            assert e.code == 101

        # Test URLs
        assert server.get_url() == base_url
        assert "ws://" in server.get_websocket_url()

    finally:
        server.stop()

    # test stop when not started
    from ml_switcheroo_compiler.distributed.webrtc_signaling import SignalingConfig, SignalingStore

    store = SignalingStore()
    store.set_offer("p", "off")
    assert store.get_offer("p") == "off"
    store.set_answer("p", "ans")
    assert store.get_answer("p") == "ans"
    store.add_candidate("p", "cand1")
    assert store.get_candidates("p") == ["cand1"]
    store.clear()
    assert store.get_offer("p") is None

    cfg = SignalingConfig(host="127.0.0.1", port=18081)
    s2 = SignalingServer(config=cfg)
    s2.stop()
