import json
import time
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from ml_switcheroo_compiler.distributed.webrtc_signaling import SignalingServer, _answers, _candidates, _offers


def test_webrtc_signaling():
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

    finally:
        server.stop()

    # test stop when not started
    s2 = SignalingServer()
    s2.stop()
