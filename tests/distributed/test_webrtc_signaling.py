"""Tests for WebRTC signaling server."""

import json
import time
from urllib.request import Request, urlopen

from ml_switcheroo_compiler.distributed.webrtc_signaling import SignalingServer, _answers, _candidates, _offers


def test_signaling_server():
    _offers.clear()
    _answers.clear()
    _candidates.clear()

    server = SignalingServer(port=8081)
    server.start()
    time.sleep(0.1)  # Wait for start

    base_url = "http://127.0.0.1:8081"

    try:
        # Test OPTIONS
        req = Request(f"{base_url}/offer", method="OPTIONS")
        with urlopen(req) as response:
            assert response.status == 200

        # Test POST offer
        data = json.dumps({"peer_id": "peer1", "offer": "sdp_offer_data"}).encode("utf-8")
        req = Request(f"{base_url}/offer", data=data, method="POST")
        req.add_header("Content-Type", "application/json")
        with urlopen(req) as response:
            assert response.status == 200

        # Test GET offer
        req = Request(f"{base_url}/offer?peer_id=peer1")
        with urlopen(req) as response:
            res_data = json.loads(response.read().decode("utf-8"))
            assert res_data["offer"] == "sdp_offer_data"

        # Test GET missing offer
        req = Request(f"{base_url}/offer?peer_id=peer2")
        with urlopen(req) as response:
            res_data = json.loads(response.read().decode("utf-8"))
            assert res_data == {}

        # Test POST answer
        data = json.dumps({"peer_id": "peer1", "answer": "sdp_answer_data"}).encode("utf-8")
        req = Request(f"{base_url}/answer", data=data, method="POST")
        req.add_header("Content-Type", "application/json")
        with urlopen(req) as response:
            assert response.status == 200

        # Test GET answer
        req = Request(f"{base_url}/answer?peer_id=peer1")
        with urlopen(req) as response:
            res_data = json.loads(response.read().decode("utf-8"))
            assert res_data["answer"] == "sdp_answer_data"

        # Test GET missing answer
        req = Request(f"{base_url}/answer?peer_id=peer2")
        with urlopen(req) as response:
            res_data = json.loads(response.read().decode("utf-8"))
            assert res_data == {}

        # Test POST candidate
        data = json.dumps({"peer_id": "peer1", "candidate": "ice_candidate_data"}).encode("utf-8")
        req = Request(f"{base_url}/candidate", data=data, method="POST")
        req.add_header("Content-Type", "application/json")
        with urlopen(req) as response:
            assert response.status == 200

        # Test POST candidate again for the False branch
        data = json.dumps({"peer_id": "peer1", "candidate": "ice_candidate_data_2"}).encode("utf-8")
        req = Request(f"{base_url}/candidate", data=data, method="POST")
        req.add_header("Content-Type", "application/json")
        with urlopen(req) as response:
            assert response.status == 200

        # Test GET candidate
        req = Request(f"{base_url}/candidate?peer_id=peer1")
        with urlopen(req) as response:
            res_data = json.loads(response.read().decode("utf-8"))
            assert res_data["candidates"] == ["ice_candidate_data", "ice_candidate_data_2"]

        # Test GET missing candidate
        req = Request(f"{base_url}/candidate?peer_id=peer2")
        with urlopen(req) as response:
            res_data = json.loads(response.read().decode("utf-8"))
            assert res_data["candidates"] == []

        # Test missing peer_id on GET
        import urllib.error

        req = Request(f"{base_url}/candidate")
        try:
            urlopen(req)
        except urllib.error.HTTPError as e:
            assert e.code == 400

        # Test 404 POST
        req = Request(f"{base_url}/unknown", data=b"{}", method="POST")
        try:
            urlopen(req)
        except urllib.error.HTTPError as e:
            assert e.code == 404

        # Test 404 GET
        req = Request(f"{base_url}/unknown?peer_id=peer1", method="GET")
        try:
            urlopen(req)
        except urllib.error.HTTPError as e:
            assert e.code == 404

    finally:
        server.stop()


def test_signaling_server_stop_without_start():
    server = SignalingServer(port=8082)
    server.stop()
