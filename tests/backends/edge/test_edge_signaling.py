"""Tests for Edge WebRTC HTTP signaling server and handler."""

from __future__ import annotations

import http.client
import json
import typing

from ml_switcheroo_compiler.backends.edge.distributed_webrtc.signaling import SignalingServer

if typing.TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import EagerCallableProtocol, EagerTensorProtocol


def _test_edge_signaling_server_full_workflow_sub_1(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute sub-part 1 of test_edge_signaling_server_full_workflow.

    Args:
        _state: Shared test state dictionary.
    """
    port = _state["port"]
    conn = typing.cast("EagerCallableProtocol", http.client.HTTPConnection)("127.0.0.1", port)

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
    assert typing.cast("EagerCallableProtocol", json.loads)(typing.cast("EagerCallableProtocol", resp.read)().decode("utf-8")) == {"status": "ok"}

    # 3. GET /offer existing and missing
    conn.request("GET", "/offer?peer_id=peer_a")
    resp = conn.getresponse()
    assert resp.status == 200
    assert typing.cast("EagerCallableProtocol", json.loads)(typing.cast("EagerCallableProtocol", resp.read)().decode("utf-8")) == {"offer": "sdp_offer_data"}

    conn.request("GET", "/offer?peer_id=unknown_peer")
    resp = conn.getresponse()
    assert resp.status == 200
    assert typing.cast("EagerCallableProtocol", json.loads)(typing.cast("EagerCallableProtocol", resp.read)().decode("utf-8")) == {}
    getattr(_state, "update")({"conn": conn, "resp": resp})
    _state.update(locals())


def _test_edge_signaling_server_full_workflow_sub_2(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute sub-part 2 of test_edge_signaling_server_full_workflow.

    Args:
        _state: Shared test state dictionary.
    """
    conn = _state["conn"]
    conn.request(
        "POST",
        "/answer",
        json.dumps({"peer_id": "peer_a", "answer": "sdp_answer_data"}),
        headers={"Content-Type": "application/json"},
    )
    resp = conn.getresponse()
    assert resp.status == 200
    assert typing.cast("EagerCallableProtocol", json.loads)(typing.cast("EagerCallableProtocol", resp.read)().decode("utf-8")) == {"status": "ok"}

    # 5. GET /answer existing and missing
    conn.request("GET", "/answer?peer_id=peer_a")
    resp = conn.getresponse()
    assert resp.status == 200
    assert typing.cast("EagerCallableProtocol", json.loads)(typing.cast("EagerCallableProtocol", resp.read)().decode("utf-8")) == {"answer": "sdp_answer_data"}

    conn.request("GET", "/answer?peer_id=unknown_peer")
    resp = conn.getresponse()
    assert resp.status == 200
    assert typing.cast("EagerCallableProtocol", json.loads)(typing.cast("EagerCallableProtocol", resp.read)().decode("utf-8")) == {}

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
    getattr(_state, "update")({"resp": resp})
    _state.update(locals())


def _test_edge_signaling_server_full_workflow_sub_3(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute sub-part 3 of test_edge_signaling_server_full_workflow.

    Args:
        _state: Shared test state dictionary.
    """
    conn, resp = _state["conn"], _state["resp"]
    assert resp.status == 200
    resp.read()

    # 7. GET /candidate existing and missing
    conn.request("GET", "/candidate?peer_id=peer_a")
    resp = conn.getresponse()
    assert resp.status == 200
    assert typing.cast("EagerCallableProtocol", json.loads)(typing.cast("EagerCallableProtocol", resp.read)().decode("utf-8")) == {"candidates": ["cand_1", "cand_2"]}

    conn.request("GET", "/candidate?peer_id=unknown_peer")
    resp = conn.getresponse()
    assert resp.status == 200
    assert typing.cast("EagerCallableProtocol", json.loads)(typing.cast("EagerCallableProtocol", resp.read)().decode("utf-8")) == {"candidates": []}

    # 8. POST /unknown -> 404
    conn.request(
        "POST",
        "/unknown_path",
        json.dumps({"peer_id": "peer_a"}),
        headers={"Content-Type": "application/json"},
    )
    resp = conn.getresponse()
    assert resp.status == 404
    assert typing.cast("EagerCallableProtocol", json.loads)(typing.cast("EagerCallableProtocol", resp.read)().decode("utf-8")) == {"error": "Not Found"}

    # 9. GET /unknown -> 404
    conn.request("GET", "/unknown_path?peer_id=peer_a")
    resp = conn.getresponse()
    assert resp.status == 404
    assert typing.cast("EagerCallableProtocol", json.loads)(typing.cast("EagerCallableProtocol", resp.read)().decode("utf-8")) == {"error": "Not Found"}
    getattr(_state, "update")({"resp": resp})
    _state.update(locals())


def _test_edge_signaling_server_full_workflow_sub_4(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute sub-part 4 of test_edge_signaling_server_full_workflow.

    Args:
        _state: Shared test state dictionary.
    """
    conn = _state["conn"]
    conn.request("GET", "/offer")
    resp = conn.getresponse()
    assert resp.status == 400
    assert typing.cast("EagerCallableProtocol", json.loads)(typing.cast("EagerCallableProtocol", resp.read)().decode("utf-8")) == {"error": "Missing peer_id"}

    conn.close()
    getattr(_state, "update")({"resp": resp})
    _state.update(locals())


def test_edge_signaling_server_full_workflow() -> None:
    """Verify complete HTTP signaling workflow: OPTIONS, offer, answer, candidate, and 404/400 errors."""
    server = SignalingServer(host="127.0.0.1", port=0)
    port = server.server.server_port
    server.start()

    try:
        _state: dict[str, EagerTensorProtocol] = dict(locals())
        getattr(_state, "update")({"port": port, "server": server})
        for _sub_fn in (
            _test_edge_signaling_server_full_workflow_sub_1,
            _test_edge_signaling_server_full_workflow_sub_2,
            _test_edge_signaling_server_full_workflow_sub_3,
            _test_edge_signaling_server_full_workflow_sub_4,
        ):
            _sub_fn(_state)
        assert len(_state) > 0
    finally:
        server.stop()

    # Verify stop when server thread was not started
    unstarted = SignalingServer(host="127.0.0.1", port=0)
    unstarted.stop()
