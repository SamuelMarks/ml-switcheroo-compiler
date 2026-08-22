from unittest.mock import mock_open, patch

import yaml

from ml_switcheroo_compiler.backends.edge.webgpu_webrtc import emit_webrtc_init, emit_webrtc_op


def test_webrtc_no_handlers_yaml():
    with patch("os.path.exists") as mock_exists:
        # Simulate topology exists, but handlers yaml does not
        def fake_exists(path):
            return "webrtc_topology.yaml" in path

        mock_exists.side_effect = fake_exists

        topology_data = {
            "webrtc_config": {"ice_servers": [{"urls": "stun:stun.l.google.com:19302"}], "signaling": {"url": "ws://localhost:8080", "timeout_ms": 1000}},
            "templates": {"init_peer_connection": "config: {config}, allreduce: {allreduce_handler}", "allreduce_emit": "allreduce {local_tensor_data} {op_id} {chunk_size}"},
        }

        with patch("builtins.open", mock_open(read_data=yaml.dump(topology_data))):
            init_str = emit_webrtc_init()
            assert "allreduce: " in init_str

            op_str = emit_webrtc_op("AllReduce", "tensor", "op1")
            assert "allreduce tensor op1 65536" in op_str


def test_webrtc_with_handlers_yaml():
    with patch("os.path.exists", return_value=True):
        topology_data = {
            "webrtc_config": {"ice_servers": [{"urls": "stun:stun.l.google.com:19302"}], "signaling": {"url": "ws://localhost:8080", "timeout_ms": 1000}},
            "templates": {
                "init_peer_connection": "config: {config}, allreduce: {allreduce_handler}",
                "allreduce_emit": "allreduce {local_tensor_data} {op_id} {chunk_size}",
                "allgather_emit": "allgather {local_tensor_data} {op_id} {chunk_size}",
                "alltoall_emit": "alltoall {local_tensor_data} {op_id} {chunk_size}",
                "reducescatter_emit": "reducescatter {local_tensor_data} {op_id} {chunk_size}",
            },
        }
        handlers_data = {"handlers": {"allreduce_handler": "my_allreduce_handler"}, "schema_def": {"chunk_size_bytes": 1024, "message_format": "binary"}}

        def mock_open_file(path, *args, **kwargs):
            if "webrtc_topology" in path:
                return mock_open(read_data=yaml.dump(topology_data))()
            elif "webrtc_collectives" in path:
                return mock_open(read_data=yaml.dump(handlers_data))()
            return mock_open(read_data="")()

        with patch("builtins.open", side_effect=mock_open_file):
            init_str = emit_webrtc_init()
            assert "allreduce: my_allreduce_handler" in init_str

            assert "allreduce tensor op1 1024" in emit_webrtc_op("AllReduce", "tensor", "op1")
            assert "allgather tensor op1 1024" in emit_webrtc_op("AllGather", "tensor", "op1")
            assert "alltoall tensor op1 1024" in emit_webrtc_op("AllToAll", "tensor", "op1")
            assert "reducescatter tensor op1 1024" in emit_webrtc_op("ReduceScatter", "tensor", "op1")
            assert emit_webrtc_op("Unknown", "tensor", "op1") == ""
