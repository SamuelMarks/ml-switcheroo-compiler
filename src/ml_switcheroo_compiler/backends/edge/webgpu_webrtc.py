"""WebRTC Javascript string template emitters for Edge Backends."""

import json
import os

import yaml

from ml_switcheroo_compiler.backends.edge.config_models import WebrtcCollectivesConfig, WebrtcTopologyConfig


def emit_webrtc_init() -> str:
    """Emit WebRTC initialization JavaScript string."""
    yaml_path: object = os.path.join(os.path.dirname(__file__), "../../distributed/webrtc_topology.yaml")
    if not os.path.exists(yaml_path):
        return ""

    with open(yaml_path) as f:
        data: object = WebrtcTopologyConfig(**yaml.safe_load(f)).model_dump()

    handlers_yaml_path: object = os.path.join(os.path.dirname(__file__), "webrtc_collectives.yaml")
    if os.path.exists(handlers_yaml_path):
        with open(handlers_yaml_path) as f:
            handlers_data: object = WebrtcCollectivesConfig(**yaml.safe_load(f)).model_dump()
            handlers: object = handlers_data.get("handlers", {})
    else:
        handlers: object = {}

    config: object = json.dumps(data.get("webrtc_config", {}))
    templates: object = data.get("templates", {})
    init_tpl: object = templates.get("init_peer_connection", "")

    return str(init_tpl.format(config=config, allreduce_handler=handlers.get("allreduce_handler", ""), allgather_handler=handlers.get("allgather_handler", ""), alltoall_handler=handlers.get("alltoall_handler", ""), reducescatter_handler=handlers.get("reducescatter_handler", "")))


def emit_webrtc_op(op_type: str, local_tensor_var: str, op_id: str) -> str:
    """Emit WebRTC collective operation JavaScript string."""
    yaml_path: object = os.path.join(os.path.dirname(__file__), "../../distributed/webrtc_topology.yaml")
    if not os.path.exists(yaml_path):
        return ""

    with open(yaml_path) as f:
        data: object = WebrtcTopologyConfig(**yaml.safe_load(f)).model_dump()

    handlers_yaml_path: object = os.path.join(os.path.dirname(__file__), "webrtc_collectives.yaml")
    chunk_size: object = 65536
    if os.path.exists(handlers_yaml_path):
        with open(handlers_yaml_path) as f:
            handlers_data: object = WebrtcCollectivesConfig(**yaml.safe_load(f)).model_dump()
            chunk_size: object = handlers_data.get("schema_def", {}).get("chunk_size_bytes", 65536)

    templates: object = data.get("templates", {})
    tpl: object = ""
    if op_type == "AllReduce":
        tpl: object = str(templates.get("allreduce_emit", ""))
    elif op_type == "AllGather":
        tpl: object = str(templates.get("allgather_emit", ""))
    elif op_type == "AllToAll":
        tpl: object = str(templates.get("alltoall_emit", ""))
    elif op_type == "ReduceScatter":
        tpl: object = str(templates.get("reducescatter_emit", ""))
    else:
        return ""

    # Ensure signaling logic complies with binary payload chunking schema
    return str(tpl.format(local_tensor_data=local_tensor_var, op_id=op_id, chunk_size=chunk_size))
