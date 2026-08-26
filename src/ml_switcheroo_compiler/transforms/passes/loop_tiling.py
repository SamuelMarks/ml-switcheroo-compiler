"""Loop tiling preparation pass for edge optimization heuristics."""

import os

import yaml

from ml_switcheroo_compiler.backends.registry import get_active_backend
from ml_switcheroo_compiler.ir.core import IRGraph


def _load_heuristics():
    """Load optimization heuristics.

    Returns:
        dict[str, object]: The loaded heuristics.
    """
    yaml_path = os.path.join(os.path.dirname(__file__), "../../backends/edge/optimization_heuristics.yaml")
    if os.path.exists(yaml_path):
        with open(yaml_path) as f:
            from typing import cast

            return cast(dict[str, object], yaml.safe_load(f))
    return {}


def _get_tiling_config():
    """Get the tiling configuration based on active backend.

    Returns:
        dict[str, object]: The tiling configuration.
    """
    heuristics = _load_heuristics()
    if not heuristics:
        return {}

    backend = get_active_backend()
    backend_name = getattr(backend, "__name__", type(backend).__name__).lower()

    profile_name = "default_wgsl" if ("wgsl" in backend_name or "webgpu" in backend_name) else "default_wasm"
    profiles = heuristics.get("profiles", {})
    from typing import cast

    return cast(dict[str, object], profiles.get(profile_name, profiles.get("default_wasm", {})).get("tiling", {}))


def _should_tile(op_type: str, shape, op_config) -> bool:
    """Determine if a node should be tiled based on shape and thresholds.

    Args:
        op_type (str): The operation type.
        shape (object): The node's shape metadata.
        op_config (dict[str, object]): The operation's configuration.

    Returns:
        bool: True if the node should be tiled.
    """
    if op_type == "matmul" and len(shape) >= 2:
        m = shape[-2] if isinstance(shape[-2], int) else 0
        n = shape[-1] if isinstance(shape[-1], int) else 0
        return bool(m >= op_config.get("threshold_M", 0) or n >= op_config.get("threshold_N", 0))

    if op_type == "conv2d" and len(shape) >= 4:
        h = shape[1] if isinstance(shape[1], int) else 0
        w = shape[2] if isinstance(shape[2], int) else 0
        return bool((h * w) >= op_config.get("threshold_HW", 0))

    return False


def loop_tiling_pass(graph: IRGraph) -> bool:
    """Pad shapes or add attributes for loop tiling based on heuristics.

    Args:
        graph (IRGraph): The IR graph.

    Returns:
        bool: True if modified.
    """
    tiling_config = _get_tiling_config()
    if not tiling_config:
        return False

    modified = False
    for node in list(graph.nodes.values()):
        op_type = getattr(node, "op_type", "").lower()
        if op_type not in ("matmul", "conv2d"):
            continue

        op_config = tiling_config.get(op_type, {})
        shape = getattr(node, "shape_metadata", None)

        if not op_config or not shape:
            continue

        if _should_tile(op_type, shape, op_config):
            if not hasattr(node, "attributes"):
                node.attributes = {}

            if not node.attributes.get("tiling"):
                node.attributes["tiling"] = True
                node.attributes["tile_m"] = op_config.get("TILE_M", 1)
                node.attributes["tile_n"] = op_config.get("TILE_N", 1)
                node.attributes["tile_k"] = op_config.get("TILE_K", 1)
                modified = True

    return modified
