"""Loop tiling preparation pass for edge optimization heuristics."""

import os
from typing import cast

import yaml

from ml_switcheroo_compiler.backends.registry import get_active_backend
from ml_switcheroo_compiler.ir.core import IRGraph


def _load_heuristics() -> dict[str, object]:
    """Load optimization heuristics.

    Returns:
        dict[str, object]: The loaded heuristics.
    """
    yaml_path = os.path.join(os.path.dirname(__file__), "loop_tiling_heuristics.yaml")
    if os.path.exists(yaml_path):
        with open(yaml_path) as f:
            res = yaml.safe_load(f)
            return cast(dict[str, object], res) if res else {}
    return {}


def _get_tiling_config() -> dict[str, object]:
    """Get the tiling configuration based on active backend.

    Returns:
        dict[str, object]: The tiling configuration.
    """
    heuristics = _load_heuristics()
    if not heuristics:
        return {}

    backend = get_active_backend()
    backend_name = getattr(backend, "__name__", type(backend).__name__).lower() if backend else ""

    profile_name = "default_wgsl" if ("wgsl" in backend_name or "webgpu" in backend_name) else "default_wasm"
    profiles = cast(dict[str, object], heuristics.get("profiles", {}))

    prof = cast(dict[str, object], profiles.get(profile_name, profiles.get("default_wasm", {})))
    return cast(dict[str, object], prof.get("tiling", {}))


def _should_tile(op_type: str, shape: object, op_config: dict[str, object]) -> bool:
    """Determine if a node should be tiled based on shape and thresholds.

    Args:
        op_type (str): The operation type.
        shape (object): The node's shape metadata.
        op_config (dict[str, object]): The operation's configuration.

    Returns:
        bool: True if the node should be tiled.
    """
    if not isinstance(shape, (tuple, list)):
        return False

    if op_type == "matmul" and len(shape) >= 2:
        m = shape[-2] if isinstance(shape[-2], int) else 0
        n = shape[-1] if isinstance(shape[-1], int) else 0
        return bool(m >= cast(int, op_config.get("threshold_M", 0)) or n >= cast(int, op_config.get("threshold_N", 0)))

    if op_type == "conv2d" and len(shape) >= 4:
        h = shape[1] if isinstance(shape[1], int) else 0
        w = shape[2] if isinstance(shape[2], int) else 0
        return bool((h * w) >= cast(int, op_config.get("threshold_HW", 0)))

    return False


def _split_shape(op_type: str, shape: tuple, op_config: dict[str, object]) -> tuple:
    """Split multi-dimensional shape into outer/inner tile chunks.

    Args:
        op_type (str): Operation type.
        shape (tuple): The original shape.
        op_config (dict): Tiling config.

    Returns:
        tuple: The tiled shape.
    """
    if op_type == "matmul":
        # matmul is typically (..., M, N)
        # We split M into (M // TILE_M, TILE_M) and N into (N // TILE_N, TILE_N)
        tile_m = cast(int, op_config.get("TILE_M", 1))
        tile_n = cast(int, op_config.get("TILE_N", 1))
        m = shape[-2]
        n = shape[-1]
        prefix = shape[:-2]
        if isinstance(m, int) and isinstance(n, int):
            m_outer = (m + tile_m - 1) // tile_m
            n_outer = (n + tile_n - 1) // tile_n
            return (*prefix, m_outer, tile_m, n_outer, tile_n)
        return shape

    if op_type == "conv2d":
        # conv2d is (N, H, W, C)
        tile_h = cast(int, op_config.get("TILE_H", 1))
        tile_w = cast(int, op_config.get("TILE_W", 1))
        n = shape[0]
        h = shape[1]
        w = shape[2]
        c = shape[3]
        if isinstance(h, int) and isinstance(w, int):
            h_outer = (h + tile_h - 1) // tile_h
            w_outer = (w + tile_w - 1) // tile_w
            return (n, h_outer, tile_h, w_outer, tile_w, c)
        return shape

    return shape


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

        op_config = cast(dict[str, object], tiling_config.get(op_type, {}))
        shape = getattr(node, "shape_metadata", None)

        if not op_config or not shape:
            continue

        if _should_tile(op_type, shape, op_config):
            if not hasattr(node, "attributes"):
                node.attributes = {}

            # Perform actual IR mutation (splitting the shape for tile chunks)
            new_shape = _split_shape(op_type, tuple(shape), op_config)
            if new_shape != shape:
                node.shape_metadata = new_shape
                node.attributes["tiling"] = True
                if op_type == "matmul":
                    node.attributes["tile_m"] = op_config.get("TILE_M", 1)
                    node.attributes["tile_n"] = op_config.get("TILE_N", 1)
                    node.attributes["tile_k"] = op_config.get("TILE_K", 1)
                elif op_type == "conv2d":
                    node.attributes["tile_h"] = op_config.get("TILE_H", 1)
                    node.attributes["tile_w"] = op_config.get("TILE_W", 1)
                modified = True

    return modified
