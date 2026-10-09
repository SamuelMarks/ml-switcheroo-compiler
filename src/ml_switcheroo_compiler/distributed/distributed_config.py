# ruff: noqa: E402, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, E701, E722, F403, E711, E712, PLR0913, PLR0915
"""Distributed configuration classes."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class DistributedConfig:
    """Configuration for distributed execution.

    Attributes:
        world_size (int): Total number of processes in the distributed group.
        rank (int): Global rank of the current process.
        local_rank (int): Local rank of the current process on this node.
        backend (str): The distributed backend protocol (e.g., "nccl", "gloo", "mpi", "webrtc").
        init_method (str | None): Initialization method URI (e.g., "env://", "tcp://...").
        timeout_sec (float): Timeout for distributed operations in seconds.
        topology_type (str | None): Optional specified topology from topologies YAML.
        options (dict[str, Any]): Additional backend-specific configuration options.
    """

    world_size: int = 1
    rank: int = 0
    local_rank: int = 0
    backend: str = "nccl"
    init_method: str | None = "env://"
    timeout_sec: float = 1800.0  # 30 minutes default
    topology_type: str | None = None
    options: dict[str, Any] = field(default_factory=dict)
