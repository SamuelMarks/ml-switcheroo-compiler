# ruff: noqa: E402, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, E701, E722, F403, E711, E712, PLR0913, PLR0915
"""Module __init__.py."""

from typing import Any

"""Neural network modules and layers.

Per the ML ecosystem architecture (see ARCHITECTURE.md), high-level neural network
modules, layer state management, and framework-specific API mimicry must NOT exist in this
repository. All API routing belongs strictly in the frontend `zero-*` repositories.

This `nn/` directory exists solely to provide abstract representations of initialization
routines (`initializers.py`) and explicit empty facades (`activations.py`) to enforce
the "No API Shell" rule. Actual mathematical computation for activations or layer
forward passes is defined in `src/ml_switcheroo_compiler/ops/`.
"""

__all__ = []
