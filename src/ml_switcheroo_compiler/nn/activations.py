# ruff: noqa: E402, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, E701, E722, F403, E711, E712, PLR0913, PLR0915
"""Module activations.py."""

"""Activations facade.

Per the ML ecosystem architecture (see ARCHITECTURE.md and the "No API Shell" Rule),
high-level neural network modules, stateful layers, and framework-specific API routing
belong strictly in the frontend `zero-*` repositories (e.g., `zero-torch`, `zero-jax`).

The `ml-switcheroo-compiler` is exclusively responsible for the mathematical operations
and IR emission. Therefore, standard activation representations exist as core mathematical
operations in `src/ml_switcheroo_compiler/ops/` rather than stateful NN modules here.

This file serves as an empty export facade to prevent accidental inclusion of
framework-specific syntactic sugar.
"""

__all__ = []
