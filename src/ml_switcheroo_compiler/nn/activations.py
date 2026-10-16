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
