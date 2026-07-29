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
