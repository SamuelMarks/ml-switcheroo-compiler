"""Neural network initialization routines.

Per the ML ecosystem architecture (see ARCHITECTURE.md and QA_PROMPT.md Rule 1),
high-level neural network modules, layer state management, and framework-specific API
mimicry must NOT exist in this repository. All API routing belongs strictly in the
frontend `zero-*` repositories.

This package exports stateless mathematical parameter initializers (`initializers.py`).
Core mathematical activation functions are defined in `src/ml_switcheroo_compiler/ops/nn/activations.py`.
"""

from __future__ import annotations

from ml_switcheroo_compiler.nn.initializers import (
    InitializerFn,
    constant,
    delta_orthogonal,
    glorot,
    glorot_normal,
    glorot_uniform,
    he_normal,
    he_uniform,
    kaiming_normal,
    kaiming_uniform,
    lecun_normal,
    lecun_uniform,
    normal,
    ones,
    orthogonal,
    truncated_normal,
    uniform,
    variance_scaling,
    xavier_normal,
    xavier_uniform,
    zeros,
)

__all__ = [
    "InitializerFn",
    "constant",
    "delta_orthogonal",
    "glorot",
    "glorot_normal",
    "glorot_uniform",
    "he_normal",
    "he_uniform",
    "kaiming_normal",
    "kaiming_uniform",
    "lecun_normal",
    "lecun_uniform",
    "normal",
    "ones",
    "orthogonal",
    "truncated_normal",
    "uniform",
    "variance_scaling",
    "xavier_normal",
    "xavier_uniform",
    "zeros",
]
