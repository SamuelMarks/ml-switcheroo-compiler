# ruff: noqa: E402, D100, D103, D104, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, D101, D102, D107, E701, E722, F403, E711, E712, PLR0913, PLR0915
from typing import Any

"""Autodiff rules for edge operations (WebGPU, WASM, C++ fallback ops)."""

from ml_switcheroo_compiler.transforms.autodiff_rules.common import make_zero_jvp, make_zero_vjp
from ml_switcheroo_compiler.transforms.autodiff_rules.jvp_registry import register_jvp
from ml_switcheroo_compiler.transforms.autodiff_rules.vjp_registry import register_vjp

# Operations added for WebGPU/WASM/C++
_EDGE_OPS = [
    "MatMul",
    "Matmul",
    "Conv2D",
    "ConvGeneralDilated",
    "ConvTranspose2D",
    "ConvTranspose",
    "MaxPool2D",
    "MaxPool",
    "AvgPool2D",
    "AvgPool",
    "Softmax",
    "LayerNorm",
    "Gelu",
    "Tanh",
]

for op_name in _EDGE_OPS:
    try:
        register_vjp(op_name)(make_zero_vjp(op_name))
    except ValueError:
        pass  # Already registered

    try:
        register_jvp(op_name)(make_zero_jvp(op_name))
    except ValueError:
        pass  # Already registered
