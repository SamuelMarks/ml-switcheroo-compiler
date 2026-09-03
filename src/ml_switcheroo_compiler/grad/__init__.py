# ruff: noqa: F401
"""grad module."""

from ml_switcheroo_compiler.transforms.autodiff_rules.common import UnconnectedGradients

from .api import (
    RegisterGradient,
    backward,
    grad,
    hook_gradient,
    ir_grad,
    overwrite_with_gradient,
    value_and_grad,
)
from .checkpointing import (
    checkpoint,
    recompute_grad,
    remat,
)
from .custom_vjp_ops import (
    CustomVJPFunction,
    custom_vjp,
)
from .jit import (
    disable_jit,
    eval_shape,
    jit,
)
from .jvp_vjp import (
    custom_jvp,
    hessian,
    hvp,
    jacfwd,
    jacrev,
    jvp,
    vjp,
)
from .options import DEFAULT_GRAD_EPSILON, GradCheckOptions, GradOptions, JitOptions
from .testing import (
    check_numerical_grads,
)
from .utils import (
    _check_scalar,
    _compute_grad_and_value,
    _convert_to_tensors,
    _find_wrt_tensors,
    _get_concrete_val,
    _get_fun_primal,
    _get_inputs_dict,
    _to_original_type,
    value_and_grad_wrt_vars,
)

__all__ = [
    "CustomVJPFunction",
    "DEFAULT_GRAD_EPSILON",
    "GradCheckOptions",
    "GradOptions",
    "JitOptions",
    "DEFAULT_GRAD_EPSILON",
    "GradOptions",
    "JitOptions",
    "RegisterGradient",
    "_check_scalar",
    "_compute_grad_and_value",
    "_convert_to_tensors",
    "_find_wrt_tensors",
    "_get_concrete_val",
    "_get_fun_primal",
    "_get_inputs_dict",
    "_to_original_type",
    "backward",
    "check_numerical_grads",
    "checkpoint",
    "custom_jvp",
    "custom_vjp",
    "disable_jit",
    "eval_shape",
    "grad",
    "hessian",
    "hvp",
    "ir_grad",
    "jacfwd",
    "jacrev",
    "jit",
    "jvp",
    "overwrite_with_gradient",
    "hook_gradient",
    "recompute_grad",
    "remat",
    "value_and_grad",
    "value_and_grad_wrt_vars",
    "vjp",
    "UnconnectedGradients",
]
