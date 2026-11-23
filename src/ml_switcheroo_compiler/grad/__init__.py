"""Public package exports and initialization for `ml_switcheroo_compiler.grad`."""

from __future__ import annotations

from ml_switcheroo_compiler.transforms.autodiff_rules.common import UnconnectedGradients

from .api import (
    RegisterGradient,
    backward,
    grad,
    hook_gradient,
    hvp_graph,
    ir_grad,
    nth_order_grad,
    nth_order_grad_graph,
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
    CustomJVPFunction,  # noqa: F401
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
    check_jvp_vjp_duality,
    check_numerical_grads,
)
from .typeclasses import (
    SCALAR_TANGENT_EQ,
    SCALAR_TANGENT_MONOID,
    DifferentiableOpDictionary,
    GraphCotangentElem,
    ScalarTangentElem,
    TangentKind,
    build_graph_cotangent_monoid,
    is_symbolic_zero_node,
    verify_jvp_vjp_adjoint_law,
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
    "DifferentiableOpDictionary",
    "GradCheckOptions",
    "GradOptions",
    "GraphCotangentElem",
    "JitOptions",
    "RegisterGradient",
    "SCALAR_TANGENT_EQ",
    "SCALAR_TANGENT_MONOID",
    "ScalarTangentElem",
    "TangentKind",
    "UnconnectedGradients",
    "_check_scalar",
    "_compute_grad_and_value",
    "_convert_to_tensors",
    "_find_wrt_tensors",
    "_get_concrete_val",
    "_get_fun_primal",
    "_get_inputs_dict",
    "_to_original_type",
    "backward",
    "build_graph_cotangent_monoid",
    "check_jvp_vjp_duality",
    "check_numerical_grads",
    "checkpoint",
    "custom_jvp",
    "custom_vjp",
    "disable_jit",
    "eval_shape",
    "grad",
    "hessian",
    "hook_gradient",
    "hvp",
    "hvp_graph",
    "ir_grad",
    "is_symbolic_zero_node",
    "jacfwd",
    "jacrev",
    "jit",
    "jvp",
    "nth_order_grad",
    "nth_order_grad_graph",
    "overwrite_with_gradient",
    "recompute_grad",
    "remat",
    "value_and_grad",
    "value_and_grad_wrt_vars",
    "verify_jvp_vjp_adjoint_law",
    "vjp",
]
