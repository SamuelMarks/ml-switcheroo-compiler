"""Reverse-mode and Forward-mode Automatic Differentiation (AD) Engine."""

from __future__ import annotations

from ml_switcheroo_compiler.core.errors import MissingJVPRuleError as MissingJVPRuleError
from ml_switcheroo_compiler.transforms._autodiff_grad_api import (
    _extract_gradients as _extract_gradients,
)
from ml_switcheroo_compiler.transforms._autodiff_grad_api import (
    _init_adjoints as _init_adjoints,
)
from ml_switcheroo_compiler.transforms._autodiff_grad_api import (
    _maybe_vectorize_graph as _maybe_vectorize_graph,
)
from ml_switcheroo_compiler.transforms._autodiff_grad_api import (
    grad as grad,
)
from ml_switcheroo_compiler.transforms._autodiff_jvp_api import (
    _forward_pass_jvp as _forward_pass_jvp,
)
from ml_switcheroo_compiler.transforms._autodiff_jvp_api import (
    hvp as hvp,
)
from ml_switcheroo_compiler.transforms._autodiff_jvp_api import (
    jvp as jvp,
)
from ml_switcheroo_compiler.transforms._autodiff_jvp_core import (
    JvpRuleCallable as JvpRuleCallable,
)
from ml_switcheroo_compiler.transforms._autodiff_jvp_core import (
    _compile_jvp_expr as _compile_jvp_expr,
)
from ml_switcheroo_compiler.transforms._autodiff_jvp_core import (
    _get_input_tangents as _get_input_tangents,
)
from ml_switcheroo_compiler.transforms._autodiff_jvp_core import (
    _invoke_jvp_rule as _invoke_jvp_rule,
)
from ml_switcheroo_compiler.transforms._autodiff_jvp_core import (
    _invoke_style2_jvp_rule as _invoke_style2_jvp_rule,
)
from ml_switcheroo_compiler.transforms._autodiff_jvp_core import (
    _process_jvp_node as _process_jvp_node,
)
from ml_switcheroo_compiler.transforms._autodiff_nn_vjp_rules import (
    _avg_pool2d_vjp as _avg_pool2d_vjp,
)
from ml_switcheroo_compiler.transforms._autodiff_nn_vjp_rules import (
    _batch_norm_vjp as _batch_norm_vjp,
)
from ml_switcheroo_compiler.transforms._autodiff_nn_vjp_rules import (
    _conv2d_vjp_rule as _conv2d_vjp_rule,
)
from ml_switcheroo_compiler.transforms._autodiff_nn_vjp_rules import (
    _group_norm_vjp as _group_norm_vjp,
)
from ml_switcheroo_compiler.transforms._autodiff_nn_vjp_rules import (
    _layer_norm_vjp as _layer_norm_vjp,
)
from ml_switcheroo_compiler.transforms._autodiff_nn_vjp_rules import (
    _max_pool2d_with_argmax_vjp as _max_pool2d_with_argmax_vjp,
)
from ml_switcheroo_compiler.transforms._autodiff_nn_vjp_rules import (
    _node_shape_meta as _node_shape_meta,
)
from ml_switcheroo_compiler.transforms._autodiff_nn_vjp_rules import (
    _rms_norm_vjp as _rms_norm_vjp,
)
from ml_switcheroo_compiler.transforms._autodiff_nn_vjp_rules import (
    conv2d_vjp_input as conv2d_vjp_input,
)
from ml_switcheroo_compiler.transforms._autodiff_nn_vjp_rules import (
    conv2d_vjp_weight as conv2d_vjp_weight,
)
from ml_switcheroo_compiler.transforms._autodiff_vjp_core import (
    _accumulate_gradients as _accumulate_gradients,
)
from ml_switcheroo_compiler.transforms._autodiff_vjp_core import (
    _add_nodes as _add_nodes,
)
from ml_switcheroo_compiler.transforms._autodiff_vjp_core import (
    _backward_pass as _backward_pass,
)
from ml_switcheroo_compiler.transforms._autodiff_vjp_core import (
    _copy_graph as _copy_graph,
)
from ml_switcheroo_compiler.transforms._autodiff_vjp_core import (
    _get_reachable_from_output as _get_reachable_from_output,
)
from ml_switcheroo_compiler.transforms._autodiff_vjp_core import (
    _is_zero_node as _is_zero_node,
)
from ml_switcheroo_compiler.transforms._autodiff_vjp_core import (
    _load_rematerialization_rules as _load_rematerialization_rules,
)
from ml_switcheroo_compiler.transforms._autodiff_vjp_core import (
    _recompute_subgraph as _recompute_subgraph,
)
from ml_switcheroo_compiler.transforms._autodiff_vjp_core import (
    _should_rematerialize as _should_rematerialize,
)
from ml_switcheroo_compiler.transforms.autodiff_rules.common import UnconnectedGradients as UnconnectedGradients
from ml_switcheroo_compiler.transforms.autodiff_rules.jvp_registry import JVPRuleCallable as JVPRuleCallable
from ml_switcheroo_compiler.transforms.autodiff_rules.vjp_registry import (
    _VJP_REGISTRY as _VJP_REGISTRY,
)
from ml_switcheroo_compiler.transforms.autodiff_rules.vjp_registry import (
    get_vjp as get_vjp,
)

__all__ = [
    "JVPRuleCallable",
    "JvpRuleCallable",
    "MissingJVPRuleError",
    "UnconnectedGradients",
    "_VJP_REGISTRY",
    "_accumulate_gradients",
    "_add_nodes",
    "_avg_pool2d_vjp",
    "_backward_pass",
    "_batch_norm_vjp",
    "_compile_jvp_expr",
    "_conv2d_vjp_rule",
    "_copy_graph",
    "_extract_gradients",
    "_forward_pass_jvp",
    "_get_input_tangents",
    "_get_reachable_from_output",
    "_group_norm_vjp",
    "_init_adjoints",
    "_invoke_jvp_rule",
    "_invoke_style2_jvp_rule",
    "_is_zero_node",
    "_layer_norm_vjp",
    "_load_rematerialization_rules",
    "_max_pool2d_with_argmax_vjp",
    "_maybe_vectorize_graph",
    "_node_shape_meta",
    "_process_jvp_node",
    "_recompute_subgraph",
    "_rms_norm_vjp",
    "_should_rematerialize",
    "conv2d_vjp_input",
    "conv2d_vjp_weight",
    "get_vjp",
    "grad",
    "hvp",
    "jvp",
]
