"""Autodiff rules for custom and platform-specific kernel operations."""

from __future__ import annotations

from ml_switcheroo_compiler.transforms.autodiff_rules._custom_rules_cf_jvp import (
    _assoc_scan_jvp as _assoc_scan_jvp,
)
from ml_switcheroo_compiler.transforms.autodiff_rules._custom_rules_cf_jvp import (
    _cond_jvp as _cond_jvp,
)
from ml_switcheroo_compiler.transforms.autodiff_rules._custom_rules_cf_jvp import (
    _if_jvp as _if_jvp,
)
from ml_switcheroo_compiler.transforms.autodiff_rules._custom_rules_cf_jvp import (
    _loop_jvp as _loop_jvp,
)
from ml_switcheroo_compiler.transforms.autodiff_rules._custom_rules_cf_jvp import (
    _scan_jvp as _scan_jvp,
)
from ml_switcheroo_compiler.transforms.autodiff_rules._custom_rules_cf_jvp import (
    _while_loop_jvp as _while_loop_jvp,
)
from ml_switcheroo_compiler.transforms.autodiff_rules._custom_rules_cf_vjp import (
    _assoc_scan_vjp as _assoc_scan_vjp,
)
from ml_switcheroo_compiler.transforms.autodiff_rules._custom_rules_cf_vjp import (
    _cond_vjp as _cond_vjp,
)
from ml_switcheroo_compiler.transforms.autodiff_rules._custom_rules_cf_vjp import (
    _if_vjp as _if_vjp,
)
from ml_switcheroo_compiler.transforms.autodiff_rules._custom_rules_cf_vjp import (
    _loop_vjp as _loop_vjp,
)
from ml_switcheroo_compiler.transforms.autodiff_rules._custom_rules_cf_vjp import (
    _scan_vjp as _scan_vjp,
)
from ml_switcheroo_compiler.transforms.autodiff_rules._custom_rules_cf_vjp import (
    _while_loop_vjp as _while_loop_vjp,
)
from ml_switcheroo_compiler.transforms.autodiff_rules._custom_rules_cf_vjp import (
    checkpoint_vjp as checkpoint_vjp,
)
from ml_switcheroo_compiler.transforms.autodiff_rules._custom_rules_common import (
    _coerce_node_attrs as _coerce_node_attrs,
)
from ml_switcheroo_compiler.transforms.autodiff_rules._custom_rules_common import (
    _extract_graph_nodes as _extract_graph_nodes,
)
from ml_switcheroo_compiler.transforms.autodiff_rules._custom_rules_common import (
    _inline_grad_subgraph as _inline_grad_subgraph,
)
from ml_switcheroo_compiler.transforms.autodiff_rules._custom_rules_common import (
    _inline_subgraph as _inline_subgraph,
)
from ml_switcheroo_compiler.transforms.autodiff_rules._custom_rules_common import (
    _is_subgraph_branch as _is_subgraph_branch,
)
from ml_switcheroo_compiler.transforms.autodiff_rules._custom_rules_common import (
    _is_subgraph_with_nodes as _is_subgraph_with_nodes,
)
from ml_switcheroo_compiler.transforms.autodiff_rules._custom_rules_common import (
    _make_node_attrs as _make_node_attrs,
)
from ml_switcheroo_compiler.transforms.autodiff_rules._custom_rules_common import (
    _node_shape_meta as _node_shape_meta,
)
from ml_switcheroo_compiler.transforms.autodiff_rules._custom_rules_custom_ad import (
    custom_jvp_jvp as custom_jvp_jvp,
)
from ml_switcheroo_compiler.transforms.autodiff_rules._custom_rules_custom_ad import (
    custom_jvp_vjp as custom_jvp_vjp,
)
from ml_switcheroo_compiler.transforms.autodiff_rules._custom_rules_custom_ad import (
    custom_vjp_vjp as custom_vjp_vjp,
)
from ml_switcheroo_compiler.transforms.autodiff_rules._custom_rules_custom_ad import (
    recompute_vjp as recompute_vjp,
)
from ml_switcheroo_compiler.transforms.autodiff_rules._custom_rules_indexing_loss import (
    getitem_vjp as getitem_vjp,
)
from ml_switcheroo_compiler.transforms.autodiff_rules._custom_rules_indexing_loss import (
    index_put_jvp as index_put_jvp,
)
from ml_switcheroo_compiler.transforms.autodiff_rules._custom_rules_indexing_loss import (
    log_poisson_loss_jvp as log_poisson_loss_jvp,
)
from ml_switcheroo_compiler.transforms.autodiff_rules._custom_rules_indexing_loss import (
    log_poisson_loss_vjp as log_poisson_loss_vjp,
)
from ml_switcheroo_compiler.transforms.autodiff_rules._custom_rules_indexing_loss import (
    zeros_like_jvp as zeros_like_jvp,
)

__all__ = [
    "_assoc_scan_jvp",
    "_assoc_scan_vjp",
    "_coerce_node_attrs",
    "_cond_jvp",
    "_cond_vjp",
    "_extract_graph_nodes",
    "_if_jvp",
    "_if_vjp",
    "_inline_grad_subgraph",
    "_inline_subgraph",
    "_is_subgraph_branch",
    "_is_subgraph_with_nodes",
    "_loop_jvp",
    "_loop_vjp",
    "_make_node_attrs",
    "_node_shape_meta",
    "_scan_jvp",
    "_scan_vjp",
    "_while_loop_jvp",
    "_while_loop_vjp",
    "checkpoint_vjp",
    "custom_jvp_jvp",
    "custom_jvp_vjp",
    "custom_vjp_vjp",
    "getitem_vjp",
    "index_put_jvp",
    "log_poisson_loss_jvp",
    "log_poisson_loss_vjp",
    "recompute_vjp",
    "zeros_like_jvp",
]
