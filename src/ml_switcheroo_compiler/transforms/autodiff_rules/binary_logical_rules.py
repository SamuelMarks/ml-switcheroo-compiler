# ruff: noqa: E402, D100, D103, D104, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, D101, D102, D107, E701, E722, F403, E711, E712, PLR0913, PLR0915
from typing import Any

"""Binary logical AD rules."""

from ml_switcheroo_compiler.transforms.autodiff_rules.jvp_registry import register_jvp
from ml_switcheroo_compiler.transforms.autodiff_rules.vjp_registry import register_vjp


@register_vjp("Greater")
def _greater_vjp(graph, node, out_grad):  # type: ignore  # Justification: Polymorphic / Duck Typing for Framework Agnosticism
    return (None, None)


@register_jvp("Greater")
def _greater_jvp(graph, node, tangents):  # type: ignore  # Justification: Polymorphic / Duck Typing for Framework Agnosticism
    import uuid

    from ml_switcheroo_ir import LogicalNode

    out_id = f"jvp_zeros_{uuid.uuid4().hex[:6]}"
    zeros_node = LogicalNode(
        id=out_id,
        op_type="Constant",
        attributes={"value": 0.0},
        shape_metadata=node.shape_metadata,
    )
    graph.nodes[out_id] = zeros_node
    return out_id
