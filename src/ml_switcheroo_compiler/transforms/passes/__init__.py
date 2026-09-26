# ruff: noqa: E402, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, E701, E722, F403, E711, E712, PLR0913, PLR0915
"""Passes package exposing middle-end graph transformation passes."""

from ml_switcheroo_compiler.transforms.passes.axis_translation import axis_translation_pass
from ml_switcheroo_compiler.transforms.passes.batch_norm_folding import batch_norm_folding_pass
from ml_switcheroo_compiler.transforms.passes.broadcast_explicitizer import broadcast_explicitizer_pass
from ml_switcheroo_compiler.transforms.passes.buffer_allocation import (
    BufferAllocationPass,
    buffer_allocation_pass,
)
from ml_switcheroo_compiler.transforms.passes.constant_folding import constant_folding_pass
from ml_switcheroo_compiler.transforms.passes.control_flow_opt import control_flow_optimization_pass
from ml_switcheroo_compiler.transforms.passes.cse import cse_pass as common_subexpression_elimination_pass
from ml_switcheroo_compiler.transforms.passes.dce import dce_pass as dead_code_elimination_pass
from ml_switcheroo_compiler.transforms.passes.dtype_inference import dtype_inference_pass
from ml_switcheroo_compiler.transforms.passes.graph_scheduling import (
    GraphSchedulingPass,
    graph_scheduling_pass,
)
from ml_switcheroo_compiler.transforms.passes.lift_state import (
    lift_module_state,
    lift_state,
    lift_state_pass,
)
from ml_switcheroo_compiler.transforms.passes.loop_tiling import loop_tiling_pass
from ml_switcheroo_compiler.transforms.passes.loop_unrolling import loop_unrolling_pass
from ml_switcheroo_compiler.transforms.passes.mixed_precision import mixed_precision_pass
from ml_switcheroo_compiler.transforms.passes.operator_fusion import (
    HorizontalFusionPass,
    VerticalFusionPass,
    apply_operator_fusion,
    operator_fusion_pass,
)
from ml_switcheroo_compiler.transforms.passes.parallel_scan import parallel_scan_pass
from ml_switcheroo_compiler.transforms.passes.rematerialization import rematerialization_pass
from ml_switcheroo_compiler.transforms.passes.shape_inference import shape_inference_pass
from ml_switcheroo_compiler.transforms.passes.spmd import (
    inject_spmd_communication_pass,
    propagate_sharding,
    spmd_partitioning_pass,
)
from ml_switcheroo_compiler.transforms.passes.state_lowering import (
    StateLoweringPass,
    state_lowering_pass,
)
from ml_switcheroo_compiler.transforms.passes.strip_offline_nodes import (
    OFFLINE_DIAGNOSTIC_OPS,
    export_graph_to_dot_string,
    strip_offline_diagnostic_nodes_pass,
)
from ml_switcheroo_compiler.transforms.passes.type_promotion_explicitizer import type_promotion_explicitizer_pass
from ml_switcheroo_compiler.transforms.passes.vectorization import vectorization_pass

__all__ = [
    "BufferAllocationPass",
    "GraphSchedulingPass",
    "HorizontalFusionPass",
    "VerticalFusionPass",
    "apply_operator_fusion",
    "axis_translation_pass",
    "batch_norm_folding_pass",
    "broadcast_explicitizer_pass",
    "buffer_allocation_pass",
    "common_subexpression_elimination_pass",
    "constant_folding_pass",
    "control_flow_optimization_pass",
    "dead_code_elimination_pass",
    "dtype_inference_pass",
    "graph_scheduling_pass",
    "inject_spmd_communication_pass",
    "lift_module_state",
    "lift_state",
    "lift_state_pass",
    "loop_tiling_pass",
    "loop_unrolling_pass",
    "mixed_precision_pass",
    "operator_fusion_pass",
    "parallel_scan_pass",
    "propagate_sharding",
    "rematerialization_pass",
    "shape_inference_pass",
    "spmd_partitioning_pass",
    "StateLoweringPass",
    "state_lowering_pass",
    "strip_offline_diagnostic_nodes_pass",
    "type_promotion_explicitizer_pass",
    "vectorization_pass",
]
from .poly_lower import polyfill_lowering_pass
