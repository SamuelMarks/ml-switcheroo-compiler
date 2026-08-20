# ruff: noqa: E402, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, E701, E722, F403, E711, E712, PLR0913, PLR0915
"""Module __init__.py."""

from typing import Any

"""Passes package."""

from ml_switcheroo_compiler.transforms.passes.axis_translation import axis_translation_pass
from ml_switcheroo_compiler.transforms.passes.batch_norm_folding import batch_norm_folding_pass
from ml_switcheroo_compiler.transforms.passes.broadcast_explicitizer import broadcast_explicitizer_pass
from ml_switcheroo_compiler.transforms.passes.buffer_allocation import buffer_allocation_pass
from ml_switcheroo_compiler.transforms.passes.constant_folding import constant_folding_pass
from ml_switcheroo_compiler.transforms.passes.cse import cse_pass as common_subexpression_elimination_pass
from ml_switcheroo_compiler.transforms.passes.dce import dce_pass as dead_code_elimination_pass
from ml_switcheroo_compiler.transforms.passes.dtype_inference import dtype_inference_pass
from ml_switcheroo_compiler.transforms.passes.graph_scheduling import graph_scheduling_pass
from ml_switcheroo_compiler.transforms.passes.lift_state import lift_state_pass
from ml_switcheroo_compiler.transforms.passes.loop_tiling import loop_tiling_pass
from ml_switcheroo_compiler.transforms.passes.loop_unrolling import loop_unrolling_pass
from ml_switcheroo_compiler.transforms.passes.mixed_precision import mixed_precision_pass
from ml_switcheroo_compiler.transforms.passes.rematerialization import rematerialization_pass
from ml_switcheroo_compiler.transforms.passes.shape_inference import shape_inference_pass
from ml_switcheroo_compiler.transforms.passes.spmd import inject_spmd_communication_pass
from ml_switcheroo_compiler.transforms.passes.state_lowering import state_lowering_pass
from ml_switcheroo_compiler.transforms.passes.type_promotion_explicitizer import type_promotion_explicitizer_pass

__all__ = [
    "broadcast_explicitizer_pass",
    "common_subexpression_elimination_pass",
    "constant_folding_pass",
    "dead_code_elimination_pass",
    "dtype_inference_pass",
    "inject_spmd_communication_pass",
    "lift_state_pass",
    "loop_tiling_pass",
    "shape_inference_pass",
    "rematerialization_pass",
    "type_promotion_explicitizer_pass",
    "axis_translation_pass",
    "batch_norm_folding_pass",
    "buffer_allocation_pass",
    "graph_scheduling_pass",
    "loop_unrolling_pass",
    "mixed_precision_pass",
    "state_lowering_pass",
]
from .poly_lower import polyfill_lowering_pass
