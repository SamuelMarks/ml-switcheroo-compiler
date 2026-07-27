"""Passes package."""

from ml_switcheroo_compiler.transforms.passes.broadcast_explicitizer import broadcast_explicitizer_pass
from ml_switcheroo_compiler.transforms.passes.constant_folding import constant_folding_pass
from ml_switcheroo_compiler.transforms.passes.cse import cse_pass as common_subexpression_elimination_pass
from ml_switcheroo_compiler.transforms.passes.dce import dce_pass as dead_code_elimination_pass
from ml_switcheroo_compiler.transforms.passes.dtype_inference import dtype_inference_pass
from ml_switcheroo_compiler.transforms.passes.lift_state import lift_state_pass
from ml_switcheroo_compiler.transforms.passes.shape_inference import shape_inference_pass
from ml_switcheroo_compiler.transforms.passes.spmd import inject_spmd_communication_pass
from ml_switcheroo_compiler.transforms.passes.type_promotion_explicitizer import type_promotion_explicitizer_pass

__all__ = [
    "broadcast_explicitizer_pass",
    "common_subexpression_elimination_pass",
    "constant_folding_pass",
    "dead_code_elimination_pass",
    "dtype_inference_pass",
    "inject_spmd_communication_pass",
    "lift_state_pass",
    "shape_inference_pass",
    "type_promotion_explicitizer_pass",
]
