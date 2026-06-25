"""Compiler passes."""

from ml_switcheroo_compiler.transforms.passes.constant_folding import constant_folding_pass
from ml_switcheroo_compiler.transforms.passes.cse import cse_pass
from ml_switcheroo_compiler.transforms.passes.dce import dce_pass
from ml_switcheroo_compiler.transforms.passes.lift_state import lift_state_pass
from ml_switcheroo_compiler.transforms.passes.shape_inference import shape_inference_pass
from ml_switcheroo_compiler.transforms.passes.spmd import inject_spmd_communication_pass

__all__ = [
    "constant_folding_pass",
    "cse_pass",
    "dce_pass",
    "inject_spmd_communication_pass",
    "lift_state_pass",
    "shape_inference_pass",
]
