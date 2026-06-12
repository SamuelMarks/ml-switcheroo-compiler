"""Compiler passes."""

from ml_switcheroo.transforms.passes.dce import dce_pass
from ml_switcheroo.transforms.passes.cse import cse_pass
from ml_switcheroo.transforms.passes.constant_folding import constant_folding_pass
from ml_switcheroo.transforms.passes.shape_inference import shape_inference_pass
from ml_switcheroo.transforms.passes.lift_state import lift_state_pass

__all__ = [
    "dce_pass",
    "cse_pass",
    "constant_folding_pass",
    "shape_inference_pass",
    "lift_state_pass",
]
