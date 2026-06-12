"""Transforms and Pass Manager package."""

from ml_switcheroo.transforms.pass_manager import (
    PassManager,
    DAGTopologicalSorter,
    IRValidator,
)
from ml_switcheroo.transforms.passes import (
    dce_pass,
    cse_pass,
    constant_folding_pass,
    shape_inference_pass,
    lift_state_pass,
)

__all__ = [
    "PassManager",
    "DAGTopologicalSorter",
    "IRValidator",
    "dce_pass",
    "cse_pass",
    "constant_folding_pass",
    "shape_inference_pass",
    "lift_state_pass",
]
