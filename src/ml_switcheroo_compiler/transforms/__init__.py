"""Transforms and Pass Manager package."""

from ml_switcheroo_compiler.transforms.pass_manager import (
    DAGTopologicalSorter,
    IRValidator,
    PassManager,
)
from ml_switcheroo_compiler.transforms.passes import (
    constant_folding_pass,
    cse_pass,
    dce_pass,
    lift_state_pass,
    shape_inference_pass,
)

__all__ = [
    "DAGTopologicalSorter",
    "IRValidator",
    "PassManager",
    "constant_folding_pass",
    "cse_pass",
    "dce_pass",
    "lift_state_pass",
    "shape_inference_pass",
]
