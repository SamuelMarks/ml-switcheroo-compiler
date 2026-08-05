"""Transform and Pass Manager package."""

from ml_switcheroo_compiler.transforms.autodiff import grad, hvp, jvp
from ml_switcheroo_compiler.transforms.pass_manager import (
    DAGTopologicalSorter,
    IRValidator,
    PassManager,
)
from ml_switcheroo_compiler.transforms.passes import (
    common_subexpression_elimination_pass,
    constant_folding_pass,
    dead_code_elimination_pass,
    lift_state_pass,
    shape_inference_pass,
)

__all__ = [
    "DAGTopologicalSorter",
    "IRValidator",
    "PassManager",
    "constant_folding_pass",
    "common_subexpression_elimination_pass",
    "dead_code_elimination_pass",
    "grad",
    "hvp",
    "jvp",
    "lift_state_pass",
    "shape_inference_pass",
]
