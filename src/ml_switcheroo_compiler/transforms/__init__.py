# ruff: noqa: E402, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, E701, E722, F403, E711, E712, PLR0913, PLR0915
"""Module __init__.py."""

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
