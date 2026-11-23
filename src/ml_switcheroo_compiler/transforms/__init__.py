"""Transform and Pass Manager package."""

from __future__ import annotations

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
from ml_switcheroo_compiler.transforms.typeclasses import (
    LOGICAL_GRAPH_EQ,
    LOGICAL_GRAPH_PRETTY,
    LOGICAL_NODE_EQ,
    LOGICAL_NODE_REWRITE,
    SCHEMA_PROOF_PASS,
    SHARDING_PROOF_PASS,
    TypedCompilerPass,
    build_grounding_proof_pass,
    compose_typed_passes,
    fixpoint_endo_pass,
    fold_graph_monoid,
    prove_full_graph_pipeline,
    rewrite_graph_recursive,
)

__all__ = [
    "DAGTopologicalSorter",
    "IRValidator",
    "LOGICAL_GRAPH_EQ",
    "LOGICAL_GRAPH_PRETTY",
    "LOGICAL_NODE_EQ",
    "LOGICAL_NODE_REWRITE",
    "PassManager",
    "SCHEMA_PROOF_PASS",
    "SHARDING_PROOF_PASS",
    "TypedCompilerPass",
    "build_grounding_proof_pass",
    "common_subexpression_elimination_pass",
    "compose_typed_passes",
    "constant_folding_pass",
    "dead_code_elimination_pass",
    "fixpoint_endo_pass",
    "fold_graph_monoid",
    "grad",
    "hvp",
    "jvp",
    "lift_state_pass",
    "prove_full_graph_pipeline",
    "rewrite_graph_recursive",
    "shape_inference_pass",
]
