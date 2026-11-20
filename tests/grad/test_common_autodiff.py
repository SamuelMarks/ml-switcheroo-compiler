"""Module test_common_autodiff.py."""

from __future__ import annotations

import typing

from tests.testing_utils import MockNodeStub

if typing.TYPE_CHECKING:
    pass


def test_autodiff_common() -> None:
    """test_autodiff_common."""
    from ml_switcheroo_ir import LogicalGraph, LogicalNode

    from ml_switcheroo_compiler.transforms.autodiff_rules.common import (
        UnconnectedGradients,
        make_zero_jvp,
        make_zero_vjp,
    )

    class DummyNode(MockNodeStub):
        """DummyNode."""

        def __init__(self) -> None:
            """Initialize stub attributes."""
            super().__init__()
            self.inputs = ["1", "2"]

    g = LogicalGraph("dummy_g")
    vjp_fn = make_zero_vjp("dummy")
    assert vjp_fn(g, DummyNode(), "cot") == (
        UnconnectedGradients.ZERO,
        UnconnectedGradients.ZERO,
    )

    jvp_fn = make_zero_jvp("dummy")
    assert jvp_fn(g, LogicalNode(id="n", op_type="dummy"), ["t0"]) == ""
