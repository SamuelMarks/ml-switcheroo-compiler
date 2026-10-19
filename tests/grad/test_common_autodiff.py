"""Module test_common_autodiff.py."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import EagerCallableProtocol


def test_autodiff_common() -> None:
    """test_autodiff_common."""
    from ml_switcheroo_compiler.transforms.autodiff_rules.common import UnconnectedGradients, make_zero_jvp, make_zero_vjp

    class DummyNode:
        """DummyNode."""

        inputs = [1, 2]

    vjp_fn = make_zero_vjp("dummy")
    assert cast("EagerCallableProtocol", vjp_fn)(None, cast("EagerCallableProtocol", DummyNode)(), "cot") == (UnconnectedGradients.ZERO, UnconnectedGradients.ZERO)

    jvp_fn = make_zero_jvp("dummy")
    assert cast("EagerCallableProtocol", jvp_fn)(None, None, None) == ""
