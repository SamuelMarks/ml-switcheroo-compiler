"""Module test_common_autodiff.py."""

from __future__ import annotations

import typing

if typing.TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import EagerCallableProtocol


def test_autodiff_common() -> None:
    """test_autodiff_common."""
    from ml_switcheroo_compiler.transforms.autodiff_rules.common import UnconnectedGradients, make_zero_jvp, make_zero_vjp

    class DummyNode:
        """DummyNode."""

        inputs = [1, 2]

    vjp_fn = make_zero_vjp("dummy")
    assert typing.cast("EagerCallableProtocol", vjp_fn)(None, typing.cast("EagerCallableProtocol", DummyNode)(), "cot") == (UnconnectedGradients.ZERO, UnconnectedGradients.ZERO)

    jvp_fn = make_zero_jvp("dummy")
    assert typing.cast("EagerCallableProtocol", jvp_fn)(None, None, None) == ""
