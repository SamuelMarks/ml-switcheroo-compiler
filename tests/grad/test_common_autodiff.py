"""Module test_common_autodiff.py."""

from __future__ import annotations

import typing

from tests.testing_utils import ensure_callable

if typing.TYPE_CHECKING:
    pass


def test_autodiff_common() -> None:
    """test_autodiff_common."""
    from ml_switcheroo_compiler.transforms.autodiff_rules.common import UnconnectedGradients, make_zero_jvp, make_zero_vjp

    class DummyNode:
        """DummyNode."""

        inputs = [1, 2]

    vjp_fn = make_zero_vjp("dummy")
    assert ensure_callable(vjp_fn)(None, ensure_callable(DummyNode)(), "cot") == (UnconnectedGradients.ZERO, UnconnectedGradients.ZERO)

    jvp_fn = make_zero_jvp("dummy")
    assert ensure_callable(jvp_fn)(None, None, None) == ""
