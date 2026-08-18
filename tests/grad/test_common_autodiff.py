"""Module test_common_autodiff.py."""


def test_autodiff_common():
    """test_autodiff_common."""
    from ml_switcheroo_compiler.transforms.autodiff_rules.common import UnconnectedGradients, make_zero_jvp, make_zero_vjp

    class DummyNode:
        """DummyNode."""

        inputs = [1, 2]

    vjp_fn = make_zero_vjp("dummy")
    assert vjp_fn(None, DummyNode(), "cot") == (UnconnectedGradients.ZERO, UnconnectedGradients.ZERO)

    jvp_fn = make_zero_jvp("dummy")
    assert jvp_fn(None, None, None) == ""
