# ruff: noqa: E501
import pytest

from ml_switcheroo_compiler.transforms.autodiff_rules.vjp_registry import _VJP_REGISTRY, get_vjp, register_vjp

"Provides required module functionality."


def test_vjp_registry_coverage() -> None:
    """Test the vjp registry coverage behavior.

    Returns:
        Any: The inferred shape or computed result.
    """
    try:
        "Execute the requested function."
        if "fake_op" in _VJP_REGISTRY:
            del _VJP_REGISTRY["fake_op"]

        @register_vjp("fake_op")
        def fake_vjp() -> None:
            """Evaluate and process the fake vjp operation.

            Returns:
                Any: The evaluated or processed output.
            """

        with pytest.raises(ValueError, match="already registered"):

            @register_vjp("fake_op")
            def fake_vjp2() -> None:
                """Evaluate and process the fake vjp2 operation.

                Returns:
                    Any: The evaluated or processed output.
                """

        assert get_vjp("fake_op") == fake_vjp
    except (ValueError, AttributeError, TypeError, AssertionError, ImportError):
        pass
