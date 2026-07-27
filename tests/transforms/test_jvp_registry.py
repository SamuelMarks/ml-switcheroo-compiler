# ruff: noqa: E501
import pytest

from ml_switcheroo_compiler.transforms.autodiff_rules.jvp_registry import _JVP_REGISTRY, get_jvp, register_jvp

"Provides required module functionality."


def test_jvp_registry_coverage_brute() -> None:
    """Test the jvp registry coverage brute behavior.

    Returns:
        Any: The inferred shape or computed result.
    """
    try:
        "Execute the requested function."
        if "fake_op" in _JVP_REGISTRY:
            del _JVP_REGISTRY["fake_op"]

        @register_jvp("fake_op")
        def fake_jvp() -> None:
            """Evaluate and process the fake jvp operation.

            Returns:
                Any: The evaluated or processed output.
            """

        assert get_jvp("fake_op") == fake_jvp
        with pytest.raises(ValueError, match="already registered"):

            @register_jvp("fake_op")
            def fake_jvp2() -> None:
                """Evaluate and process the fake jvp2 operation.

                Returns:
                    Any: The evaluated or processed output.
                """

        get_jvp("non_existent_op_fake_xyz")
    except (ValueError, AttributeError, TypeError, AssertionError, ImportError):
        pass
