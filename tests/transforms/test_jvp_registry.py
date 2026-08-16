# ruff: noqa: E501
import pytest

from ml_switcheroo_compiler.transforms.autodiff_rules.jvp_registry import _JVP_REGISTRY, get_jvp, register_jvp

"Provides required module functionality."


def test_jvp_registry_coverage_brute() -> None:
    """Test the jvp registry coverage brute behavior.

    Returns:
        Any: The inferred shape or computed result.
    """
    "Execute the requested function."
    if "fake_op" in _JVP_REGISTRY:
        del _JVP_REGISTRY["fake_op"]

    @register_jvp("fake_op")
    def fake_jvp() -> None:
        """Evaluate and process the fake jvp operation.

        Returns:
            Any: The evaluated or processed output.
        """

    from unittest.mock import patch

    with patch("ml_switcheroo_compiler.transforms.autodiff_rules.jvp_registry.get_jvp_from_data", return_value=None):
        assert get_jvp("fake_op") == fake_jvp

    with pytest.raises(ValueError, match="already registered"):

        @register_jvp("fake_op")
        def fake_jvp2() -> None:
            """Evaluate and process the fake jvp2 operation.

            Returns:
                Any: The evaluated or processed output.
            """

    # Test error
    with patch("ml_switcheroo_compiler.transforms.autodiff_rules.jvp_registry.get_jvp_from_data", return_value=None):
        with pytest.raises(ValueError, match="No JVP rule"):
            get_jvp("NonExistentOp")

    # Test data jvp
    from unittest.mock import patch

    with patch("ml_switcheroo_compiler.transforms.autodiff_rules.jvp_registry.get_jvp_from_data", return_value="data_jvp"):
        assert get_jvp("fake_op_2") == "data_jvp"

    # Test get_jvp_from_data and has_jvp
    from ml_switcheroo_compiler.transforms.autodiff_rules.jvp_registry import has_jvp

    with patch("ml_switcheroo_compiler.transforms.autodiff_rules.jvp_registry.get_jvp_from_data", return_value=None):
        assert has_jvp("fake_op") is True
        assert has_jvp("NonExistentOp") is False
    with patch("ml_switcheroo_compiler.transforms.autodiff_rules.jvp_registry.get_jvp_from_data", return_value="something"):
        assert has_jvp("fake_op") is True
