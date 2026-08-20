# ruff: noqa: E501
import pytest

from ml_switcheroo_compiler.transforms.autodiff_rules.vjp_registry import _VJP_REGISTRY, get_vjp, register_vjp

"Provides required module functionality."


def test_vjp_registry_coverage() -> None:
    """Test the vjp registry coverage behavior.

    Returns:
        Any: The inferred shape or computed result.
    """
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

    # Test error
    with pytest.raises(ValueError, match="No VJP rule"):
        get_vjp("NonExistentOp")

    # Test data vjp
    from unittest.mock import patch

    with patch("ml_switcheroo_compiler.transforms.autodiff_rules.vjp_registry.get_vjp_from_data", return_value="data_vjp"):
        assert get_vjp("fake_op_2") == "data_vjp"

    # Test get_vjp_from_data and has_vjp
    from ml_switcheroo_compiler.transforms.autodiff_rules.vjp_registry import has_vjp

    with patch("ml_switcheroo_compiler.transforms.autodiff_rules.vjp_registry.get_vjp_from_data", return_value=None):
        assert has_vjp("fake_op") is True
        assert has_vjp("NonExistentOp") is False
    with patch("ml_switcheroo_compiler.transforms.autodiff_rules.vjp_registry.get_vjp_from_data", return_value="something"):
        assert has_vjp("fake_op") is True


def test_vjp_registry_lazy_load():
    from unittest.mock import patch

    from ml_switcheroo_compiler.transforms.autodiff_rules.vjp_registry import _VJP_REGISTRY, get_vjp

    def mock_get_data(name):
        _VJP_REGISTRY[name] = "lazy_loaded_vjp"
        return None

    with patch("ml_switcheroo_compiler.transforms.autodiff_rules.vjp_registry.get_vjp_from_data", side_effect=mock_get_data):
        res = get_vjp("test_lazy_op")
        assert res == "lazy_loaded_vjp"
