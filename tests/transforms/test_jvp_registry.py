"""Tests for test_jvp_registry."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import pytest

from ml_switcheroo_compiler.transforms.autodiff_rules.jvp_registry import _JVP_REGISTRY, get_jvp, register_jvp

if TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import EagerTensorProtocol, EagerValue


def test_jvp_registry_coverage_brute() -> None:
    """Test the jvp registry coverage brute behavior.

    Returns:
        TensorOrScalar: The inferred shape or computed result.
    """
    if "fake_op" in _JVP_REGISTRY:
        del _JVP_REGISTRY["fake_op"]

    @register_jvp("fake_op")
    def fake_jvp() -> None:
        """Evaluate and process the fake jvp operation.

        Returns:
            TensorOrScalar: The evaluated or processed output.
        """

    from unittest.mock import patch

    with patch("ml_switcheroo_compiler.transforms.autodiff_rules.jvp_registry.get_jvp_from_data", return_value=None):
        assert get_jvp("fake_op") == fake_jvp
    with pytest.raises(ValueError, match="already registered"):

        @register_jvp("fake_op")
        def fake_jvp2() -> None:
            """Evaluate and process the fake jvp2 operation.

            Returns:
                TensorOrScalar: The evaluated or processed output.
            """

    with patch("ml_switcheroo_compiler.transforms.autodiff_rules.jvp_registry.get_jvp_from_data", return_value=None):
        with pytest.raises(ValueError, match="No JVP rule"):
            get_jvp("NonExistentOp")
    from unittest.mock import patch

    with patch("ml_switcheroo_compiler.transforms.autodiff_rules.jvp_registry.get_jvp_from_data", return_value="data_jvp"):
        assert cast("EagerTensorProtocol", get_jvp("fake_op_2")) == cast("EagerTensorProtocol", "data_jvp")
    from ml_switcheroo_compiler.transforms.autodiff_rules.jvp_registry import has_jvp

    with patch("ml_switcheroo_compiler.transforms.autodiff_rules.jvp_registry.get_jvp_from_data", return_value=None):
        assert has_jvp("fake_op") is True
        assert has_jvp("NonExistentOp") is False
    with patch("ml_switcheroo_compiler.transforms.autodiff_rules.jvp_registry.get_jvp_from_data", return_value="something"):
        assert has_jvp("OpOnlyInData") is True


def test_jvp_registry_full_branches() -> None:
    """Test remaining branch coverage in jvp_registry.py."""
    import tempfile
    from unittest.mock import patch

    import yaml

    from ml_switcheroo_compiler.transforms.autodiff_rules.jvp_registry import _JVP_REGISTRY, get_jvp, load_primitive_jvp_rules

    load_primitive_jvp_rules(path=None)
    rules_cached = load_primitive_jvp_rules(path=None)
    assert len(rules_cached) > 0

    def mock_get_data(name: EagerValue) -> None:
        """Test function mock_get_data."""
        getattr(_JVP_REGISTRY, "__setitem__")(name, "lazy_jvp_result")
        return None

    with patch("ml_switcheroo_compiler.transforms.autodiff_rules.jvp_registry.get_jvp_from_data", side_effect=mock_get_data):
        assert cast("EagerTensorProtocol", get_jvp("lazy_op_jvp")) == cast("EagerTensorProtocol", "lazy_jvp_result")
    with patch("os.path.exists", return_value=False):
        rules_none = load_primitive_jvp_rules(path="/nonexistent/path/jvp.yaml")
        assert rules_none == {}
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(["item1", "item2"], f)
        temp_list = f.name
    rules_list = load_primitive_jvp_rules(path=temp_list)
    assert rules_list == {}
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump({"op_no_jvp": {"other": 1}, "op_str": "val"}, f)
        temp_no_jvp = f.name
    rules_no_jvp = load_primitive_jvp_rules(path=temp_no_jvp)
    assert rules_no_jvp == {}
