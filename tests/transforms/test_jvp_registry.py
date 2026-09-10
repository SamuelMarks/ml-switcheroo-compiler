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
        assert has_jvp("OpOnlyInData") is True


def test_jvp_registry_full_branches():
    """Test remaining branch coverage in jvp_registry.py."""
    import tempfile
    from unittest.mock import patch

    import yaml

    from ml_switcheroo_compiler.transforms.autodiff_rules.jvp_registry import (
        _JVP_REGISTRY,
        get_jvp,
        load_primitive_jvp_rules,
    )

    # Cache hit branch (line 42)
    load_primitive_jvp_rules(path=None)
    rules_cached = load_primitive_jvp_rules(path=None)
    assert len(rules_cached) > 0

    # 1. Lazy load branch where get_jvp_from_data populates _JVP_REGISTRY (line 117)
    def mock_get_data(name):
        _JVP_REGISTRY[name] = "lazy_jvp_result"
        return None

    with patch("ml_switcheroo_compiler.transforms.autodiff_rules.jvp_registry.get_jvp_from_data", side_effect=mock_get_data):
        assert get_jvp("lazy_op_jvp") == "lazy_jvp_result"

    # 2. load_primitive_jvp_rules with nonexistent path (branches 47, 50->58)
    with patch("os.path.exists", return_value=False):
        rules_none = load_primitive_jvp_rules(path="/nonexistent/path/jvp.yaml")
        assert rules_none == {}

    # 3. load_primitive_jvp_rules with non-dict data (branch 53->58)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(["item1", "item2"], f)
        temp_list = f.name
    rules_list = load_primitive_jvp_rules(path=temp_list)
    assert rules_list == {}

    # 4. load_primitive_jvp_rules with dict without "jvp" key (branch 55->54)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump({"op_no_jvp": {"other": 1}, "op_str": "val"}, f)
        temp_no_jvp = f.name
    rules_no_jvp = load_primitive_jvp_rules(path=temp_no_jvp)
    assert rules_no_jvp == {}
