"""Unit tests for Vector-Jacobian Product (VJP) rule registry."""

from __future__ import annotations

import typing

# ruff: noqa: E501
import pytest

from ml_switcheroo_compiler.transforms.autodiff_rules.vjp_registry import _VJP_REGISTRY, get_vjp, register_vjp

if typing.TYPE_CHECKING:
    pass


def test_vjp_registry_coverage() -> None:
    """Test the vjp registry coverage behavior.

    Returns:
        None
    """
    from ml_switcheroo_ir import LogicalGraph, LogicalNode

    if "fake_op" in _VJP_REGISTRY:
        del _VJP_REGISTRY["fake_op"]

    @register_vjp("fake_op")
    def fake_vjp(g: LogicalGraph, n: LogicalNode, cot: str) -> tuple[str, ...]:
        """Evaluate and process the fake vjp operation.

        Args:
            g: Graph.
            n: Node.
            cot: Cotangent ID.

        Returns:
            tuple[str, ...]: The evaluated or processed output.
        """
        del g, n, cot
        return ()

    with pytest.raises(ValueError, match="already registered"):

        @register_vjp("fake_op")
        def fake_vjp2(g: LogicalGraph, n: LogicalNode, cot: str) -> tuple[str, ...]:
            """Evaluate and process the fake vjp2 operation.

            Args:
                g: Graph.
                n: Node.
                cot: Cotangent ID.

            Returns:
                tuple[str, ...]: The evaluated or processed output.
            """
            del g, n, cot
            return ()

    assert id(get_vjp("fake_op")) == id(fake_vjp)

    # Test error
    with pytest.raises(ValueError, match="No VJP rule"):
        get_vjp("NonExistentOp")

    # Test data vjp
    from unittest.mock import patch

    with patch(
        "ml_switcheroo_compiler.transforms.autodiff_rules.vjp_registry.get_vjp_from_data", return_value="data_vjp"
    ):
        from tests.testing_utils import require_callable

        assert require_callable(get_vjp)("fake_op_2") == "data_vjp"

    # Test get_vjp_from_data and has_vjp
    from ml_switcheroo_compiler.transforms.autodiff_rules.vjp_registry import has_vjp

    with patch("ml_switcheroo_compiler.transforms.autodiff_rules.vjp_registry.get_vjp_from_data", return_value=None):
        assert has_vjp("fake_op") is True
        assert has_vjp("NonExistentOp") is False
    with patch(
        "ml_switcheroo_compiler.transforms.autodiff_rules.vjp_registry.get_vjp_from_data", return_value="something"
    ):
        assert has_vjp("fake_op") is True


def test_vjp_registry_lazy_load() -> None:
    """Test lazy loading in get_vjp when entry is initially missing from the registry."""
    from unittest.mock import patch

    from ml_switcheroo_ir import LogicalGraph, LogicalNode

    from ml_switcheroo_compiler.transforms.autodiff_rules.vjp_registry import _VJP_REGISTRY, get_vjp

    def lazy_vjp_fn(g: LogicalGraph, n: LogicalNode, cot: str) -> str:
        """Lazy VJP stub.

        Args:
            g: Graph.
            n: Node.
            cot: Cotangent ID.

        Returns:
            str: Sentinel cotangent ID.
        """
        del g, n, cot
        return "lazy_loaded_vjp"

    def mock_get_data(name: str) -> None:
        """Mock loader that populates registry on demand.

        Args:
            name: Test parameter `name`.
        """
        _VJP_REGISTRY[name] = lazy_vjp_fn
        return None

    with patch(
        "ml_switcheroo_compiler.transforms.autodiff_rules.vjp_registry.get_vjp_from_data", side_effect=mock_get_data
    ):
        res = get_vjp("test_lazy_op")
        assert id(res) == id(lazy_vjp_fn)


def test_has_vjp_not_in_registry_or_data() -> None:
    """Test has_vjp returns False when operation is absent from registry and file data."""
    from unittest.mock import patch

    from ml_switcheroo_compiler.transforms.autodiff_rules.vjp_registry import has_vjp

    with patch("ml_switcheroo_compiler.transforms.autodiff_rules.vjp_registry.get_vjp_from_data", return_value=None):
        assert has_vjp("AbsolutelyMissingOpTypeXYZ123") is False


def test_has_vjp_data() -> None:
    """Test has_vjp returns True when operation is found in file data."""
    from unittest.mock import patch

    from ml_switcheroo_compiler.transforms.autodiff_rules.vjp_registry import has_vjp

    with patch(
        "ml_switcheroo_compiler.transforms.autodiff_rules.vjp_registry.get_vjp_from_data", return_value=lambda x: x
    ):
        assert has_vjp("AbsolutelyMissingOpTypeXYZ123_Data") is True


def test_vjp_registry_full_branches() -> None:
    """Test remaining branch coverage in vjp_registry.py."""
    import tempfile
    from unittest.mock import patch

    import yaml

    import ml_switcheroo_compiler.transforms.autodiff_rules.vjp_registry as vr
    from ml_switcheroo_compiler.transforms.autodiff_rules.vjp_registry import load_primitive_vjp_rules

    vr._DECLARATIVE_VJP_CACHE = None
    # Cache hit branch (line 43)
    load_primitive_vjp_rules(path=None)
    rules_cached = load_primitive_vjp_rules(path=None)
    assert len(rules_cached) > 0

    # 1. load_primitive_vjp_rules with nonexistent path (branches 47, 50->58)
    with patch("os.path.exists", return_value=False):
        rules_none = load_primitive_vjp_rules(path="/nonexistent/path/vjp.yaml")
        assert rules_none == {}

    # 2. load_primitive_vjp_rules with non-dict data (branch 53->58)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(["item1", "item2"], f)
        temp_list = f.name
    rules_list = load_primitive_vjp_rules(path=temp_list)
    assert rules_list == {}

    # 3. load_primitive_vjp_rules with dict without "vjp" key (branch 55->54)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump({"op_no_vjp": {"other": 1}, "op_str": "val"}, f)
        temp_no_vjp = f.name
    rules_no_vjp = load_primitive_vjp_rules(path=temp_no_vjp)
    assert rules_no_vjp == {}
