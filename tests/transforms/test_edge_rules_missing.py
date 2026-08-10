import importlib
import sys

from ml_switcheroo_compiler.transforms.autodiff_rules.jvp_registry import _JVP_REGISTRY
from ml_switcheroo_compiler.transforms.autodiff_rules.vjp_registry import _VJP_REGISTRY


def test_edge_rules_already_registered():
    # Make sure something is registered so that reload hits the ValueError
    # MatMul is one of the _EDGE_OPS
    assert "MatMul" in _VJP_REGISTRY
    assert "MatMul" in _JVP_REGISTRY

    # Ensure it's in sys.modules before reloading
    if "ml_switcheroo_compiler.transforms.autodiff_rules.edge_rules" not in sys.modules:
        importlib.import_module("ml_switcheroo_compiler.transforms.autodiff_rules.edge_rules")

    # Reloading the module will attempt to register it again and hit the ValueError
    importlib.reload(sys.modules["ml_switcheroo_compiler.transforms.autodiff_rules.edge_rules"])
