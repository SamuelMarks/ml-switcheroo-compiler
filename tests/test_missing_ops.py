"""Tests to ensure that all registered operations can be retrieved."""

import os

import yaml

from ml_switcheroo_compiler.ops import get_op
from tests.test_models import AllOpsManifest

_MANIFEST_PATH = os.path.join(os.path.dirname(__file__), "manifests", "all_ops.yaml")
with open(_MANIFEST_PATH) as f:
    _manifest = AllOpsManifest(**yaml.safe_load(f))
ALL_OPS: list[str] = _manifest.all_ops


def test_all_missing_ops() -> None:
    """Test that all known operations can be successfully retrieved from the registry.

    This test iterates over all operation names and asserts that `get_op` does not return None.
    It collects all missing operations and reports them together if any are missing.
    """
    missing: list[str] = []
    for op_name in ALL_OPS:
        try:
            get_op(op_name)
        except KeyError:
            missing.append(op_name)
    assert not missing, f"Missing ops: {missing}"
