"""Tests for test_facades_coverage."""

from __future__ import annotations

import ml_switcheroo_compiler.export as export_pkg
import ml_switcheroo_compiler.interpreter as interpreter_pkg
import ml_switcheroo_compiler.nn as nn_pkg
import ml_switcheroo_compiler.transforms as transforms_pkg
import ml_switcheroo_compiler.transforms.autodiff_rules as autodiff_rules_pkg
from tests.testing_utils import has_backend_member


class _DummyBadShape:
    """Shape object whose iterator raises TypeError."""

    def __iter__(self) -> _DummyBadShape:
        """Return self as iterator.

        Returns:
            _DummyBadShape: Self.
        """
        return self

    def __next__(self) -> int:
        """Raise TypeError on iteration.

        Raises:
            TypeError: Simulated invalid dimension.

        Returns:
            int: Computed output value from `__next__`.
        """
        raise TypeError("Not iterable dimension")


def test_package_facades_exhaustive() -> None:
    """Verify exports and facades for export, interpreter, nn, transforms, and autodiff packages."""
    assert has_backend_member(export_pkg, "ExportArchive")

    assert has_backend_member(interpreter_pkg, "Environment")
    assert has_backend_member(interpreter_pkg, "evaluate_graph")

    assert "glorot_uniform" in nn_pkg.__all__
    assert has_backend_member(nn_pkg, "glorot_uniform")

    assert has_backend_member(transforms_pkg, "PassManager")
    assert has_backend_member(transforms_pkg, "constant_folding_pass")
    assert has_backend_member(transforms_pkg, "grad")

    assert has_backend_member(autodiff_rules_pkg, "custom_rules")
