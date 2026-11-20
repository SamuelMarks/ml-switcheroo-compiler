"""Behavioral unit tests for facades."""

from __future__ import annotations

import ml_switcheroo_compiler.export as export_pkg
import ml_switcheroo_compiler.interpreter as interpreter_pkg
import ml_switcheroo_compiler.nn as nn_pkg
import ml_switcheroo_compiler.transforms as transforms_pkg
import ml_switcheroo_compiler.transforms.autodiff_rules as autodiff_rules_pkg
from tests.testing_utils import adapt_backend_module


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
    assert adapt_backend_module(export_pkg).has_symbol("ExportArchive")

    assert adapt_backend_module(interpreter_pkg).has_symbol("Environment")
    assert adapt_backend_module(interpreter_pkg).has_symbol("evaluate_graph")

    assert "glorot_uniform" in nn_pkg.__all__
    assert adapt_backend_module(nn_pkg).has_symbol("glorot_uniform")

    assert adapt_backend_module(transforms_pkg).has_symbol("PassManager")
    assert adapt_backend_module(transforms_pkg).has_symbol("constant_folding_pass")
    assert adapt_backend_module(transforms_pkg).has_symbol("grad")

    assert adapt_backend_module(autodiff_rules_pkg).has_symbol("custom_rules")
