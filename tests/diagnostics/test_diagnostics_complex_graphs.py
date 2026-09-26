# ruff: noqa: D103
"""Tests for diagnostics extras."""

import numpy as np

from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.diagnostics import enable_dump_debug_info, encode_image, write_raw_pb


def test_diagnostics_extras() -> None:
    t = Tensor(np.zeros((10, 10, 3), dtype=np.uint8), TensorConfig((10, 10, 3), "uint8", "cpu"))

    assert encode_image(t).startswith(b"\x89PNG")

    import shutil

    shutil.rmtree("/tmp/test_diag_ml_switcheroo", ignore_errors=True)

    write_raw_pb(b"data", "/tmp/test_diag_ml_switcheroo")

    enable_dump_debug_info("/tmp/test_diag_ml_switcheroo_dump")
