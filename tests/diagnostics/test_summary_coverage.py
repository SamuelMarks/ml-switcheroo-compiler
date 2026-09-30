"""Tests for test_summary_coverage."""

from __future__ import annotations

import os
import shutil
import tempfile

import numpy as np

import ml_switcheroo_compiler.diagnostics.summary as summary_mod
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig


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
        """
        raise TypeError("Not iterable dimension")


def test_summary_exhaustive() -> None:
    """Verify writing protobuf events and encoding 2D, 3D, and 4D image tensors."""
    temp_dir = tempfile.mkdtemp()
    try:
        # 1. write_raw_pb
        summary_mod.write_raw_pb(b"\x08\x01\x12\x04test", temp_dir)
        pb_file = os.path.join(temp_dir, "events.out.tfevents.pb")
        assert os.path.exists(pb_file)
        with open(pb_file, "rb") as f:
            assert f.read() == b"\x08\x01\x12\x04test"

        # 2. encode_image with 4D float tensor: (1, 16, 16, 3)
        raw4d = np.random.rand(1, 16, 16, 3).astype(np.float32)
        img4d = Tensor(raw4d, TensorConfig((1, 16, 16, 3), "float32", "cpu"))
        bytes4d = summary_mod.encode_image(img4d)
        assert bytes4d[:4] == b"\x89PNG"

        # 3. encode_image with 3D channels-first tensor: (3, 16, 16)
        raw3d_cf = np.random.rand(3, 16, 16).astype(np.float32)
        img3d_ch_first = Tensor(raw3d_cf, TensorConfig((3, 16, 16), "float32", "cpu"))
        bytes3d_cf = summary_mod.encode_image(img3d_ch_first)
        assert bytes3d_cf[:4] == b"\x89PNG"

        # 4. encode_image with 3D single channel: (16, 16, 1)
        raw3d_s = (np.random.rand(16, 16, 1) * 255).astype(np.uint8)
        img3d_single = Tensor(raw3d_s, TensorConfig((16, 16, 1), "uint8", "cpu"))
        bytes3d_s = summary_mod.encode_image(img3d_single)
        assert bytes3d_s[:4] == b"\x89PNG"

        # 5. encode_image with integer 2D array: (16, 16)
        arr2d = np.full((16, 16), 128, dtype=np.uint8)
        bytes2d = summary_mod.encode_image(arr2d)
        assert bytes2d[:4] == b"\x89PNG"
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
