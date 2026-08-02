"""Test module."""

import os

from ml_switcheroo_compiler.export.export_api import ExportArchive
from ml_switcheroo_compiler.export.pb_utils import encode_varint


def test_export_api(tmp_path):
    arch = ExportArchive()

    resource = object()
    arch.track(resource)
    assert arch.trackables[id(resource)] is resource

    arch.add_endpoint("name", lambda x: x)
    assert "name" in arch.endpoints

    arch.add_variable_collection("vars", "variables")
    assert arch.collections["vars"] == "variables"

    arch.write_out(str(tmp_path))
    assert os.path.exists(os.path.join(tmp_path, "saved_model.pb"))
    assert os.path.exists(os.path.join(tmp_path, "variables", "variables.data-00000-of-00001"))
    assert os.path.exists(os.path.join(tmp_path, "variables", "variables.index"))


def test_export_api_branch(tmp_path):
    arch = ExportArchive()
    arch.write_out(str(tmp_path))
    assert os.path.exists(os.path.join(tmp_path, "saved_model.pb"))


def test_pb_utils():
    # Value < 0
    assert encode_varint(-1)

    # Large value
    assert encode_varint(1 << 8)
