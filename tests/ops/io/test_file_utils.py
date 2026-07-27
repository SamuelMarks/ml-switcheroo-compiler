# ruff: noqa: E501
import tempfile

from ml_switcheroo_compiler.utils.file_utils import exists


def test_exists():
    with tempfile.NamedTemporaryFile() as f:
        assert exists(f.name) is True
    assert exists("non_existent_file_path_1234567") is False
