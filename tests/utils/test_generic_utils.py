"""Tests for generic_utils.py."""

import os
import tarfile
import urllib.error
import urllib.request
import zipfile
from unittest.mock import MagicMock, patch

import pytest

from ml_switcheroo_compiler.utils.generic_utils import (
    GetFileConfig,
    _download_remote_file,
    _extract_archive,
    _validate_cache,
    get_file,
)


def test_validate_cache(tmp_path: str) -> None:
    """Test _validate_cache."""
    fpath = os.path.join(str(tmp_path), "test.txt")
    assert not _validate_cache(fpath)
    with open(fpath, "w") as f:
        f.write("test")
    assert _validate_cache(fpath)


@patch("urllib.request.urlretrieve")
def test_download_remote_file(mock_urlretrieve: MagicMock) -> None:
    """Test _download_remote_file."""
    _download_remote_file("http://example.com", "fpath")
    mock_urlretrieve.assert_called_once_with("http://example.com", "fpath")

    mock_urlretrieve.side_effect = urllib.error.URLError("error")
    with pytest.raises(RuntimeError):
        _download_remote_file("http://example.com", "fpath")


def test_extract_archive(tmp_path: str) -> None:
    """Test _extract_archive."""
    datadir = str(tmp_path)
    # create a mock zip file
    fpath = os.path.join(datadir, "test.zip")
    with zipfile.ZipFile(fpath, "w") as archive:
        archive.writestr("test.txt", "test")
    _extract_archive(fpath, datadir)
    assert os.path.exists(os.path.join(datadir, "test.txt"))

    # create a mock tar file
    fpath = os.path.join(datadir, "test.tar")
    with tarfile.open(fpath, "w") as archive:
        with open(os.path.join(datadir, "test.txt"), "w") as f:
            f.write("test")
        archive.add(os.path.join(datadir, "test.txt"), arcname="test.txt")
    _extract_archive(fpath, datadir)

    # create a mock tar.gz file
    fpath = os.path.join(datadir, "test.tar.gz")
    with tarfile.open(fpath, "w:gz") as archive:
        with open(os.path.join(datadir, "test.txt"), "w") as f:
            f.write("test")
        archive.add(os.path.join(datadir, "test.txt"), arcname="test.txt")
    _extract_archive(fpath, datadir)


@patch("ml_switcheroo_compiler.utils.generic_utils._extract_archive")
@patch("ml_switcheroo_compiler.utils.generic_utils._download_remote_file")
@patch("ml_switcheroo_compiler.utils.generic_utils._validate_cache")
def test_get_file(
    mock_validate_cache: MagicMock,
    mock_download_remote_file: MagicMock,
    mock_extract_archive: MagicMock,
) -> None:
    """Test get_file."""
    mock_validate_cache.return_value = False

    config = GetFileConfig(cache_dir=".", untar=True)
    fpath = get_file("fname", "origin", config)

    assert fpath is not None
    mock_download_remote_file.assert_called_once()
    mock_extract_archive.assert_called_once()

    # test cache hit
    mock_validate_cache.return_value = True
    mock_download_remote_file.reset_mock()
    mock_extract_archive.reset_mock()
    fpath = get_file("fname", "origin", config)
    assert fpath is not None
    mock_download_remote_file.assert_not_called()
    mock_extract_archive.assert_not_called()
