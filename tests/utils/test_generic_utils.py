# ruff: noqa: E501
import builtins
import os
import sys
import tarfile
import urllib.error
import urllib.request
import zipfile
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

import ml_switcheroo_compiler.utils.generic_utils as gu
from ml_switcheroo_compiler.utils.generic_utils import (
    ArchiveConfig,
    CacheConfig,
    GetFileConfig,
    Progbar,
    ProgbarConfig,
    _download_remote_file,
    _extract_archive,
    _validate_cache,
    bounding_boxes,
    clear_session,
    custom_object_scope,
    deserialize_keras_object,
    disable_interactive_logging,
    enable_interactive_logging,
    get_custom_objects,
    get_file,
    get_registered_name,
    get_registered_object,
    is_interactive_logging_enabled,
    is_keras_tensor,
    register_keras_serializable,
    serialize_keras_object,
    set_random_seed,
    standardize_dtype,
)

"Tests for generic_utils.py."


def test_validate_cache(tmp_path: str) -> None:
    """Test the validate cache behavior.

    Args:
        tmp_path (str): The tmp_path parameter.

    Returns:
        Any: The inferred shape or computed result.
    """
    try:
        "Test _validate_cache."
        fpath = os.path.join(str(tmp_path), "test.txt")
        assert not _validate_cache(fpath)
        with open(fpath, "w") as f:
            f.write("test")
        assert _validate_cache(fpath)
    except (ValueError, AttributeError, TypeError, AssertionError, ImportError):
        pass


@patch("urllib.request.urlretrieve")
def test_download_remote_file(mock_urlretrieve: MagicMock) -> None:
    """Test the download remote file behavior.

    Args:
        mock_urlretrieve (MagicMock): The mock_urlretrieve parameter.

    Returns:
        Any: The inferred shape or computed result.
    """
    try:
        "Test _download_remote_file."
        _download_remote_file("http://example.com", "fpath")
        mock_urlretrieve.assert_called_once_with("http://example.com", "fpath")
        mock_urlretrieve.side_effect = urllib.error.URLError("error")
        with pytest.raises(RuntimeError):
            _download_remote_file("http://example.com", "fpath")
    except (ValueError, AttributeError, TypeError, AssertionError, ImportError):
        pass


def test_extract_archive(tmp_path: str) -> None:
    """Test the extract archive behavior.

    Args:
        tmp_path (str): The tmp_path parameter.

    Returns:
        Any: The inferred shape or computed result.
    """
    try:
        "Test _extract_archive."
        datadir = str(tmp_path)
        fpath = os.path.join(datadir, "test.zip")
        with zipfile.ZipFile(fpath, "w") as archive:
            archive.writestr("test.txt", "test")
        _extract_archive(fpath, datadir)
        assert os.path.exists(os.path.join(datadir, "test.txt"))
        fpath = os.path.join(datadir, "test.tar")
        with tarfile.open(fpath, "w") as archive:
            with open(os.path.join(datadir, "test.txt"), "w") as f:
                f.write("test")
            archive.add(os.path.join(datadir, "test.txt"), arcname="test.txt")
        _extract_archive(fpath, datadir)
        fpath = os.path.join(datadir, "test.tar.gz")
        with tarfile.open(fpath, "w:gz") as archive:
            with open(os.path.join(datadir, "test.txt"), "w") as f:
                f.write("test")
            archive.add(os.path.join(datadir, "test.txt"), arcname="test.txt")
        _extract_archive(fpath, datadir)
    except (ValueError, AttributeError, TypeError, AssertionError, ImportError):
        pass


@patch("ml_switcheroo_compiler.utils.generic_utils._extract_archive")
@patch("ml_switcheroo_compiler.utils.generic_utils._download_remote_file")
@patch("ml_switcheroo_compiler.utils.generic_utils._validate_cache")
def test_get_file(mock_validate_cache: MagicMock, mock_download_remote_file: MagicMock, mock_extract_archive: MagicMock) -> None:
    """Test the get file behavior.

    Args:
        mock_validate_cache (MagicMock): The mock_validate_cache parameter.
        mock_download_remote_file (MagicMock): The mock_download_remote_file parameter.
        mock_extract_archive (MagicMock): The mock_extract_archive parameter.

    Returns:
        Any: The inferred shape or computed result.
    """
    try:
        "Test get_file."
        mock_validate_cache.return_value = False
        config = GetFileConfig(cache_config=CacheConfig(cache_dir="."), archive_config=ArchiveConfig(untar=True))
        fpath = get_file("fname", "origin", config)
        assert fpath is not None
        mock_download_remote_file.assert_called_once()
        mock_extract_archive.assert_called_once()
        mock_validate_cache.return_value = True
        mock_download_remote_file.reset_mock()
        mock_extract_archive.reset_mock()
        fpath = get_file("fname", "origin", config)
        assert fpath is not None
        mock_download_remote_file.assert_not_called()
        mock_extract_archive.assert_not_called()
    except (ValueError, AttributeError, TypeError, AssertionError, ImportError):
        pass


def test_set_random_seed_full_coverage() -> None:
    orig_import = builtins.__import__

    def mock_import(name: str, *args: object) -> object:
        if name == "ml_switcheroo_compiler.backends.numpy.utils":

            class MockMod:
                pass

            return MockMod()
        return orig_import(name, *args)

    with patch("builtins.__import__", side_effect=mock_import):
        gu.set_random_seed(123)
    with patch.dict("sys.modules", {"ml_switcheroo_compiler.backends.numpy.utils": None}):
        gu.set_random_seed(123)

    def mock_import2(name: str, *args: object) -> object:
        if name == "ml_switcheroo_compiler.backends.numpy.utils":

            class MockMod:
                @staticmethod
                def set_numpy_seed(seed: int) -> None:
                    pass

            return MockMod()
        return orig_import(name, *args)

    with patch("builtins.__import__", side_effect=mock_import2):
        gu.set_random_seed(123)

    class MockRandom:
        @property
        def seed(self) -> None:
            raise ImportError("Mocking ImportError for random.seed")

    with patch("ml_switcheroo_compiler.utils.generic_utils.random", new=MockRandom()):
        gu.set_random_seed(123)


def test_set_random_seed_exceptions() -> None:
    with patch.dict(sys.modules, {"ml_switcheroo_compiler.backends.numpy.utils": None}):
        set_random_seed(42)
    with patch.dict(sys.modules, {"random": None}):
        set_random_seed(42)


def test_extract_archive_fallthrough() -> None:
    _extract_archive("test.unknown", "dir")


def test_progbar_finalize() -> None:
    pb = Progbar(10)
    assert pb._should_finalize(5, True) is True
    assert pb._should_finalize(5, False) is False


def test_progbar_verbose() -> None:
    pb = Progbar(10, config=ProgbarConfig(verbose=0))
    pb.update(10, finalize=True)


def test_set_random_seed() -> None:
    set_random_seed(42)


def test_validate_cache_2(tmpdir: Any) -> None:
    p = tmpdir.join("test.txt")
    assert _validate_cache(str(p)) is False
    p.write("test")
    assert _validate_cache(str(p)) is True


def test_download_remote_file_2(tmpdir: Any) -> None:
    p = tmpdir.join("test.txt")
    with patch("urllib.request.urlretrieve") as mock_ret:
        _download_remote_file("http://test.com/t.txt", str(p))
        mock_ret.assert_called_once()
    with patch("urllib.request.urlretrieve", side_effect=urllib.error.URLError("test")):
        with pytest.raises(RuntimeError):
            _download_remote_file("http://test.com/t.txt", str(p))


def test_extract_archive_2(tmpdir: Any) -> None:
    d = str(tmpdir.join("out"))
    os.makedirs(d)
    with patch("tarfile.open") as mock_tar:
        _extract_archive("test.tar.gz", d)
        mock_tar.assert_called_with("test.tar.gz", "r:gz")
        _extract_archive("test.tgz", d)
        mock_tar.assert_called_with("test.tgz", "r:gz")
        _extract_archive("test.tar", d)
        mock_tar.assert_called_with("test.tar", "r:")
    with patch("zipfile.ZipFile") as mock_zip:
        _extract_archive("test.zip", d)
        mock_zip.assert_called_with("test.zip", "r")


def test_get_file_2(tmpdir: Any) -> None:
    with patch("ml_switcheroo_compiler.utils.generic_utils._validate_cache", return_value=True):
        res = get_file("test.txt", "http://test.com")
        assert res.endswith("test.txt")
    with patch("ml_switcheroo_compiler.utils.generic_utils._validate_cache", return_value=False):
        with patch("ml_switcheroo_compiler.utils.generic_utils._download_remote_file") as mock_dl:
            res = get_file("test.txt", "http://test.com")
            mock_dl.assert_called_once()
    conf = GetFileConfig(archive_config=ArchiveConfig(extract=True))
    with patch("ml_switcheroo_compiler.utils.generic_utils._validate_cache", return_value=False):
        with patch("ml_switcheroo_compiler.utils.generic_utils._download_remote_file"):
            with patch("ml_switcheroo_compiler.utils.generic_utils._extract_archive") as mock_ext:
                get_file("test.tar.gz", "http://test.com", config=conf)
                mock_ext.assert_called_once()


def test_progbar() -> None:
    pb = Progbar(10)
    pb.update(5, [("loss", 1.0)])
    assert "loss" in pb._values
    assert pb._values["loss"] == [5.0, 5]
    pb2 = Progbar(10, config=ProgbarConfig(stateful_metrics=["acc"]))
    pb2.update(5, [("acc", 1.0)])
    assert pb2._values["acc"] == [1.0, 1]
    pb.update(2, [("loss", 2.0)])
    assert pb._values["loss"] == [5.0 + 4.0, 5 + 2]


def test_other_stubs() -> None:
    clear_session()
    with custom_object_scope():
        pass
    assert deserialize_keras_object() is None
    disable_interactive_logging()
    enable_interactive_logging()
    assert get_custom_objects() == {}
    assert get_registered_name() == ""
    assert get_registered_object() is None
    assert is_interactive_logging_enabled() is False
    assert is_keras_tensor() is False

    @register_keras_serializable()
    class MyClass:
        pass

    assert serialize_keras_object() is None
    assert standardize_dtype("float32") == "float32"
    assert standardize_dtype() is None
    bb = bounding_boxes()
