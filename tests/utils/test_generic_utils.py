import sys
from unittest import mock

import pytest

from ml_switcheroo_compiler.utils.generic_utils import (
    ArchiveConfig,
    CacheConfig,
    Config,
    CustomObjectScope,
    FeatureSpace,
    GetFileConfig,
    HashConfig,
    Progbar,
    ProgbarConfig,
    PyDataset,
    Sequence,
    _download_remote_file,
    _extract_archive,
    _validate_cache,
    bounding_boxes,
    clear_session,
    custom_Any_scope,
    deserialize_keras_Any,
    disable_interactive_logging,
    enable_interactive_logging,
    get_custom_Anys,
    get_file,
    get_registered_Any,
    get_registered_name,
    is_interactive_logging_enabled,
    is_keras_tensor,
    register_keras_serializable,
    serialize_keras_Any,
    set_random_seed,
    standardize_dtype,
)


def test_configs():
    hc = HashConfig()
    ac = ArchiveConfig()
    cc = CacheConfig()
    gc = GetFileConfig()
    pc = ProgbarConfig()
    assert hc.hash_algorithm == "auto"


def test_set_random_seed():
    import types

    m = types.ModuleType("ml_switcheroo_compiler.backends.numpy.utils")
    m.set_numpy_seed = mock.Mock()
    sys.modules["ml_switcheroo_compiler.backends.numpy.utils"] = m
    set_random_seed(42)
    del sys.modules["ml_switcheroo_compiler.backends.numpy.utils"]
    with mock.patch("random.seed", side_effect=ImportError):
        set_random_seed(42)


def test_validate_cache():
    with mock.patch("os.path.exists", return_value=True):
        assert _validate_cache("dummy")
    with mock.patch("os.path.exists", return_value=False):
        assert not _validate_cache("dummy")


def test_download_remote_file():
    with mock.patch("urllib.request.urlretrieve") as mock_url:
        _download_remote_file("http://dummy", "dummy")
        mock_url.assert_called_once()

    import urllib.error

    with mock.patch("urllib.request.urlretrieve", side_effect=urllib.error.URLError("err")):
        with pytest.raises(RuntimeError):
            _download_remote_file("http://dummy", "dummy")


def test_extract_archive():
    with mock.patch("tarfile.open") as mock_tar:
        _extract_archive("test.tar.gz", "dir")
        _extract_archive("test.tar", "dir")

    with mock.patch("zipfile.ZipFile") as mock_zip:
        _extract_archive("test.zip", "dir")

    _extract_archive("test.txt", "dir")  # nothing happens


def test_get_file():
    with mock.patch("ml_switcheroo_compiler.utils.generic_utils._validate_cache", return_value=True):
        res = get_file("test", "http://test")
        assert res.endswith("test")

    with mock.patch("ml_switcheroo_compiler.utils.generic_utils._validate_cache", return_value=False), mock.patch("ml_switcheroo_compiler.utils.generic_utils._download_remote_file"), mock.patch("ml_switcheroo_compiler.utils.generic_utils._extract_archive"):
        # Extract True
        cfg = GetFileConfig(archive_config=ArchiveConfig(extract=True))
        get_file("test", "http://test", cfg)


def test_progbar():
    pb = Progbar(10)
    pb.update(1, [("loss", 0.5)])
    assert "loss" in pb._values
    assert pb._values["loss"] == [0.5, 1]

    pb.update(2, [("loss", 0.5)])
    assert pb._values["loss"] == [1.5, 3]  # 0.5*1 + 0.5*2 = 1.5, 1+2 = 3

    pb = Progbar(10, ProgbarConfig(stateful_metrics=["acc"]))
    pb.update(1, [("acc", 0.9)])
    assert pb._values["acc"] == [0.9, 1]

    # Test finalizing
    assert pb._should_finalize(10, None)
    assert not pb._should_finalize(1, None)
    assert pb._should_finalize(1, True)

    # Test should_update
    import time

    assert pb._should_update(time.time() + 100.0, False)

    # Test format info
    assert pb._format_info(5) == " - 5/10"


def test_dummy_classes():
    assert FeatureSpace() is not None
    assert Config() is not None

    with CustomObjectScope() as scope:
        assert scope is not None

    assert PyDataset() is not None
    assert Sequence() is not None
    assert bounding_boxes() is not None


def test_utility_functions():
    clear_session()

    with custom_Any_scope() as s:
        assert s is not None

    assert deserialize_keras_Any() is None

    disable_interactive_logging()
    enable_interactive_logging()

    assert get_custom_Anys() == {}
    assert get_registered_name() == ""
    assert get_registered_Any() is None
    assert not is_interactive_logging_enabled()
    assert not is_keras_tensor()
    assert serialize_keras_Any() is None
    assert standardize_dtype("float32") == "float32"
    assert standardize_dtype() is None


def test_register_keras_serializable():
    dec = register_keras_serializable()

    @dec
    class Dummy:
        pass

    assert Dummy is not None


from ml_switcheroo_compiler.utils.generic_utils import (
    custom_object_scope,
)


def test_generic_utils_stubs() -> None:
    """Test generic utils stubs."""
    FeatureSpace()
    Config()
    PyDataset()
    Sequence()
    bounding_boxes()
    with custom_object_scope():
        pass

    @register_keras_serializable()
    class A:
        pass
