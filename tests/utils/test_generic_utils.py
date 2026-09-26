"""Unit tests for generic and framework-agnostic utility functions."""

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
    get_file,
    is_interactive_logging_enabled,
    is_keras_tensor,
    register_keras_serializable,
    serialize_keras_Any,
    set_random_seed,
    standardize_dtype,
)


def test_configs() -> None:
    """Test configuration dataclass defaults."""
    hc = HashConfig()
    ac = ArchiveConfig()
    cc = CacheConfig()
    gc = GetFileConfig()
    pc = ProgbarConfig()
    assert hc.hash_algorithm == "auto"


def test_set_random_seed() -> None:
    """Test global random seed assignment across random and numpy modules."""
    import types

    m = types.ModuleType("ml_switcheroo_compiler.backends.numpy.utils")
    m.set_numpy_seed = mock.Mock()
    sys.modules["ml_switcheroo_compiler.backends.numpy.utils"] = m
    set_random_seed(42)
    del sys.modules["ml_switcheroo_compiler.backends.numpy.utils"]
    with mock.patch("random.seed", side_effect=ImportError):
        set_random_seed(42)


def test_validate_cache() -> None:
    """Test cache validation checking file existence."""
    with mock.patch("os.path.exists", return_value=True):
        assert _validate_cache("dummy")
    with mock.patch("os.path.exists", return_value=False):
        assert not _validate_cache("dummy")


def test_download_remote_file() -> None:
    """Test remote file download helper and error handling."""
    with mock.patch("urllib.request.urlretrieve") as mock_url:
        _download_remote_file("http://dummy", "dummy")
        mock_url.assert_called_once()

    import urllib.error

    with mock.patch("urllib.request.urlretrieve", side_effect=urllib.error.URLError("err")):
        with pytest.raises(RuntimeError):
            _download_remote_file("http://dummy", "dummy")


def test_extract_archive() -> None:
    """Test archive extraction for tar, tar.gz, and zip formats."""
    with mock.patch("tarfile.open") as mock_tar:
        _extract_archive("test.tar.gz", "dir")
        _extract_archive("test.tar", "dir")

    with mock.patch("zipfile.ZipFile") as mock_zip:
        _extract_archive("test.zip", "dir")

    _extract_archive("test.txt", "dir")  # nothing happens


def test_get_file() -> None:
    """Test file download and extraction caching utility."""
    with mock.patch("ml_switcheroo_compiler.utils.generic_utils._validate_cache", return_value=True):
        res = get_file("test", "http://test")
        assert res.endswith("test")

    with mock.patch("ml_switcheroo_compiler.utils.generic_utils._validate_cache", return_value=False), mock.patch("ml_switcheroo_compiler.utils.generic_utils._download_remote_file"), mock.patch("ml_switcheroo_compiler.utils.generic_utils._extract_archive") as mock_extract:
        # Extract True
        cfg = GetFileConfig(archive_config=ArchiveConfig(extract=True))
        get_file("test", "http://test", cfg)
        mock_extract.assert_called_once()

        # Extract False and untar False with custom cache_dir
        cfg_no_extract = GetFileConfig(
            cache_config=CacheConfig(cache_dir="/tmp/custom_cache"),
            archive_config=ArchiveConfig(extract=False, untar=False),
        )
        res_no_extract = get_file("test2", "http://test", cfg_no_extract)
        assert res_no_extract.endswith("test2")


def test_progbar() -> None:
    """Test progress bar progress tracking and metrics updates."""
    pb = Progbar(10)
    pb.update(1, [("loss", 0.5)])
    assert "loss" in pb._values
    assert pb._values["loss"] == [0.5, 1]

    pb.update(2, [("loss", 0.5)])
    assert pb._values["loss"] == [1.5, 3]  # 0.5*1 + 0.5*2 = 1.5, 1+2 = 3

    pb = Progbar(10, ProgbarConfig(stateful_metrics=["acc"]))
    pb.update(1, [("acc", 0.9)])
    assert pb._values["acc"] == [0.9, 1]

    pb_quiet = Progbar(10, ProgbarConfig(verbose=0))
    pb_quiet.update(1, [("loss", 0.5)])

    # Test finalizing
    assert pb._should_finalize(10, None)
    assert not pb._should_finalize(1, None)
    assert pb._should_finalize(1, True)

    # Test should_update
    import time

    assert pb._should_update(time.time() + 100.0, False)

    # Test format info
    assert pb._format_info(5) == " - 5/10"


def test_dummy_classes() -> None:
    """Test dummy utility compatibility classes."""
    assert FeatureSpace() is not None
    assert Config() is not None

    with CustomObjectScope() as scope:
        assert scope is not None

    assert PyDataset() is not None
    assert Sequence() is not None
    assert bounding_boxes() is not None


def test_utility_functions() -> None:
    """Test utility logging and session management functions."""
    clear_session()

    with custom_Any_scope() as s:
        assert s is not None

    assert deserialize_keras_Any() is None

    enable_interactive_logging()
    assert is_interactive_logging_enabled()
    disable_interactive_logging()
    assert not is_interactive_logging_enabled()
    assert not is_keras_tensor()
    assert serialize_keras_Any() is None
    assert standardize_dtype("float32") == "float32"
    assert standardize_dtype() is None


def test_register_keras_serializable() -> None:
    """Test registration decorator for serializable classes."""
    dec = register_keras_serializable()

    @dec
    class Dummy:
        """Dummy class."""

        pass

    assert Dummy is not None


from ml_switcheroo_compiler.utils.generic_utils import (
    custom_object_scope,
)


def test_generic_utils_stubs() -> None:
    """Test generic utils stubs and ecosystem utilities."""
    FeatureSpace()
    Config()
    PyDataset()
    Sequence()
    bounding_boxes()
    with custom_object_scope():
        pass

    @register_keras_serializable()
    class A:
        """Class A."""

        pass


def test_custom_object_scope_and_serialization() -> None:
    """Test functional custom_object_scope and serialization/deserialization."""
    from ml_switcheroo_compiler.utils.generic_utils import (
        deserialize_keras_object,
        get_custom_objects,
        get_registered_object,
        serialize_keras_object,
    )

    class CustomDense:
        """CustomDense test class."""

        def __init__(self, units: int = 32) -> None:
            """Initialize CustomDense."""
            self.units = units

        def get_config(self) -> dict[str, int]:
            """Get configuration dictionary."""
            return {"units": self.units}

        @classmethod
        def from_config(cls, config: dict[str, int]) -> "CustomDense":
            """Instantiate from config."""
            return cls(**config)

    # Scoped registration
    with custom_object_scope({"CustomDense": CustomDense}):
        assert "CustomDense" in get_custom_objects()
        assert get_registered_object("CustomDense") is CustomDense

        layer = CustomDense(units=64)
        serialized = serialize_keras_object(layer)
        assert serialized is not None
        assert serialized["class_name"] == "CustomDense"
        assert serialized["config"] == {"units": 64}

        deserialized = deserialize_keras_object(serialized)
        assert isinstance(deserialized, CustomDense)
        assert deserialized.units == 64

    # Restored outside scope
    assert "CustomDense" not in get_custom_objects()

    # Pass custom_objects explicitly to deserialize
    deserialized_explicit = deserialize_keras_object(serialized, custom_objects={"CustomDense": CustomDense})
    assert isinstance(deserialized_explicit, CustomDense)
    assert deserialized_explicit.units == 64

    # Class without from_config (uses cls(**config))
    class SimpleLayer:
        """SimpleLayer test class."""

        def __init__(self, val: int = 0) -> None:
            """Initialize SimpleLayer."""
            self.val = val

    deserialized_simple = deserialize_keras_object(
        {"class_name": "SimpleLayer", "config": {"val": 10}},
        custom_objects={"SimpleLayer": SimpleLayer},
    )
    assert isinstance(deserialized_simple, SimpleLayer)
    assert deserialized_simple.val == 10

    # Config not a dict
    cfg_not_dict = {"class_name": "SimpleLayer", "config": "invalid"}
    assert deserialize_keras_object(cfg_not_dict, custom_objects={"SimpleLayer": SimpleLayer}) == cfg_not_dict

    # Registered name and object lookups
    from ml_switcheroo_compiler.utils.generic_utils import (
        get_registered_name,
    )

    @register_keras_serializable(package="TestPkg", name="NamedCustomLayer")
    class NamedCustomLayer:
        """NamedCustomLayer test class."""

        pass

    assert get_registered_name(NamedCustomLayer) == "NamedCustomLayer"
    assert get_registered_name(None) == ""
    assert get_registered_name(lambda x: x) == "<lambda>"
    assert get_registered_name(123) == "int"

    assert get_registered_object(None) is None
    assert get_registered_object("") is None
    assert get_registered_object("NamedCustomLayer") is NamedCustomLayer
    assert get_registered_object("nonexistent_obj_123") is None

    # Fallbacks
    assert deserialize_keras_object(None) is None
    assert deserialize_keras_object(123) == 123
    assert deserialize_keras_object({"unknown": "dict"}) == {"unknown": "dict"}
    assert deserialize_keras_object({"class_name": 123}) == {"class_name": 123}
    assert deserialize_keras_object({"class_name": "UnregisteredClass"}) == {"class_name": "UnregisteredClass"}
    assert serialize_keras_object(None) is None


def test_is_keras_tensor_and_dtype() -> None:
    """Test is_keras_tensor type checking and standardize_dtype normalization."""
    from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig

    class FakeKerasHistory:
        """FakeKerasHistory test class."""

        _keras_history = ("layer", 0, 0)

    class FakeKerasFlag:
        """FakeKerasFlag test class."""

        is_keras_tensor = True

    class DummyObj:
        """DummyObj test class."""

        pass

    assert is_keras_tensor(FakeKerasHistory())
    assert is_keras_tensor(FakeKerasFlag())

    t = Tensor([1, 2], TensorConfig((2,), "float32", "cpu"))
    assert is_keras_tensor(t)

    assert not is_keras_tensor(DummyObj())
    assert not is_keras_tensor(123)
    assert not is_keras_tensor()

    # standardize_dtype
    import numpy as np

    assert standardize_dtype(np.float32) == "float32"
    assert standardize_dtype(float) == "float32"
    assert standardize_dtype(int) == "int32"
    assert standardize_dtype(bool) == "bool"
    assert standardize_dtype(np.dtype("complex128")) == "complex128"
    assert standardize_dtype("DOUBLE") == "float64"
    assert standardize_dtype("int64") == "int64"
    assert standardize_dtype("bool") == "bool"
    assert standardize_dtype("int") == "int32"
    assert standardize_dtype("custom_string_dtype") == "custom_string_dtype"
    assert standardize_dtype(None) is None
    assert standardize_dtype() is None


def test_bounding_boxes_conversions() -> None:
    """Test bounding_boxes coordinate format conversions and validations."""
    import numpy as np

    # xyxy <-> xywh
    xyxy_boxes = np.array([[10.0, 20.0, 50.0, 80.0]], dtype=np.float32)
    xywh_boxes = bounding_boxes.convert_format(xyxy_boxes, source="xyxy", target="xywh")
    np.testing.assert_allclose(xywh_boxes, [[10.0, 20.0, 40.0, 60.0]])

    # Custom target fallback and custom source validation
    with mock.patch.object(bounding_boxes, "SUPPORTED_FORMATS", ("xyxy", "custom")):
        res_custom = bounding_boxes.convert_format(xyxy_boxes, source="xyxy", target="custom")
        np.testing.assert_allclose(res_custom, xyxy_boxes)
        with pytest.raises(ValueError, match="Unrecognized source format"):
            bounding_boxes.convert_format(xyxy_boxes, source="custom", target="xyxy")

    # xywh -> center_xywh
    center_boxes = bounding_boxes.convert_format(xywh_boxes, source="xywh", target="center_xywh")
    np.testing.assert_allclose(center_boxes, [[30.0, 50.0, 40.0, 60.0]])

    # center_xywh -> yxyx
    yxyx_boxes = bounding_boxes.convert_format(center_boxes, source="center_xywh", target="yxyx")
    np.testing.assert_allclose(yxyx_boxes, [[20.0, 10.0, 80.0, 50.0]])

    # yxyx -> rel_xyxy
    rel_boxes = bounding_boxes.convert_format(yxyx_boxes, source="yxyx", target="rel_xyxy", image_shape=(100, 200))
    np.testing.assert_allclose(rel_boxes, [[0.05, 0.2, 0.25, 0.8]])

    # rel_xyxy -> xyxy
    back_to_xyxy = bounding_boxes.convert_format(rel_boxes, source="rel_xyxy", target="xyxy", image_shape=(100, 200))
    np.testing.assert_allclose(back_to_xyxy, xyxy_boxes)

    # Identity conversion
    same_boxes = bounding_boxes.convert_format(xyxy_boxes, source="xyxy", target="xyxy")
    np.testing.assert_allclose(same_boxes, xyxy_boxes)

    # Validations
    with pytest.raises(ValueError, match="Unsupported bounding box format"):
        bounding_boxes.convert_format(xyxy_boxes, source="invalid", target="xyxy")

    with pytest.raises(ValueError, match="Bounding boxes must have 4 coordinates"):
        bounding_boxes.convert_format(np.array([1.0, 2.0, 3.0]), source="xyxy", target="xywh")

    with pytest.raises(ValueError, match="image_shape .* is required"):
        bounding_boxes.convert_format(rel_boxes, source="rel_xyxy", target="xyxy")

    with pytest.raises(ValueError, match="image_shape .* is required"):
        bounding_boxes.convert_format(xyxy_boxes, source="xyxy", target="rel_xyxy")


def test_framework_agnostic_serialization_separation() -> None:
    """Test framework-agnostic serialization module functionality and strict decoupling."""
    from ml_switcheroo_compiler.utils.generic_utils import is_symbolic_tensor
    from ml_switcheroo_compiler.utils.serialization_utils import (
        CustomObjectScope,
        deserialize_object,
        get_registered_name,
        get_registered_object,
        register_serializable,
        serialize_object,
    )

    @register_serializable(package="CoreModule", name="AgnosticLayer")
    class AgnosticLayer:
        """Agnostic layer for serialization testing."""

        def __init__(self, size: int = 128) -> None:
            """Initialize AgnosticLayer."""
            self.size = size

        def get_config(self) -> dict[str, int]:
            """Get configuration dictionary."""
            return {"size": self.size}

        @classmethod
        def from_config(cls, config: dict[str, int]) -> "AgnosticLayer":
            """Create instance from configuration."""
            return cls(**config)

    assert get_registered_name(AgnosticLayer) == "AgnosticLayer"
    assert get_registered_object("AgnosticLayer") is AgnosticLayer

    layer = AgnosticLayer(size=256)
    serialized = serialize_object(layer)
    assert serialized is not None
    assert serialized["class_name"] == "AgnosticLayer"
    assert serialized["config"] == {"size": 256}

    deserialized = deserialize_object(serialized)
    assert isinstance(deserialized, AgnosticLayer)
    assert deserialized.size == 256

    # Test CustomObjectScope context manager enter and exit
    with CustomObjectScope({"TempLayer": AgnosticLayer}):
        assert get_registered_object("TempLayer") is AgnosticLayer
    assert get_registered_object("TempLayer") is None

    # Test is_symbolic_tensor
    from ml_switcheroo_compiler.tracing.tracer import ProxyTensor

    proxy = ProxyTensor("p1", (2, 2), "float32")
    assert is_symbolic_tensor(proxy)
