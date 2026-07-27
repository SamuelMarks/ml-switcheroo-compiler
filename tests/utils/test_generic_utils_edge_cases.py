"""Test generic utils stubs."""

from ml_switcheroo_compiler.utils.generic_utils import (
    Config,
    FeatureSpace,
    PyDataset,
    Sequence,
    bounding_boxes,
    custom_object_scope,
    register_keras_serializable,
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
