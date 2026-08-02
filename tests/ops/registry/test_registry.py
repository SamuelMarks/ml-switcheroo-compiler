import pytest

from ml_switcheroo_compiler.ops.base import OpDef
from ml_switcheroo_compiler.ops.registry import get_frontend, get_op, get_util, register_frontend, register_op, register_util


def test_registry_register_op():
    @register_op("TestRegistryOp1")
    class TestRegistryOp1(OpDef):
        def infer_shape(self):
            return ()

    assert get_op("TestRegistryOp1") == TestRegistryOp1

    # re-registration with same name and same class name is allowed
    @register_op("TestRegistryOp1")
    class TestRegistryOp1(OpDef):
        def infer_shape(self):
            return ()

    # re-registration with same name but DIFFERENT class name is ValueError
    with pytest.raises(ValueError, match="already registered"):

        @register_op("TestRegistryOp1")
        class AnotherOpClass(OpDef):
            def infer_shape(self):
                return ()


def test_registry_util():
    @register_util("my_util")
    def my_util_fn():
        return 42

    assert get_util("my_util") == my_util_fn

    with pytest.raises(KeyError, match="not found"):
        get_util("non_existent_util")


def test_registry_frontend():
    @register_frontend("my_frontend")
    def my_frontend_fn():
        return 42

    assert get_frontend("my_frontend") == my_frontend_fn

    with pytest.raises(KeyError, match="not found"):
        get_frontend("non_existent_frontend")


def test_get_op_not_found():
    with pytest.raises(KeyError, match="not found"):
        get_op("NonExistentOp")


def test_registry_files_load():
    pass
