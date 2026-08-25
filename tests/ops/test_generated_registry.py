from unittest.mock import patch

import pytest

from ml_switcheroo_compiler.ops.generated_registry import OPS_REGISTRY
from ml_switcheroo_compiler.ops.registry import _OP_REGISTRY, get_op


def test_generated_registry_loaded():
    assert isinstance(OPS_REGISTRY, dict)
    assert "Abs" in OPS_REGISTRY


def test_dynamic_infer_shape():
    # Force the registry to fallback to dynamic class creation for a known op
    # Let's say "Add" which might not be registered yet or we can use a mock name

    # We add a fake entry to OPS_REGISTRY just for the test
    from ml_switcheroo_compiler.ops.registry import _YAML_REGISTRY

    original = _YAML_REGISTRY.get("FakeShapeOp")
    _YAML_REGISTRY["FakeShapeOp"] = {"operation": "FakeShapeOp"}

    try:
        OpCls = get_op("FakeShapeOp")
        op = OpCls()

        # 1. inputs via list of args
        class Dummy:
            pass

        d1 = Dummy()
        d1.shape_metadata = (2, 2)
        d2 = Dummy()
        d2.shape = (2, 2)

        assert op.infer_shape(d1) == (2, 2)
        assert op.infer_shape(inputs=[d2]) == (2, 2)

        # 2. list of ints
        assert op.infer_shape([[2, 3]]) == (2, 3)
        assert op.get_yaml_data() == {"operation": "FakeShapeOp"}

        # 3. broadcast
        assert op.infer_shape(d1, inputs=[d2]) == (2, 2)

        # 4. no shapes
        assert op.infer_shape(Dummy()) == ()

        # 5. Broadcast fallback
        d3 = Dummy()
        d3.shape = (3, 3)
        d4 = Dummy()
        d4.shape = (2, 2)
        with patch("numpy.broadcast_shapes", side_effect=Exception):
            assert op.infer_shape(d3, d4) == (3, 3)  # max by len, equal len means first

    finally:
        if original is None:
            del _YAML_REGISTRY["FakeShapeOp"]
        else:
            _YAML_REGISTRY["FakeShapeOp"] = original
        if "FakeShapeOp" in _OP_REGISTRY:
            del _OP_REGISTRY["FakeShapeOp"]


def test_get_all_ops():
    from ml_switcheroo_compiler.ops.registry import get_all_ops

    ops = get_all_ops()
    assert len(ops) > 0


def test_get_op_exceptions():
    with pytest.raises(KeyError):
        get_op("NonExistentOp")
    with pytest.raises(ValueError):
        get_op("TotallyMissingOpXYZ")


def test_registry_shim_methods():
    from ml_switcheroo_compiler.ops.registry import _RegistryShim, _RegistryShimFix, get_backend_mapping, get_frontend, register_frontend

    data = {"Op1": {"variants": {"np": {"generator": "np_gen", "eager": "np_eager"}}}}

    # _RegistryShim
    shim1 = _RegistryShim(data)
    assert shim1.get_generator_mapping("np", "Op1") == "np_gen"
    assert shim1.get_generator_mapping("np", "Op2") is None

    # _RegistryShimFix
    shim2 = _RegistryShimFix(data)
    assert shim2.get_generator_mapping("np", "Op1") == "np_gen"
    assert shim2.get_generator_mapping("np", "Op2") is None

    assert shim2.get_eager_mapping("np", "Op1") == "np_eager"
    assert shim2.get_eager_mapping("np", "Op2") is None

    assert shim2.get_op("Op1") == data["Op1"]

    # get_backend_mapping
    from ml_switcheroo_compiler.ops.registry import _YAML_REGISTRY

    _YAML_REGISTRY["Op1"] = data["Op1"]
    assert get_backend_mapping("Op1") == data["Op1"]["variants"]
    assert get_backend_mapping("Op2") == {}

    # frontends
    @register_frontend("TestFrontend")
    class TestFE:
        pass

    assert get_frontend("TestFrontend") is TestFE
    with pytest.raises(KeyError):
        get_frontend("MissingFE")


def test_register_op_duplicate():
    from ml_switcheroo_compiler.ops.registry import register_op

    @register_op("TestDupOp")
    class TestDup1:
        pass

    with pytest.raises(ValueError):

        @register_op("TestDupOp")
        class TestDup2:
            pass


def test_util_registry():
    from ml_switcheroo_compiler.ops.registry import get_util, register_util

    @register_util("TestUtil")
    def my_util():
        pass

    assert get_util("TestUtil") is my_util
    with pytest.raises(KeyError):
        get_util("MissingUtil")


def test_yaml_registry_missing_file():
    from ml_switcheroo_compiler.ops.registry import _load_yaml_registry

    with patch("os.path.exists", return_value=False):
        _load_yaml_registry(force=True)
