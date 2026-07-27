# ruff: noqa: E501


def test_h5_branches():
    import os

    import numpy as np

    from ml_switcheroo_compiler.serialization.formats.h5 import H5WeightFormat

    ser = H5WeightFormat()
    test_file = "test_h5_serialization_branches.h5"
    if os.path.exists(test_file):
        os.remove(test_file)

    class FakeTensor1:
        def numpy(self):
            return np.array([1, 2])

    class FakeTensor2:
        def __init__(self):

            class D:
                def numpy(self):
                    return np.array([3, 4])

            self.data = D()

    class FakeTensor3:
        def tolist(self):
            return [5, 6]

    weights = {"t1": FakeTensor1(), "t2": FakeTensor2(), "t3": FakeTensor3()}
    import ml_switcheroo_compiler.backends.registry as reg

    old = reg.get_active_backend

    class FakeBackend:
        pass

    reg.get_active_backend = lambda: FakeBackend()
    ser.save(weights, test_file)
    loaded = ser.load(test_file)
    assert np.array_equal(loaded["t1"], [1, 2])
    assert np.array_equal(loaded["t2"], [3, 4])
    assert np.array_equal(loaded["t3"], [5, 6])
    reg.get_active_backend = old
    if os.path.exists(test_file):
        os.remove(test_file)


def test_h5_backend_branches():
    from ml_switcheroo_compiler.serialization.formats.h5 import H5WeightFormat

    ser = H5WeightFormat()

    class FakeBackend:
        def load_h5(self, path):
            return {"a": 1}

        def save_h5(self, weights, path):
            pass

    import ml_switcheroo_compiler.backends.registry as reg

    old = reg.get_active_backend
    reg.get_active_backend = lambda: FakeBackend()
    assert ser.load("dummy.h5") == {"a": 1}
    ser.save({"a": 1}, "dummy.h5")
    reg.get_active_backend = old


def test_h5_visit_not_dataset():
    import os

    import h5py

    from ml_switcheroo_compiler.serialization.formats.h5 import H5WeightFormat

    test_file = "test_h5_group.h5"
    if os.path.exists(test_file):
        os.remove(test_file)
    with h5py.File(test_file, "w") as f:
        f.create_group("my_group")
    ser = H5WeightFormat()
    loaded = ser.load(test_file)
    assert "my_group" not in loaded
    if os.path.exists(test_file):
        os.remove(test_file)


def test_h5_save_plain():
    import os

    import numpy as np

    from ml_switcheroo_compiler.serialization.formats.h5 import H5WeightFormat

    test_file = "test_h5_plain.h5"
    if os.path.exists(test_file):
        os.remove(test_file)
    ser = H5WeightFormat()
    ser.save({"plain": np.array(5)}, test_file)
    loaded = ser.load(test_file)
    assert "plain" in loaded
    if os.path.exists(test_file):
        os.remove(test_file)


def test_h5_save_no_branches_taken():
    import os

    from ml_switcheroo_compiler.serialization.formats.h5 import H5WeightFormat

    test_file = "test_h5_plain2.h5"
    if os.path.exists(test_file):
        os.remove(test_file)
    ser = H5WeightFormat()

    class FakeNoAttribs:
        pass

    ser.save({"plain2": 42}, test_file)
    loaded = ser.load(test_file)
    assert loaded["plain2"] == 42
    if os.path.exists(test_file):
        os.remove(test_file)
