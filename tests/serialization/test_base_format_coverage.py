def test_base_format_coverage():
    from ml_switcheroo_compiler.serialization.formats.base import WeightLoader, WeightSaver

    class DummyLoader(WeightLoader):
        def load(self, filepath):
            return super().load(filepath)

    class DummySaver(WeightSaver):
        def save(self, weights_np, filepath):
            return super().save(weights_np, filepath)

    assert DummySaver().save({}, "test.txt") is None
    assert DummyLoader().load("test.txt") is None
