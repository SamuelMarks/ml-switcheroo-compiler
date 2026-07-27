"""Test module."""

import pytest

from ml_switcheroo_compiler.backends.eager_registry import EagerOpRegistry


def test_eager_registry():
    r = EagerOpRegistry()
    assert r.get("foo") is None

    @r.register("foo")
    def foo(x):
        return x + 1

    assert r.get("foo") is foo

    assert r.dispatch("foo", 1) == 2

    with pytest.raises(ValueError):
        r.dispatch("bar", 1)


def test_eager_registry_fallback():
    from ml_switcheroo_compiler.backends.eager_registry import global_eager_registry

    r = EagerOpRegistry()

    @global_eager_registry.register("global_foo_test")
    def gfoo(x):
        return x + 2

    assert r.dispatch("global_foo_test", 3) == 5


def test_global_eager_registry_fallback_missing():
    import pytest

    from ml_switcheroo_compiler.backends.eager_registry import global_eager_registry

    with pytest.raises(ValueError):
        global_eager_registry.dispatch("op_does_not_exist_ever_123")
