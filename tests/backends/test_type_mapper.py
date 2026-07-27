"""Test module."""

from ml_switcheroo_compiler.backends.type_mapper import TypeMapper


def test_type_mapper():
    m1 = TypeMapper()
    assert m1.map_type("float32") == "float32"

    m2 = TypeMapper({"float32": "jnp.float32"})
    assert m2.map_type("float32") == "jnp.float32"
    assert m2.map_type("int32") == "int32"
